import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infertuner.length_predictor import build_mlp, masked_pool  # noqa: E402


def embed(texts, enc_tok, encoder, device, batch=32):
    cls, pool, lens = [], [], []
    for i in range(0, len(texts), batch):
        inp = enc_tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, return_length=True)
        inp.pop("length")
        lens += inp["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            h = encoder(**{k: v.to(device) for k, v in inp.items()}).last_hidden_state
        cls.append(h[:, 0, :].float().cpu().numpy())
        pool.append(masked_pool(h, inp["attention_mask"].to(device)).float().cpu().numpy())
    return np.vstack(cls), np.vstack(pool), np.array(lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--llm-tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.data)
    thinking = bool(df["enable_thinking"].iloc[0]) if "enable_thinking" in df else False
    if "enable_thinking" in df and df["enable_thinking"].nunique() != 1:
        raise ValueError("Train separate predictors for different thinking modes")
    llm_tok = AutoTokenizer.from_pretrained(a.llm_tokenizer)
    texts = [llm_tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking) for p in df["prompt"]]
    enc_tok = AutoTokenizer.from_pretrained(a.encoder)
    encoder = AutoModel.from_pretrained(a.encoder).to(device).eval()
    cls, pool, lens = embed(texts, enc_tok, encoder, device)
    lens = np.array([len(ids) for ids in llm_tok(texts, add_special_tokens=False)["input_ids"]])

    if len(df) < 10:
        raise ValueError("At least 10 labeled samples are required for training and validation")
    # Split by prompt before fitting PCA: repeated profiling observations of a
    # prompt must not leak into the validation set.
    rng = np.random.default_rng(0)
    unique = rng.permutation(df["prompt"].unique())
    if len(unique) < 5:
        raise ValueError("At least 5 distinct prompts are required")
    held_out = set(unique[:max(1, len(unique) // 5)])
    te = np.flatnonzero(df["prompt"].isin(held_out).to_numpy())
    tr = np.flatnonzero(~df["prompt"].isin(held_out).to_numpy())
    if a.pca_dim > min(len(tr), cls.shape[1]):
        raise ValueError("pca-dim exceeds available training samples or embedding dimensions")

    pca_cls = PCA(a.pca_dim, random_state=0).fit(cls[tr])
    pca_pool = PCA(a.pca_dim, random_state=0).fit(pool[tr])
    extra = np.column_stack([lens, df["temperature"], df["top_p"], df["top_k"], df["repetition_penalty"]])
    X = np.hstack([pca_cls.transform(cls), pca_pool.transform(pool), extra]).astype(np.float32)
    y = df["output_tokens"].to_numpy(np.float32)

    mlp = build_mlp(X.shape[1]).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.L1Loss()
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    for ep in range(a.epochs):
        rng.shuffle(tr)
        mlp.train()
        for i in range(0, len(tr), 128):
            bi = torch.from_numpy(tr[i:i + 128]).to(device)
            if bi.numel() < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(mlp(Xt[bi]).squeeze(1), yt[bi])
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            mae = loss_fn(mlp(Xt[torch.from_numpy(te).to(device)]).squeeze(1), yt[torch.from_numpy(te).to(device)])
        print(f"epoch {ep + 1:3d}  test MAE={mae.item():.1f} tokens")

    torch.save(mlp.state_dict(), os.path.join(a.out, "mlp.pt"))
    joblib.dump(pca_cls, os.path.join(a.out, "pca_cls.joblib"))
    joblib.dump(pca_pool, os.path.join(a.out, "pca_pool.joblib"))
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump({"feature_version": 2, "enable_thinking": thinking, "encoder": a.encoder,
                   "llm_tokenizer": a.llm_tokenizer, "train_samples": len(tr), "validation_samples": len(te),
                   "validation_mae_tokens": float(mae.item()), "pca_dim": a.pca_dim,
                   "split": "distinct prompts; PCA fit only on training split", "seed": 0}, f, indent=2)


if __name__ == "__main__":
    main()
