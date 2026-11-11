import math
import json
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def masked_pool(hidden, attention_mask):
    mask = attention_mask[:, 1:].unsqueeze(-1).to(hidden.dtype)
    return (hidden[:, 1:, :] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def build_mlp(input_dim: int, dropout: float = 0.3):
    import torch.nn as nn
    layers, dims = [], [input_dim, 1024, 512, 256, 128, 64]
    for i in range(5):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


@dataclass
class GenerationParams:
    temperature: float
    top_p: float
    top_k: int = 40
    repetition_penalty: float = 1.15
    enable_thinking: bool = False


class LengthPredictor:

    def __init__(self, model_dir: str, encoder_dir: str, tokenizer_dir: str, params: GenerationParams,
                 device: str = "cuda"):
        import joblib
        import torch
        from transformers import AutoModel, AutoTokenizer
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError("Missing feature metadata; retrain the length predictor with the current training script")
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("feature_version") != 2 or metadata.get("enable_thinking") != params.enable_thinking:
            raise ValueError("Length predictor feature version or thinking mode does not match this deployment")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.torch, self.device, self.params = torch, device, params
        self.enc_tok = AutoTokenizer.from_pretrained(encoder_dir)
        self.encoder = AutoModel.from_pretrained(encoder_dir).to(device).eval()
        self.llm_tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.pca_cls = joblib.load(os.path.join(model_dir, "pca_cls.joblib"))
        self.pca_pool = joblib.load(os.path.join(model_dir, "pca_pool.joblib"))
        dim = self.pca_cls.n_components_ + self.pca_pool.n_components_ + 5
        self.mlp = build_mlp(dim)
        self.mlp.load_state_dict(torch.load(os.path.join(model_dir, "mlp.pt"), map_location=device, weights_only=True))
        self.mlp.to(device).eval()

    def features(self, prompts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        texts = [self.llm_tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=self.params.enable_thinking) for p in prompts]
        inputs = self.enc_tok(texts, return_tensors="pt", padding=True, truncation=True, return_length=True)
        inputs.pop("length")
        lengths = [len(ids) for ids in self.llm_tok(texts, add_special_tokens=False)["input_ids"]]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            h = self.encoder(**inputs).last_hidden_state
        cls = self.pca_cls.transform(h[:, 0, :].float().cpu().numpy())
        pool = self.pca_pool.transform(masked_pool(h, inputs["attention_mask"]).float().cpu().numpy())
        g = self.params
        extra = np.array([[n, g.temperature, g.top_p, g.top_k, g.repetition_penalty] for n in lengths])
        return np.hstack([cls, pool, extra]).astype(np.float32)

    def predict(self, prompts: Sequence[str]) -> List[int]:
        torch = self.torch
        x = torch.from_numpy(self.features(prompts)).to(self.device)
        with torch.no_grad():
            y = self.mlp(x).cpu().numpy().ravel()
        return [max(1, math.ceil(v)) for v in y]
