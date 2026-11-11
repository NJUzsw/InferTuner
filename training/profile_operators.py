import argparse
import csv
import json
import os
import random
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

OP_PATTERNS = {
    "embedding": "model.embed_tokens",
    "rotary_embedding": "model.rotary_emb",
    "rmsnorm": "model.norm",
    "lm_head": "lm_head",
}
LAYER_PATTERNS = {"attention": "self_attn", "mlp": "mlp",
                  "input_layernorm": "input_layernorm", "post_attention_layernorm": "post_attention_layernorm"}
COLUMNS = {
    "embedding": ["batch_size", "token_idx", "prompt_token_num", "post_token_num"],
    "rotary_embedding": ["batch_size", "token_idx", "prompt_token_num", "post_token_num"],
    "lm_head": ["batch_size", "token_idx", "prompt_token_num", "post_token_num"],
    "rmsnorm": ["batch_size", "token_idx", "norm_idx", "prompt_token_num", "post_token_num"],
    "attention": ["batch_size", "token_idx", "layer_idx", "prompt_token_num", "post_token_num"],
    "mlp": ["batch_size", "token_idx", "layer_idx", "prompt_token_num", "post_token_num"],
    "input_layernorm": ["batch_size", "token_idx", "norm_idx", "prompt_token_num", "post_token_num"],
    "post_attention_layernorm": ["batch_size", "token_idx", "norm_idx", "prompt_token_num", "post_token_num"],
}


class Labeler:

    def __init__(self, model):
        self.token_idx, self.ctx = 0, {}
        self.handles = []
        modules = dict(model.named_modules())
        num_layers = model.config.num_hidden_layers
        for op, name in OP_PATTERNS.items():
            if name in modules:
                idx = 2 * num_layers if op == "rmsnorm" else -1
                self._hook(modules[name], op, idx)
        for i in range(num_layers):
            for op, name in LAYER_PATTERNS.items():
                key = f"model.layers.{i}.{name}"
                if key in modules:
                    idx = {"input_layernorm": 2 * i, "post_attention_layernorm": 2 * i + 1}.get(op, i)
                    self._hook(modules[key], op, idx)
        # one forward of the whole model = one iteration (token index 0 is the prefill)
        self.handles.append(model.register_forward_hook(lambda *_: self._next_token()))

    def _next_token(self):
        self.token_idx += 1

    def _hook(self, module, op, idx):
        def pre(mod, args):
            torch.cuda.synchronize()
            rf = record_function(f"op|{op}|{idx}|{self.token_idx}")
            rf.__enter__()
            self.ctx[id(mod)] = rf

        def post(mod, args, out):
            torch.cuda.synchronize()
            self.ctx.pop(id(mod)).__exit__(None, None, None)

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def remove(self):
        for h in self.handles:
            h.remove()


def count_output(row, eos):
    for k, t in enumerate(row):
        if int(t) in eos:
            return k + 1
    return len(row)


def generate(model, tok, prompts, a, eos, seed=0):
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=a.enable_thinking)
             for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature,
                             top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.repetition_penalty,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    padded = int(inputs["input_ids"].shape[1])
    lens = [count_output(row, eos) for row in out[:, padded:].tolist()]
    return t2 - t0, t1 - t0, padded, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 48, 64])
    ap.add_argument("--batches-per-size", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype), device_map="cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tok.eos_token_id}
    prompts = [line.strip() for line in open(a.prompts) if line.strip()]
    random.seed(0)

    writers, files = {}, []
    for op, cols in COLUMNS.items():
        f = open(os.path.join(a.out, f"{op}.csv"), "w", newline="")
        files.append(f)
        writers[op] = csv.writer(f)
        writers[op].writerow(cols + ["time_us"])
    host_f = open(os.path.join(a.out, "host.csv"), "w", newline="")
    host_w = csv.writer(host_f)
    host_w.writerow(["batch_size", "prompt_token_num", "time_us"])
    len_f = open(os.path.join(a.out, "lengths.csv"), "w", newline="")
    len_w = csv.writer(len_f)
    len_w.writerow(["prompt", "output_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "enable_thinking"])
    files += [host_f, len_f]

    generate(model, tok, prompts[:1], a, eos)          # warm-up
    calibration = {}
    for n in a.batch_sizes:
        batches = [random.sample(prompts, n) for _ in range(a.batches_per_size)]
        plain = []
        for k, B in enumerate(batches):                 # without profiling
            dur, host, padded, lens = generate(model, tok, B, a, eos, seed=k)
            plain.append(dur)
            host_w.writerow([n, padded, host * 1e6])
            len_w.writerows([[p, o, a.temperature, a.top_p, a.top_k, a.repetition_penalty, a.enable_thinking] for p, o in zip(B, lens)])
        labeler, profiled = Labeler(model), []
        for k, B in enumerate(batches):                 # under the profiler, same seeds
            labeler.token_idx = 0
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                dur, _, padded, _ = generate(model, tok, B, a, eos, seed=k)
            profiled.append(dur)
            for ev in prof.events():
                if not ev.name.startswith("op|"):
                    continue
                _, op, idx, j = ev.name.split("|")
                j, idx = int(j), int(idx)
                post = padded + j                       # context length: prompt plus generated tokens
                t = ev.cpu_time_total                   # range duration; the GPU is synchronized at both ends
                writers[op].writerow([n, j] + ([idx] if idx >= 0 else []) + [padded, post, t])
        labeler.remove()
        calibration[n] = (sum(profiled) / len(profiled)) / (sum(plain) / len(plain))
        print(f"batch size {n}: calibration factor {calibration[n]:.3f}", flush=True)
    for f in files:
        f.close()
    with open(os.path.join(a.out, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)


if __name__ == "__main__":
    main()
