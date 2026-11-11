import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infertuner.length_predictor import GenerationParams, LengthPredictor  # noqa: E402

log = logging.getLogger("engine")


class Engine:
    def __init__(self, a):
        self.params = GenerationParams(a.temperature, a.top_p, a.top_k, a.repetition_penalty, a.enable_thinking)
        self.max_new_tokens = a.max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok_lock = threading.Lock()
        dtype = getattr(torch, a.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="cuda").eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.predictor = None
        self.predict_lock = threading.Lock()
        if a.length_model:
            self.predictor = LengthPredictor(a.length_model, a.encoder, a.model, self.params, device="cuda")
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.ready = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _chat(self, prompts):
        return [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=self.params.enable_thinking) for p in prompts]

    def predict(self, prompts):
        with self.tok_lock:
            n_prompt = [len(ids) for ids in self.tok(self._chat(prompts))["input_ids"]]
        if self.predictor is None:
            raise RuntimeError("the server was started without --length-model")
        with self.predict_lock:
            pred = self.predictor.predict(prompts)
        # generation stops at max_new_tokens, so longer predictions are capped
        return {"prompt_tokens": n_prompt, "pred_tokens": [min(x, self.max_new_tokens) for x in pred]}

    def _count(self, row) -> int:
        for k, t in enumerate(row):
            if int(t) in self.eos:
                return k + 1
        return len(row)

    def _generate(self, prompts, max_new_tokens=None):
        t0 = time.perf_counter()
        with self.tok_lock:
            inputs = self.tok(self._chat(prompts), return_tensors="pt", padding=True).to("cuda")
        p = self.params
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=True, temperature=p.temperature, top_p=p.top_p, top_k=p.top_k,
                                      repetition_penalty=p.repetition_penalty, pad_token_id=self.tok.pad_token_id)
        torch.cuda.synchronize()
        padded = int(inputs["input_ids"].shape[1])
        gen = out[:, padded:].tolist()
        counts = [self._count(row) for row in gen]
        with self.tok_lock:
            texts = self.tok.batch_decode([row[:c] for row, c in zip(gen, counts)], skip_special_tokens=True)
        return {"texts": texts, "output_tokens": counts, "prompt_tokens": padded,
                "engine_ms": (time.perf_counter() - t0) * 1e3}

    def _worker(self):
        while True:
            prompts, fut, enq = self.jobs.get()
            try:
                res = self._generate(prompts)
                res["queue_ms"] = (time.perf_counter() - enq) * 1e3 - res["engine_ms"]
                fut.set_result(res)
            except Exception as e:      # reported to the Flink task, which retries
                fut.set_exception(e)

    def generate(self, prompts):
        fut: Future = Future()
        self.jobs.put((prompts, fut, time.perf_counter()))
        return fut.result()

    def flush(self) -> int:
        n = 0
        while True:
            try:
                _, fut, _ = self.jobs.get_nowait()
            except queue.Empty:
                return n
            fut.set_exception(RuntimeError("dropped by /flush"))
            n += 1

    def warm_up(self, sizes=(1, 8, 64)):
        for n in sizes:
            self._generate(["Hello."] * n, max_new_tokens=8)
        if self.predictor is not None:
            self.predictor.predict(["Hello."])
        self.ready = True
        log.info("engine ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--length-model", help="directory with mlp.pt and the PCA models")
    ap.add_argument("--encoder", help="local path of BAAI/bge-m3")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--enable-thinking", action="store_true", help="must match profiling and length-model metadata")
    ap.add_argument("--warmup-sizes", default="1,8,64", help="comma-separated warm-up batch sizes")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = Engine(a)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(200, {"ready": engine.ready}) if self.path == "/health" else self._json(404, {})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/predict":
                    self._json(200, engine.predict(req["prompts"]))
                elif self.path == "/generate":
                    self._json(200, engine.generate(req["prompts"]))
                elif self.path == "/flush":
                    self._json(200, {"dropped": engine.flush()})
                else:
                    self._json(404, {})
            except Exception as e:
                log.exception("request to %s failed", self.path)
                self._json(500, {"error": str(e)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine.warm_up(tuple(int(n) for n in a.warmup_sizes.split(",")))
    threading.Event().wait()


if __name__ == "__main__":
    main()
