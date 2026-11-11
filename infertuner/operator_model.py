import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Operators of one decoder layer, and operators that run once per iteration.
LAYER_OPS = ("input_layernorm", "attention", "post_attention_layernorm", "mlp")
ITER_OPS = ("embedding", "rotary_embedding", "rmsnorm", "lm_head")


@dataclass
class ModelSpec:
    num_layers: int            # number of decoder layers
    final_norm_idx: int        # norm_idx of the final RMSNorm (2 * num_layers in the profiled models)


def context_length(prompt_len, j):
    return prompt_len + j


class OperatorTimeModel:

    def __init__(self, models: Dict[str, object], spec: ModelSpec, calibration: Dict[int, float],
                 host_model: object = None, prompt_bucket: int = 32):
        self.models, self.spec, self.host_model = models, spec, host_model
        self.calibration = calibration
        self.prompt_bucket = prompt_bucket          # padded prompt lengths are bucketed to reuse predictions
        self._cum: Dict[Tuple[int, int], np.ndarray] = {}     # cumulative decode times per (n, bucket)
        self._fixed: Dict[Tuple[int, int], float] = {}        # t_pf + t_host per (n, bucket)

    @classmethod
    def load(cls, model_dir: str, spec: ModelSpec) -> "OperatorTimeModel":
        import joblib
        models = {op: joblib.load(os.path.join(model_dir, f"{op}.joblib")) for op in LAYER_OPS + ITER_OPS}
        host_path = os.path.join(model_dir, "host.joblib")
        host = joblib.load(host_path) if os.path.exists(host_path) else None
        with open(os.path.join(model_dir, "calibration.json")) as f:
            calibration = {int(k): float(v) for k, v in json.load(f).items()}
        return cls(models, spec, calibration, host)

    def _factor(self, n: int) -> float:
        if n in self.calibration:
            return self.calibration[n]
        keys = np.array(sorted(self.calibration))
        return self.calibration[int(keys[np.argmin(np.abs(keys - n))])] if keys.size else 1.0

    def _features(self, n: int, j: np.ndarray, prompt_len: int) -> Dict[str, np.ndarray]:
        L = self.spec.num_layers
        post = context_length(prompt_len, j)
        full = lambda x, v: np.full_like(x, v)
        rows = {}
        it = np.stack([full(j, n), j, full(j, prompt_len), post], axis=1)
        for op in ("embedding", "rotary_embedding", "lm_head"):
            rows[op] = it
        rows["rmsnorm"] = np.stack([full(j, n), j, full(j, self.spec.final_norm_idx), full(j, prompt_len), post], axis=1)
        jj, ll = np.repeat(j, L), np.tile(np.arange(L), j.size)
        pp = np.repeat(post, L)
        base = lambda idx: np.stack([full(jj, n), jj, idx, full(jj, prompt_len), pp], axis=1)
        rows["attention"] = base(ll)
        rows["mlp"] = base(ll)
        rows["input_layernorm"] = base(2 * ll)
        rows["post_attention_layernorm"] = base(2 * ll + 1)
        return rows

    def iteration_times(self, n: int, j: np.ndarray, prompt_len: int) -> np.ndarray:
        j = np.asarray(j, dtype=np.int64)
        rows = self._features(n, j, prompt_len)
        L = self.spec.num_layers
        total = np.zeros(j.size)
        for op, X in rows.items():
            t = np.asarray(self.models[op].predict(X), dtype=np.float64)
            total += t.reshape(j.size, L).sum(axis=1) if op in LAYER_OPS else t
        return total * 1e-6 / self._factor(n)            # profiled in microseconds

    def _fixed_time(self, n: int, pb: int) -> float:
        key = (n, pb)
        if key not in self._fixed:
            t = float(self.iteration_times(n, np.array([0]), pb)[0])
            if self.host_model is not None:
                t += float(self.host_model.predict(np.array([[n, pb]]))[0]) * 1e-6
            self._fixed[key] = t
        return self._fixed[key]

    def _decode_prefix(self, n: int, pb: int, iters: int) -> np.ndarray:
        key = (n, pb)
        cum = self._cum.get(key, np.zeros(1))
        if cum.size <= iters:
            grow = max(iters + 1, 2 * cum.size, 256)
            j = np.arange(cum.size, grow)
            cum = np.concatenate([cum, cum[-1] + np.cumsum(self.iteration_times(n, j, pb))])
            self._cum[key] = cum
        return cum

    def bundle_times(self, n: np.ndarray, padded: np.ndarray, iters: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=np.int64)
        iters = np.asarray(iters, dtype=np.int64)
        pb = np.maximum(1, np.round(np.asarray(padded, dtype=np.float64) / self.prompt_bucket)).astype(np.int64)
        pb *= self.prompt_bucket
        out = np.empty(n.size)
        keys, inverse = np.unique(np.stack([n, pb], axis=1), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        for g, (nk, bk) in enumerate(keys):
            sel = inverse == g
            cum = self._decode_prefix(int(nk), int(bk), int(iters[sel].max()))
            out[sel] = self._fixed_time(int(nk), int(bk)) + cum[iters[sel]]
        return out


class TableModel:

    def __init__(self, a=0.015, c=0.0001, e=2e-8, f=0.02, g=1e-5, h=1e-4):
        self.a, self.c, self.e, self.f, self.g, self.h = a, c, e, f, g, h

    def bundle_times(self, n: np.ndarray, padded: np.ndarray, iters: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=np.float64)
        P = np.asarray(padded, dtype=np.float64)
        J = np.asarray(iters, dtype=np.float64)
        # sum_{j=1}^{J} (a + c n + e n (P + j)) = J (a + c n + e n P) + e n J (J + 1) / 2
        decode = J * (self.a + self.c * n + self.e * n * P) + self.e * n * J * (J + 1) / 2
        return self.f + self.g * n * P + self.h * n + decode
