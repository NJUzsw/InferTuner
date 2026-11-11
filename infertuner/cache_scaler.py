from typing import Iterable, Optional, Tuple

import numpy as np


def target_hit_rate(t_hit: float, t_miss: float, lam_instance: float) -> float:
    if lam_instance <= 0:
        return 0.0
    if t_miss <= t_hit:
        return 0.0
    h = (t_miss - 1.0 / lam_instance) / (t_miss - t_hit)
    return float(min(1.0, max(0.0, h)))


def required_capacity(counts: Iterable[int], h_target: float, key_size: float, value_size: float) -> Tuple[int, float]:
    f = np.sort(np.fromiter(counts, dtype=np.float64))[::-1]
    if f.size == 0 or h_target <= 0:
        return 0, 0.0
    share = np.cumsum(f) / f.sum()
    d = int(np.searchsorted(share, h_target - 1e-12) + 1)
    d = min(d, int(f.size))
    return d, d * (key_size + value_size)


class CacheScaler:
    def __init__(self, epsilon: float, min_bytes: float, max_bytes: float):
        self.epsilon, self.min_bytes, self.max_bytes = epsilon, min_bytes, max_bytes

    def update(self, counts: Iterable[int], t_hit: float, t_miss: float, lam_instance: float,
               key_size: float, value_size: float, current_bytes: float) -> Optional[float]:
        h = target_hit_rate(t_hit, t_miss, lam_instance)
        _, s = required_capacity(counts, h, key_size, value_size)
        s = float(min(self.max_bytes, max(self.min_bytes, s)))
        if current_bytes > 0 and abs(s - current_bytes) / current_bytes <= self.epsilon:
            return None
        return s
