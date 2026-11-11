from dataclasses import dataclass
from typing import Sequence


@dataclass
class Config:
    # End-to-end latency bound of the SLO, t_SLO (Section II-A).
    slo: float = 6.0
    # Candidate sets P = {1, ..., p_max} and B of Equation 1. p_max is the number of GPUs.
    p_max: int = 20
    bundle_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 48, 64)
    # Monitoring window W.
    window: float = 300.0
    # Margin for scaling in (Algorithm 1), on both constraints of Equation 1.
    delta: float = 0.1
    # The latency estimate is the p99 of the per-event end-to-end latencies.
    latency_quantile: float = 0.99
    # Threshold for resizing the lookup cache (Section VI-A).
    epsilon: float = 0.08

    @property
    def parallelisms(self) -> Sequence[int]:
        return range(1, self.p_max + 1)
