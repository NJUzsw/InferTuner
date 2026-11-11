from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


class IterationTimeModel(Protocol):
    def bundle_times(self, n: np.ndarray, padded: np.ndarray, iters: np.ndarray) -> np.ndarray:
        ...


@dataclass
class Event:
    arrival: float          # time the event joins a bundle in the LLM inference function (s)
    prompt_len: int         # number of prompt tokens
    output_len: int         # predicted output length o_i (tokens)
    other: float = 0.0      # measured latency outside the bundle wait, queueing, and generation (s)


def bundle_time(output_lens: Sequence[int], prompt_lens: Sequence[int], model: IterationTimeModel) -> float:
    if len(output_lens) == 0:
        return 0.0
    n = np.array([len(output_lens)])
    padded = np.array([float(max(prompt_lens))])
    iters = np.array([max(int(max(output_lens)) - 1, 0)])
    return float(model.bundle_times(n, padded, iters)[0])


@dataclass
class Estimate:
    throughput: float      # T_hat(p, b), events per second
    latency: float         # L_hat(p, b), seconds


@dataclass
class Bundles:
    of_event: np.ndarray   # bundle index of each event
    instance: np.ndarray   # instance of each bundle
    ready: np.ndarray      # time at which each bundle is full
    size: np.ndarray       # number of events of the window in each bundle


def form_bundles(arrival: np.ndarray, p: int, b: int, window: float) -> Bundles:
    N = arrival.size
    k = np.arange(N)
    m = (N - np.arange(p) + p - 1) // p                   # events per instance
    nb = -(-m // b)                                       # bundles per instance
    offset = np.cumsum(nb) - nb
    of_event = offset[k % p] + (k // p) // b
    inst = np.repeat(np.arange(p), nb)
    j = np.arange(int(nb.sum())) - offset[inst]           # bundle index within its instance
    last = np.minimum((j + 1) * b, m[inst]) - 1           # position of its last event on the instance
    size = last - j * b + 1
    gap = window / np.maximum(m[inst], 1)                 # mean inter-arrival time on the instance
    ready = arrival[inst + last * p] + (b - size) * gap
    return Bundles(of_event, inst, ready, size)


def completion_times(ready: np.ndarray, service: np.ndarray, instance: Optional[np.ndarray] = None) -> np.ndarray:
    if instance is None or instance.size == 0:
        c = np.cumsum(service)
        return c + np.maximum.accumulate(ready - (c - service))
    start = np.r_[True, instance[1:] != instance[:-1]]
    seg = np.cumsum(start) - 1                              # segment index of each bundle
    c = np.cumsum(service)
    c -= (c - service)[start][seg]                          # prefix sums restart at each instance
    x = ready - (c - service)
    # a running maximum that restarts at each segment: shift later segments above earlier ones
    shift = (np.abs(x).max() + 1.0) * 2.0 * seg
    return c + np.maximum.accumulate(x + shift) - shift


class Estimator:

    def __init__(self, model: IterationTimeModel, quantile: float = 0.99):
        self.model = model
        self.quantile = quantile

    @staticmethod
    def _arrays(events: Sequence[Event]):
        ev = sorted(events, key=lambda e: e.arrival)
        t0 = ev[0].arrival if ev else 0.0           # relative times keep the prefix sums precise
        return (np.array([e.arrival - t0 for e in ev], dtype=np.float64),
                np.array([e.prompt_len for e in ev], dtype=np.float64),
                np.array([e.output_len for e in ev], dtype=np.int64),
                np.array([e.other for e in ev], dtype=np.float64))

    def _estimate(self, arrival, prompt, out, other, p, b, window) -> Tuple[Estimate, np.ndarray, np.ndarray]:
        B = form_bundles(arrival, p, b, window)
        nb = B.ready.size
        padded = np.zeros(nb)
        np.maximum.at(padded, B.of_event, prompt)
        longest = np.zeros(nb, dtype=np.int64)
        np.maximum.at(longest, B.of_event, out)
        # Every bundle runs with b events. The missing events of a tail bundle are
        # unknown, so only its known events set P and the longest output.
        n = np.full(nb, b)
        service = np.asarray(self.model.bundle_times(n, padded, np.maximum(longest - 1, 0)), dtype=np.float64)
        end = completion_times(B.ready, service, B.instance)
        latency = other + end[B.of_event] - arrival
        mean_s = float(service.mean())
        est = Estimate(throughput=p * b / mean_s if mean_s > 0 else float("inf"),
                       latency=float(np.quantile(latency, self.quantile)))
        return est, latency, service

    def estimate(self, events: Sequence[Event], p: int, b: int, window: float) -> Estimate:
        return self._estimate(*self._arrays(events), p, b, window)[0]

    def estimate_all(self, events: Sequence[Event], parallelisms: Sequence[int], bundle_sizes: Sequence[int],
                     window: float) -> Dict[Tuple[int, int], Estimate]:
        if not events:
            return {}
        arrays = self._arrays(events)
        return {(p, b): self._estimate(*arrays, p, b, window)[0] for b in bundle_sizes for p in parallelisms}
