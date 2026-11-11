from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .estimator import Estimate

Candidates = Dict[Tuple[int, int], Estimate]


@dataclass
class Choice:
    p: int
    b: int
    feasible: bool              # the choice satisfies Equation 1
    overload: bool = False      # not even the highest throughput keeps up with lambda


def feasible(est: Estimate, lam: float, slo: float, margin: float = 0.0) -> bool:
    return est.throughput >= (1.0 + margin) * lam and est.latency <= (1.0 - margin) * slo


def select_b(cands: Candidates, p: int, lam: float, slo: float) -> Choice:
    under_p = {b: est for (pp, b), est in cands.items() if pp == p}
    ok = [(est.latency, b) for b, est in under_p.items() if feasible(est, lam, slo)]
    if ok:
        return Choice(p, min(ok)[1], True)
    keep_up = [(est.latency, b) for b, est in under_p.items() if est.throughput >= lam]
    if keep_up:
        return Choice(p, min(keep_up)[1], False)
    return Choice(p, max(under_p.items(), key=lambda kv: (kv[1].throughput, -kv[1].latency))[0], False, True)


def config_opt(cands: Candidates, lam: float, slo: float, p_max: int) -> Choice:
    ok = [(p, est.latency, b) for (p, b), est in cands.items() if feasible(est, lam, slo)]
    if ok:
        p, _, b = min(ok)
        return Choice(p, b, True)
    return select_b(cands, min(p_max, max(p for p, _ in cands)), lam, slo)


@dataclass
class Decision:
    p: int
    b: int
    rescale: bool          # p changes: savepoint and restart
    change_b: bool         # only b changes: broadcast, no restart
    overload: bool = False
    target: Optional[Choice] = None     # (p*, b*) of this window


class ScalingTrigger:

    def __init__(self, p: int, b: int, p_max: int, delta: float):
        self.p, self.b = p, b
        self.p_max, self.delta = p_max, delta

    def decide(self, cands: Candidates, lam: float, slo: float) -> Optional[Decision]:
        if not cands:
            return None
        star = config_opt(cands, lam, slo, self.p_max)                  # line 1
        p, b = self.p, self.b
        if star.p == p:                                                 # lines 2-3
            b = star.b
        elif star.p > p:                                                # lines 4-5: scale out
            p, b = star.p, star.b
        elif feasible(cands[(star.p, star.b)], lam, slo, self.delta):   # lines 6-7: scale in
            p, b = star.p, star.b
        else:                                                           # lines 8-9
            cur = cands.get((p, b))
            if cur is None or not feasible(cur, lam, slo):
                b = select_b(cands, p, lam, slo).b
        dec = Decision(p, b, rescale=p != self.p, change_b=p == self.p and b != self.b,
                       overload=star.overload, target=star)
        self.p, self.b = p, b
        return dec
