import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infertuner import (Estimate, Estimator, Event, ScalingTrigger, bundle_time, config_opt,  # noqa: E402
                        form_bundles, required_capacity, target_hit_rate)


class ConstModel:

    def bundle_times(self, n, padded, iters):
        return 0.5 + np.asarray(iters, dtype=float)


def cands(table):
    return {k: Estimate(*v) for k, v in table.items()}


class EstimatorTest(unittest.TestCase):
    def test_longest_output_sets_the_bundle_time(self):
        # The prefill generates the first token, so 5 tokens need 4 decode iterations.
        self.assertAlmostEqual(bundle_time([3, 5, 1], [10, 10, 10], ConstModel()), 4.5)

    def test_round_robin_then_bundles_in_arrival_order(self):
        B = form_bundles(np.arange(8, dtype=float), p=2, b=2, window=8.0)
        groups = [sorted(np.flatnonzero(B.of_event == k).tolist()) for k in range(B.ready.size)]
        self.assertEqual(groups, [[0, 2], [4, 6], [1, 3], [5, 7]])

    def test_queueing_and_bundle_wait(self):
        ev = [Event(0.0, 1, 2) for _ in range(4)]
        est = Estimator(ConstModel(), quantile=1.0).estimate_all(ev, [1, 4], [1], window=1.0)
        self.assertAlmostEqual(est[(1, 1)].latency, 6.0)            # four bundles of 1.5 s in turn
        self.assertAlmostEqual(est[(4, 1)].latency, 1.5)
        self.assertAlmostEqual(est[(4, 1)].throughput, 4 * 1 / 1.5)
        ev = [Event(float(t), 10, 1) for t in range(64)]             # one event per second
        est = Estimator(ConstModel(), quantile=1.0).estimate_all(ev, [1], [1, 8], window=64.0)
        self.assertAlmostEqual(est[(1, 8)].latency, 7.5)            # waits up to 7 s for the bundle


class GpuScalingTest(unittest.TestCase):
    def test_min_p_then_lowest_latency(self):
        c = cands({(1, 1): (1, 1), (2, 1): (5, 0.9), (2, 2): (6, 0.8), (3, 1): (9, 0.5)})
        ch = config_opt(c, lam=4, slo=1.0, p_max=3)
        self.assertEqual((ch.p, ch.b, ch.feasible), (2, 2, True))

    def test_overload_picks_the_highest_throughput(self):
        c = cands({(2, 1): (1, 3.0), (2, 2): (2, 2.5), (2, 4): (3, 4.0)})
        ch = config_opt(c, lam=4, slo=1.0, p_max=2)
        self.assertEqual((ch.p, ch.b, ch.overload), (2, 4, True))

    def test_scale_out_at_once(self):
        t = ScalingTrigger(p=1, b=1, p_max=4, delta=0.1)
        d = t.decide(cands({(1, 1): (1, 2.0), (2, 1): (4, 0.5)}), lam=3, slo=1.0)
        self.assertTrue(d.rescale)
        self.assertEqual((d.p, d.b), (2, 1))

    def test_change_b_without_rescaling(self):
        t = ScalingTrigger(p=2, b=1, p_max=4, delta=0.1)
        d = t.decide(cands({(1, 1): (1, 0.5), (2, 1): (2, 0.5), (2, 2): (4, 0.7)}), lam=3, slo=1.0)
        self.assertEqual((d.p, d.b, d.rescale, d.change_b), (2, 2, False, True))

    def test_scale_in_only_with_margin(self):
        t = ScalingTrigger(p=3, b=1, p_max=4, delta=0.1)
        d = t.decide(cands({(1, 1): (2.1, 0.5), (3, 1): (15, 0.3)}), lam=2, slo=1.0)   # T_hat < 1.1 lambda
        self.assertFalse(d.rescale)
        d = t.decide(cands({(1, 1): (5, 0.5), (3, 1): (15, 0.3)}), lam=2, slo=1.0)
        self.assertTrue(d.rescale)
        self.assertEqual(d.p, 1)


class CacheScalingTest(unittest.TestCase):
    def test_target_hit_rate(self):
        # t_h = 1 ms, t_m = 11 ms, lambda' = 200/s: h = (11 - 5) / (11 - 1) = 0.6, clipped to 0 to 1
        self.assertAlmostEqual(target_hit_rate(0.001, 0.011, 200), 0.6)
        self.assertEqual(target_hit_rate(0.001, 0.011, 50), 0.0)
        self.assertEqual(target_hit_rate(0.001, 0.011, 2000), 1.0)

    def test_capacity_from_window_counts(self):
        keys, size = required_capacity([50, 30, 10, 5, 5], 0.8, 10, 90)
        self.assertEqual((keys, size), (2, 200))


if __name__ == "__main__":
    unittest.main()
