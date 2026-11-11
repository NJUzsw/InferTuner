import unittest
try:
    import torch
except ImportError:
    torch = None
from infertuner.length_predictor import masked_pool


@unittest.skipIf(torch is None, 'torch is part of requirements-ml.txt')
class FeatureTest(unittest.TestCase):
    def test_padding_does_not_change_embedding(self):
        short = torch.tensor([[[99.], [2.], [4.]]])
        padded = torch.tensor([[[99.], [2.], [4.], [900.], [800.]]])
        a = masked_pool(short, torch.tensor([[1, 1, 1]]))
        b = masked_pool(padded, torch.tensor([[1, 1, 1, 0, 0]]))
        self.assertEqual(a.item(), 3.)
        self.assertEqual(a.item(), b.item())

    def test_empty_non_cls_mask_stays_finite(self):
        x = masked_pool(torch.tensor([[[99.], [200.]]]), torch.tensor([[1, 0]]))
        self.assertEqual(x.item(), 0.)
