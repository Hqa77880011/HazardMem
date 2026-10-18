import unittest
import numpy as np
from finite_data import causal_features, physical_trace


class FiniteInputTests(unittest.TestCase):
    def test_prefix_features_cannot_read_future_or_truth(self):
        signals, label, slow = physical_trace(7, True, length=12)
        altered = signals.copy()
        altered[7:] += 99
        np.testing.assert_array_equal(causal_features(signals)[:7], causal_features(altered)[:7])
        self.assertEqual(causal_features(signals).shape, (12, 18))
        self.assertEqual(label.shape, slow.shape)
        self.assertEqual(signals.shape[1], 6)
