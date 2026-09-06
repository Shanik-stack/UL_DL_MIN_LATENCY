import unittest

import numpy as np
import torch

from latency_optimization.results.persistence import make_serializable


class ResultPersistenceTests(unittest.TestCase):
    def test_nested_numpy_and_tensor_values_are_serializable(self) -> None:
        value = {1: (np.array([2, 3]), torch.tensor([4.0, 5.0]))}
        self.assertEqual(make_serializable(value), {"1": [[2, 3], [4.0, 5.0]]})


if __name__ == "__main__":
    unittest.main()
