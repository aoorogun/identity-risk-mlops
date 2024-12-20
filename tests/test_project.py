from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prembly_risk.config import load_config
from prembly_risk.data import generate, load_data
from prembly_risk.modeling import actions, train
from prembly_risk.monitoring import psi


class ProjectTests(unittest.TestCase):
    def test_generation_and_training(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config["data"]["rows"] = 3000
            config["paths"]["data"] = str(root / "data.csv")
            config["paths"]["artifacts"] = str(root / "artifacts")
            config["paths"]["reports"] = str(root / "reports")
            config["training"]["candidates"] = ["logistic_regression"]
            generate(config)
            self.assertEqual(len(load_data(config["paths"]["data"])), 3000)
            metadata = train(config)
            self.assertIn("roc_auc", metadata["test_metrics"])

    def test_actions_are_ordered(self):
        self.assertEqual(
            actions(np.array([0.01, 0.3, 0.9]), 0.1, 0.7).tolist(), ["approve", "review", "reject"]
        )

    def test_psi_shift(self):
        rng = np.random.default_rng(2)
        a = __import__("pandas").Series(rng.normal(0, 1, 3000))
        b = __import__("pandas").Series(rng.normal(2, 1, 3000))
        self.assertGreater(psi(a, b), 0.25)


if __name__ == "__main__":
    unittest.main()
