import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "latency_optimization"


class ComputeLayerArchitectureTests(unittest.TestCase):
    def test_physics_has_no_parallel_numpy_rate_implementation(self) -> None:
        physics_source = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "physics").glob("*.py")
        )
        for forbidden in (
            "finite_blocklength_mimo_numpy",
            "finite_blocklength_from_metric_numpy",
            "mimo_numpy",
            "from_metric_numpy",
        ):
            self.assertNotIn(forbidden, physics_source)

    def test_model_inference_modules_return_only_tensors(self) -> None:
        for link in ("uplink", "downlink"):
            source = (ROOT / link / "precoders" / "inference.py").read_text(encoding="utf-8")
            self.assertNotIn("numpy", source.lower())
            self.assertNotIn(".cpu().numpy()", source)

    def test_objectives_are_torch_only_and_simulator_independent(self) -> None:
        for link in ("uplink", "downlink"):
            source = (ROOT / link / "objective.py").read_text(encoding="utf-8")
            self.assertNotIn("import numpy", source)
            self.assertNotIn(".system import", source)
            self.assertNotIn(".cpu().numpy()", source)

    def test_both_links_use_the_same_precoder_package_layout(self) -> None:
        expected = {"models.py", "inference.py", "checkpoints.py"}
        for link in ("uplink", "downlink"):
            actual = {path.name for path in (ROOT / link / "precoders").glob("*.py")}
            self.assertTrue(expected.issubset(actual), f"{link} precoder package is incomplete")

    def test_both_links_expose_only_the_standard_config_loader(self) -> None:
        for link in ("uplink", "downlink"):
            source = (ROOT / link / "config.py").read_text(encoding="utf-8")
            self.assertIn("def load_config(", source)
            self.assertNotIn("def get_config(", source)


if __name__ == "__main__":
    unittest.main()
