import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_PACKAGES = ("experiments", "optimization", "physics", "precoders", "results")


class DependencyBoundaryTests(unittest.TestCase):
    def test_shared_packages_do_not_import_link_implementations(self) -> None:
        violations: list[str] = []
        for package_name in SHARED_PACKAGES:
            package = PROJECT_ROOT / "latency_optimization" / package_name
            for source_path in package.rglob("*.py"):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.module.startswith(
                        ("latency_optimization.uplink", "latency_optimization.downlink")
                    ):
                        violations.append(f"{source_path.relative_to(PROJECT_ROOT)}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_method_packages_have_one_obvious_entry_path(self) -> None:
        convergence_files = {
            "experiment.py",
            "optimize_transmission.py",
            "optimize_payload.py",
            "optimize_streaming.py",
            "optimize_precoder.py",
        }
        monte_carlo_files = {
            "experiment.py",
            "build_training_rollouts.py",
            "train_precoder_network.py",
            "evaluate_precoder_network.py",
            "precoder_network.py",
        }
        for link in ("uplink", "downlink"):
            methods = PROJECT_ROOT / "latency_optimization" / link / "methods"
            self.assertTrue(convergence_files.issubset({path.name for path in (methods / "convergence").glob("*.py")}))
            self.assertTrue(monte_carlo_files.issubset({path.name for path in (methods / "monte_carlo").glob("*.py")}))
            self.assertFalse((PROJECT_ROOT / "latency_optimization" / link / "convergence").exists())
            self.assertFalse((PROJECT_ROOT / "latency_optimization" / link / "monte_carlo").exists())
        self.assertFalse((PROJECT_ROOT / "latency_optimization" / "core").exists())

    def test_methods_do_not_import_sibling_method_internals(self) -> None:
        violations: list[str] = []
        for link in ("uplink", "downlink"):
            methods = PROJECT_ROOT / "latency_optimization" / link / "methods"
            for method_name, sibling_name in (("convergence", "monte_carlo"), ("monte_carlo", "convergence")):
                for source_path in (methods / method_name).glob("*.py"):
                    source = source_path.read_text(encoding="utf-8")
                    if f"methods.{sibling_name}" in source or f"..{sibling_name}" in source:
                        violations.append(str(source_path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
