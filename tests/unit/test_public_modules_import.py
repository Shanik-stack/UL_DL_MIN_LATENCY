import importlib
import unittest


PUBLIC_RUN_MODULES = (
    "latency_optimization.uplink.convergence.main",
    "latency_optimization.uplink.monte_carlo.main",
    "latency_optimization.downlink.convergence.main",
    "latency_optimization.downlink.monte_carlo.main",
    "latency_optimization.uplink.benchmarks.evaluate_test_dataset",
    "latency_optimization.downlink.benchmarks.evaluate_test_dataset",
)


class PublicModuleImportTests(unittest.TestCase):
    def test_public_run_modules_import(self) -> None:
        for module_name in PUBLIC_RUN_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
