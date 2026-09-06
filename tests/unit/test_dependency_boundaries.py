import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_PACKAGES = ("core", "experiments", "optimization", "physics", "precoders", "results")


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


if __name__ == "__main__":
    unittest.main()
