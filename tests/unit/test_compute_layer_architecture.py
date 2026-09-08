import ast
import builtins
import symtable
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "latency_optimization"


class ComputeLayerArchitectureTests(unittest.TestCase):
    def test_modules_do_not_reference_undefined_global_names(self) -> None:
        builtin_names = set(dir(builtins))
        runtime_names = {"__file__", "__name__", "__package__", "__path__"}
        unresolved: dict[str, list[str]] = {}

        for path in ROOT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            module_table = symtable.symtable(source, str(path), "exec")
            defined_at_module_scope = {
                symbol.get_name()
                for symbol in module_table.get_symbols()
                if symbol.is_assigned()
                or symbol.is_imported()
                or symbol.is_namespace()
                or symbol.is_parameter()
            }
            globally_referenced: set[str] = set()

            def collect_global_references(table: symtable.SymbolTable) -> None:
                globally_referenced.update(
                    symbol.get_name()
                    for symbol in table.get_symbols()
                    if symbol.is_referenced() and symbol.is_global()
                )
                for child in table.get_children():
                    collect_global_references(child)

            collect_global_references(module_table)
            missing = sorted(
                globally_referenced
                - defined_at_module_scope
                - builtin_names
                - runtime_names
            )
            if missing:
                unresolved[str(path.relative_to(ROOT))] = missing

        self.assertEqual({}, unresolved)

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
            source = (ROOT / link / "objectives" / "precoder.py").read_text(encoding="utf-8")
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
            source = (ROOT / link / "configuration" / "loader.py").read_text(encoding="utf-8")
            self.assertIn("def load_config(", source)
            self.assertNotIn("def get_config(", source)

    def test_link_roots_contain_no_mixed_responsibility_modules(self) -> None:
        for link in ("uplink", "downlink"):
            root_modules = {path.name for path in (ROOT / link).glob("*.py")}
            self.assertEqual({"__init__.py"}, root_modules)

    def test_both_links_use_the_same_responsibility_packages(self) -> None:
        expected = {
            "benchmarks",
            "configuration",
            "methods",
            "objectives",
            "precoders",
            "results",
            "simulation",
            "physics",
        }
        for link in ("uplink", "downlink"):
            actual = {path.name for path in (ROOT / link).iterdir() if path.is_dir()}
            self.assertTrue(expected.issubset(actual), f"{link} package layout is incomplete")

    def test_payload_and_streaming_baselines_are_explicit(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for link in ("uplink", "downlink")
            for path in (ROOT / link).rglob("*.py")
        )
        self.assertNotIn("random_precoder_schedule_for_scenario", source)
        self.assertNotIn("random_precoders_for_scenario", source)

    def test_uplink_has_one_active_monte_carlo_evaluation_path(self) -> None:
        uplink_source = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "uplink").rglob("*.py")
        )
        self.assertNotIn("evaluate_payload_with_trained_precoder_network", uplink_source)
        self.assertFalse((ROOT / "uplink" / "model_service.py").exists())

    def test_critical_execution_functions_explain_their_role(self) -> None:
        required = {
            "downlink/simulation/block_state.py": {
                "expand_precoders_for_plan",
                "ensure_precoder_block",
                "user_link_budget",
                "evaluate_block_candidate",
            },
            "downlink/simulation/system.py": {
                "compose_full_precoder",
                "project_block_precoders_to_power",
                "compute_block_rate",
                "apply_solution",
            },
            "physics/finite_blocklength.py": {
                "finite_blocklength_from_metric",
                "finite_blocklength_mimo",
            },
            "uplink/physics/rate.py": {
                "build_uplink_rate_covariance_torch",
                "evaluate_uplink_rate_tensor",
            },
        }
        for relative_path, function_names in required.items():
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            definitions = {
                node.name: ast.get_docstring(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for function_name in function_names:
                self.assertTrue(
                    definitions.get(function_name),
                    f"{relative_path}:{function_name} needs a technical docstring",
                )


if __name__ == "__main__":
    unittest.main()
