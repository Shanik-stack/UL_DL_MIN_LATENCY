import json
import tempfile
import unittest
from pathlib import Path

from latency_optimization.results.persistence import save_text, write_result_manifest


class ResultReportingTests(unittest.TestCase):
    def test_manifest_records_files_and_removes_empty_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run"
            (root / "empty_category").mkdir(parents=True)
            save_text(["summary"], str(root / "testing" / "summary.txt"))

            manifest = write_result_manifest(
                root,
                setup={"link": "Uplink", "method": "Monte Carlo", "config_hash": "abc123"},
            )

            self.assertFalse((root / "empty_category").exists())
            self.assertEqual(manifest["status"], "complete")
            self.assertIn(str(Path("testing") / "summary.txt"), manifest["files"])
            self.assertIn("run_manifest.json", manifest["files"])
            self.assertIn("run_manifest.txt", manifest["files"])
            persisted = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["setup"]["config_hash"], "abc123")


if __name__ == "__main__":
    unittest.main()
