import tempfile
import unittest
from pathlib import Path

from q250_uzh.checkpoints import find_latest_checkpoint


class TestCheckpoints(unittest.TestCase):
    def test_find_latest_checkpoint_prefers_latest_run_and_highest_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r1 = root / "2026-01-01_00-00-00"; r2 = root / "2026-01-02_00-00-00"
            r1.mkdir(); r2.mkdir()
            (r1 / "model_999.pt").write_text("old")
            (r2 / "model_50.pt").write_text("a")
            (r2 / "model_300.pt").write_text("b")
            self.assertEqual(find_latest_checkpoint(root), r2 / "model_300.pt")

    def test_find_latest_checkpoint_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(find_latest_checkpoint(Path(td)))


if __name__ == "__main__":
    unittest.main()
