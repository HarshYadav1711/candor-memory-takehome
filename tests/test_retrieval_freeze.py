from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RetrievalFreezeTests(unittest.TestCase):
    def test_phase3_retrieved_lists_remain_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phase4 = Path(directory) / "phase4.jsonl"
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "memory",
                    "--input",
                    str(ROOT / "evals" / "memory_train.jsonl"),
                    "--output",
                    str(phase4),
                    "--data",
                    str(ROOT / "data"),
                    "--answerer",
                    "extractive",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, run.returncode, run.stderr)
            reference = [
                json.loads(line)["retrieved"]
                for line in (ROOT / "outputs" / "memory_train_phase3.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            candidate = [
                json.loads(line)["retrieved"]
                for line in phase4.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(reference, candidate)


if __name__ == "__main__":
    unittest.main()

