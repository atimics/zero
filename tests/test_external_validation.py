import hashlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExternalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.binary = cls.root / "literary_lm"
        subprocess.run(["cc", "-O1", "-std=c11", "-fsanitize=address,undefined",
                        str(ROOT / "literary_lm.c"), "-o", str(cls.binary), "-lm"], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_model(self, *args, check=True):
        return subprocess.run([str(self.binary), *map(str, args)], text=True,
                              capture_output=True, check=check)

    def test_validation_changes_scores_and_preserves_training(self):
        train = self.root / "train.txt"
        train.write_text("A boat crossed the lake. " * 80)
        checkpoints, losses = [], []
        for letter in ["a", "z"]:
            validation = self.root / f"val-{letter}.txt"
            validation.write_text(letter * 320)
            checkpoint = self.root / f"{letter}.ckpt"
            result = self.run_model("--text", train, "--validation-text", validation,
                "--context", 8, "--dim", 8, "--heads", 2, "--layers", 1, "--ff", 16,
                "--steps", 3, "--batch", 1, "--seed", 7, "--report", 3,
                "--validation", 2, "--tokens", 0, "--save", checkpoint)
            self.assertIn(f"train={len(train.read_text())} validation=320", result.stdout)
            self.assertNotIn("runtime error", result.stderr)
            loss = float(re.search(r" val ([0-9.]+)", result.stdout)[1])
            evaluated = self.run_model("--resume", checkpoint, "--steps", 0,
                "--evaluate", validation, "--validation", 2, "--tokens", 0)
            self.assertNotIn("update        ", evaluated.stdout)
            self.assertAlmostEqual(loss, float(re.search(r"loss=([0-9.]+)", evaluated.stdout)[1]), places=3)
            checkpoints.append(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            losses.append(loss)
        self.assertEqual(checkpoints[0], checkpoints[1])
        self.assertNotEqual(losses[0], losses[1])

    def test_short_evaluation_fails(self):
        short = self.root / "short.txt"
        short.write_text("abc")
        result = self.run_model("--steps", 0, "--evaluate", short, "--tokens", 0, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evaluation text is too short", result.stderr)


if __name__ == "__main__":
    unittest.main()
