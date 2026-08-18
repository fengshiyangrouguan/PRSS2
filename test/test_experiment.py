"""Experiment framework unit tests (matrix expansion, validation, id format)."""

import tempfile
import unittest
from pathlib import Path

from prss.experiment.runner import expand, job_id, load_experiment
from prss.experiment.summarize import summarize


SPEC = {
    "name": "t",
    "root": "outputs/t",
    "defaults": {"dataset": "tgbl-wiki", "epochs": 1},
    "matrix": {
        "variant": ["vanilla", "spectral"],
        "seed": [0, 1, 2],
    },
}


class TestRunner(unittest.TestCase):
    def test_expand_cartesian_product(self):
        jobs = expand(SPEC)
        self.assertEqual(len(jobs), 6)
        variants = {j["variant"] for j in jobs}
        seeds = {j["seed"] for j in jobs}
        self.assertEqual(variants, {"vanilla", "spectral"})
        self.assertEqual(seeds, {0, 1, 2})
        self.assertTrue(all(j["dataset"] == "tgbl-wiki" for j in jobs))

    def test_job_id_format(self):
        self.assertEqual(job_id({"dataset": "tgbl-wiki", "variant": "spectral",
                                 "seed": 3}), "tgbl-wiki__spectral__seed003")

    def test_load_requires_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("name: only\n")
            with self.assertRaises(ValueError):
                load_experiment(str(path))


class TestSummarize(unittest.TestCase):
    def test_summary_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, score in ((0, 0.70), (1, 0.74)):
                job_dir = root / f"tgbl-wiki__spectral__seed{seed:03d}"
                job_dir.mkdir()
                (job_dir / "summary.json").write_text(
                    '{"variant": "spectral", "dataset": "tgbl-wiki", "seed": %d, '
                    '"best_epoch": 1, "test": {"test_mrr": %s}, "spectral": {}}'
                    % (seed, score))
            table = summarize(root)
            self.assertIn("spectral", table)
            self.assertIn("0.720000 ± 0.028284", table)  # mean 0.72, std sqrt(2)*0.02


if __name__ == "__main__":
    unittest.main()
