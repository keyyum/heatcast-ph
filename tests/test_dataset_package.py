"""Tests for make_dataset_package.py: the processed data dictionary and the submission package layout."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import make_dataset_package as M  # noqa: E402


class ProcessedDictionary(unittest.TestCase):
    def test_covers_every_processed_column_in_order(self):
        cols = list(pd.read_csv(M.PROCESSED_GZ, nrows=1).columns)
        d = M.processed_dictionary()
        self.assertEqual(list(d["column"]), cols)
        self.assertFalse(d["description"].eq("").any())

    def test_lag_units_follow_their_source_column(self):
        d = M.processed_dictionary().set_index("column")
        self.assertEqual(d.loc["Rainfall_Total_lag2", "unit"], d.loc["Rainfall_Total", "unit"])
        self.assertEqual(d.loc["Temp_Max_change1d", "unit"], d.loc["Temp_Max", "unit"])

    def test_committed_dictionary_matches_the_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh = M.write_processed_dictionary(Path(tmp) / "dict.csv")
            # compare text with normalised line endings: git may check the file out with CRLF on Windows
            norm = lambda p: p.read_text(encoding="utf-8").replace("\r\n", "\n")  # noqa: E731
            self.assertEqual(norm(fresh), norm(M.PROCESSED_DICT),
                             "run `python make_dataset_package.py --dictionary-only` and commit the result")


class Package(unittest.TestCase):
    def test_layout_and_plain_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = M.build_package(Path(tmp) / "pkg")
            for rel in ("README.md", "data/original/ph_heat_index_next_day.csv", "data/original/SOURCES_AND_LICENSES.md",
                        "data/original/ph_heat_index_data_dictionary.csv", "data/processed/ncr_model_ready.csv",
                        "data/processed/ncr_model_ready_data_dictionary.csv", "data/processed/preprocessing_report.md"):
                self.assertTrue((out / rel).is_file(), rel)
            plain = pd.read_csv(out / "data/processed/ncr_model_ready.csv")
            gz = pd.read_csv(M.PROCESSED_GZ)
            pd.testing.assert_frame_equal(plain, gz)


if __name__ == "__main__":
    unittest.main()
