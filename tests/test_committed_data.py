"""Guards the committed real NCR build under data/ncr/ (see data/ncr/README.md).

These tests read real data that was fetched once in Colab and committed together with its raw API responses:
they check it still validates, that the numbers quoted in the README are true, and that the current code
still rebuilds it byte-for-byte from the committed cache (no network).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_ph_heat_index_dataset as m  # noqa: E402

DATA = ROOT / "data" / "ncr"
CSV = DATA / "output" / "ph_heat_index_next_day.csv"


class CommittedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CSV.exists():
            raise unittest.SkipTest("data/ncr not present")
        cls.df = pd.read_csv(CSV, parse_dates=["Date"])

    def test_dataset_validates(self):
        report = m.validate_dataset(self.df, "2015-01-01", "2025-12-29")
        self.assertEqual(report["errors"], [])
        self.assertEqual(self.df.shape, (64240, 26))
        self.assertEqual(self.df["City"].nunique(), 16)
        self.assertEqual(report["checks"]["tomorrow_equals_next_day_today"]["mismatches_shift"], 0)

    def test_numbers_quoted_in_the_readme_are_true(self):
        vc = self.df["HeatLevelTomorrow"].value_counts()
        self.assertEqual({k: int(vc.get(k, 0)) for k in m.HEAT_LEVELS},
                         {"Not Hazardous": 158, "Caution": 8733, "Extreme Caution": 50018, "Danger": 5331, "Extreme Danger": 0})
        self.assertEqual(round(float(self.df["HeatIndex_Max_Today"].max()), 2), 48.16)
        self.assertEqual(round(float(self.df["Temp_Max"].max()), 1), 36.9)
        readme = (DATA / "README.md").read_text(encoding="utf-8")
        sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
        self.assertIn(sha, readme)                                         # the quoted checksum is the file's checksum
        for needle in ("158 (0.25 %)", "8,733 (13.59 %)", "50,018 (77.86 %)", "5,331 (8.30 %)", "48.16"):
            self.assertIn(needle, readme)

    def test_city_list_is_the_16_ncr_cities_in_two_grid_cells(self):
        cl = pd.read_csv(DATA / "output" / "ph_heat_index_city_list.csv", dtype={"city_id": str})
        self.assertEqual(len(cl), 16)
        self.assertEqual(set(cl["region"]), {"National Capital Region (NCR)"})
        self.assertEqual(cl[["grid_latitude", "grid_longitude"]].drop_duplicates().shape[0], 2)
        self.assertEqual(sorted(cl["City"]), sorted(self.df["City"].unique()))

    def test_cache_is_complete_and_keyed_for_the_current_code(self):
        chunks = sorted((DATA / "cache" / "chunks" / m.cache_key("era5")).glob("*.csv"))
        raws = sorted((DATA / "cache" / "raw" / m.raw_key("era5")).glob("*.json.gz"))
        self.assertEqual((len(chunks), len(raws)), (176, 176))             # 16 cities x 11 calendar years
        self.assertFalse(list(DATA.rglob("api_usage_log.json")))           # runtime state is not committed
        self.assertFalse(list(DATA.rglob("*.whl")))

    def test_rebuild_from_committed_cache_is_byte_identical_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, out = Path(tmp) / "cache", Path(tmp) / "out"
            shutil.copytree(DATA / "cache", work)                           # a run appends to run_log.txt
            rc = m.main(["--work-dir", str(work), "--out-dir", str(out), "--api-url", "http://127.0.0.1:1/unused"])
            self.assertEqual(rc, 0)                                         # no API call was needed (it would have failed)
            for name in ("ph_heat_index_next_day.csv", "ph_heat_index_city_list.csv", "ph_heat_index_data_dictionary.csv"):
                self.assertEqual((out / name).read_bytes(), (DATA / "output" / name).read_bytes(), name)
            self.assertTrue((out / "ncr_dataset_analysis.md").exists())

    def test_rebuild_from_raw_responses_alone_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, out = Path(tmp) / "cache", Path(tmp) / "out"
            shutil.copytree(DATA / "cache", work)
            shutil.rmtree(work / "chunks")                                  # force re-aggregation from raw
            rc = m.main(["--work-dir", str(work), "--out-dir", str(out), "--api-url", "http://127.0.0.1:1/unused"])
            self.assertEqual(rc, 0)
            self.assertEqual((out / "ph_heat_index_next_day.csv").read_bytes(), CSV.read_bytes())
            self.assertTrue(re.search(r"176 rebuilt from retained raw data", (work / "run_log.txt").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
