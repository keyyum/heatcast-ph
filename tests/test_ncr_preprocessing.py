"""Tests for ncr_preprocessing.py: data checks, split, folds, lags, the shared transformer and class handling.

Most tests use a small synthetic frame with the real column layout. The last class checks the committed real
build (data/ncr) and that the committed processed files are exactly what the current code produces.
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_ph_heat_index_dataset as B  # noqa: E402
import ncr_modeling_utils as U  # noqa: E402
import ncr_preprocessing as P  # noqa: E402

from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def make_frame(n_days: int = 500, cities=("Alpha", "Beta", "Gamma"), seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the layout of ph_heat_index_next_day.csv (sorted by City, Date, like the real file)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days)
    season = np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 100) / 365.25)
    tmax = 31 + 3 * season + rng.normal(0, 0.8, n_days)
    hmean = 78 - 4 * season + rng.normal(0, 3, n_days)
    hi = 36 + 5 * season + 0.8 * (tmax - 31) + rng.normal(0, 1.5, n_days)
    rain = rng.gamma(0.3, 15, n_days)
    frames = []
    for k, city in enumerate(cities):
        off = 0.1 * k
        d = pd.DataFrame({
            "Date": dates, "City": city, "Latitude": 14.5 + 0.05 * k, "Longitude": 121.0, "Elevation": 5.0 + 10 * k,
            "Month": dates.month, "DayOfYear": dates.dayofyear,
            "Temp_Max": tmax - off, "Temp_Mean": tmax - off - 2.5, "Temp_Min": tmax - off - 5,
            "Humidity_Max": np.minimum(hmean + 8, 99), "Humidity_Mean": hmean, "Humidity_Min": hmean - 10,
            "DewPoint_Mean": tmax - off - 6.5, "Pressure_Mean": 1010 + rng.normal(0, 3, n_days),
            "CloudCover_Mean": rng.uniform(5, 100, n_days), "WindSpeed_Mean": 8 + rng.gamma(2, 1, n_days),
            "Rainfall_Total": rain, "RainHours": rng.integers(0, 25, n_days).astype(float),
            "SolarRadiation_Total": rng.uniform(3000, 7000, n_days), "HeatIndex_Max_Today": np.round(hi - off, 2),
        })
        d["WindSpeed_Max"] = d["WindSpeed_Mean"] + 6
        d["WindGust_Max"] = d["WindSpeed_Max"] + 12
        d["HeatLevelToday"] = B.classify_heat_level(d["HeatIndex_Max_Today"])
        d["HeatIndex_Max_Tomorrow"] = d["HeatIndex_Max_Today"].shift(-1)
        d = d.iloc[:-1].copy()                                    # a city's last day has no next day
        d["HeatLevelTomorrow"] = B.classify_heat_level(d["HeatIndex_Max_Tomorrow"])
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out[["Date", "City", "Latitude", "Longitude", "Elevation", "Month", "DayOfYear", *U.DAILY_FEATURES,
                "HeatLevelToday", "HeatIndex_Max_Tomorrow", "HeatLevelTomorrow"]]


def prepared(**kw):
    kw.setdefault("min_train_days", 120)
    kw.setdefault("n_splits", 3)
    return P.prepare_dataset(make_frame(), **kw)


class DataChecks(unittest.TestCase):
    def test_clean_frame_passes(self):
        q = P.data_quality_checks(make_frame())
        self.assertEqual(q["problems"], [])
        self.assertEqual((q["missing_cells"], q["duplicate_rows"], q["date_gaps"]), (0, 0, 0))

    def test_problems_are_detected(self):
        base = make_frame()
        dup = pd.concat([base, base.iloc[[5]]], ignore_index=True)
        self.assertTrue(P.data_quality_checks(dup)["problems"])
        bad = base.copy()
        bad.loc[10, "Humidity_Mean"] = 140.0                                 # impossible
        self.assertTrue(P.data_quality_checks(bad)["problems"])
        mislabelled = base.copy()
        mislabelled.loc[10, "HeatLevelTomorrow"] = "Danger" if mislabelled.loc[10, "HeatLevelTomorrow"] != "Danger" else "Caution"
        self.assertTrue(P.data_quality_checks(mislabelled)["problems"])
        broken = base.copy()
        broken.loc[10, "HeatIndex_Max_Tomorrow"] += 1.0                      # no longer next day's value
        self.assertTrue(P.data_quality_checks(broken)["problems"])

    def test_prepare_refuses_bad_data(self):
        bad = make_frame()
        bad.loc[3, "Rainfall_Total"] = -5.0
        with self.assertRaises(ValueError):
            P.prepare_dataset(bad)


class Splits(unittest.TestCase):
    def test_split_is_by_date_with_a_gap(self):
        prep = prepared()
        tr, te = prep.train["Date"], prep.test["Date"]
        self.assertLess(tr.max() + pd.Timedelta(days=P.DEFAULT_GAP_DAYS), te.min() + pd.Timedelta(days=1))
        self.assertGreaterEqual((te.min() - tr.max()).days, P.DEFAULT_GAP_DAYS + 1)
        self.assertEqual(set(tr) & set(te), set())
        for frame in (prep.train, prep.test):                                  # every date has all three cities
            self.assertTrue((frame.groupby("Date")["City"].nunique() == 3).all())
        self.assertEqual(prep.test_start, te.min())
        c = prep.counts
        self.assertEqual(c["rows_in"] - c["dropped_for_lag_history"] - c["gap_rows"], c["train"] + c["test"])

    def test_test_share_is_about_twenty_percent_of_dates(self):
        prep = prepared()
        n_dates = prep.train["Date"].nunique() + prep.test["Date"].nunique() + P.DEFAULT_GAP_DAYS + P.MAX_LAG
        self.assertAlmostEqual(prep.test["Date"].nunique() / n_dates, 0.20, delta=0.01)

    def test_rows_are_sorted_by_date_so_a_row_split_is_also_chronological(self):
        prep = prepared()
        self.assertTrue(prep.train["Date"].is_monotonic_increasing)
        self.assertTrue(prep.test["Date"].is_monotonic_increasing)

    def test_folds_are_chronological_gapped_and_inside_training(self):
        prep = prepared()
        self.assertEqual(len(prep.cv_splits), 3)
        last_valid_end = None
        for tr_idx, va_idx in prep.cv_splits:
            tr, va = prep.train.iloc[tr_idx]["Date"], prep.train.iloc[va_idx]["Date"]
            self.assertGreater((va.min() - tr.max()).days, P.DEFAULT_GAP_DAYS)
            self.assertEqual(set(tr) & set(va), set())
            if last_valid_end is not None:
                self.assertGreater(va.min(), last_valid_end)                  # blocks move forward in time
            last_valid_end = va.max()
        self.assertLess(last_valid_end, prep.test_start)                      # no validation block touches the test period

    def test_gap_keeps_label_and_input_windows_apart(self):
        # a training row uses days d-3..d and the label day d+1; the first test row uses T0-3..T0
        prep = prepared()
        d_last = prep.train["Date"].max()
        t0 = prep.test["Date"].min()
        self.assertLess(d_last + pd.Timedelta(days=P.HORIZON_DAYS), t0 - pd.Timedelta(days=P.MAX_LAG))


class LagFeatures(unittest.TestCase):
    def test_lags_use_only_earlier_days(self):
        full = make_frame()
        cutoff = full["Date"].min() + pd.Timedelta(days=300)
        a = U.add_lag_features(full, deltas=P.DELTA_COLUMNS)
        b = U.add_lag_features(full[full["Date"] <= cutoff], deltas=P.DELTA_COLUMNS)
        a = a[a["Date"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[P.LAG_COLUMNS], b.reset_index(drop=True)[P.LAG_COLUMNS])

    def test_first_rows_without_history_are_dropped_once(self):
        prep = prepared()
        n_cities = 3
        self.assertEqual(prep.counts["dropped_for_lag_history"], n_cities * P.MAX_LAG)
        self.assertFalse(prep.train[P.LAG_COLUMNS].isna().any().any())
        self.assertFalse(prep.test[P.LAG_COLUMNS].isna().any().any())

    def test_lag_value_is_the_value_k_days_earlier(self):
        prep = prepared()
        row = prep.train.iloc[200]
        earlier = prep.train[(prep.train["City"] == row["City"]) &
                             (prep.train["Date"] == row["Date"] - pd.Timedelta(days=2))]
        self.assertAlmostEqual(row["HeatIndex_Max_Today_lag2"], earlier["HeatIndex_Max_Today"].iloc[0], places=9)


class SharedTransformer(unittest.TestCase):
    def test_feature_sets_and_guard(self):
        self.assertEqual(len(P.feature_columns("today", "city")), 16 + 1 + 1)
        self.assertEqual(len(P.feature_columns("hi_lags", "coords")), 16 + 3 + 1 + 3)
        self.assertEqual(len(P.feature_columns("all_lags", "none")), 16 + len(P.LAG_COLUMNS) + 1)
        self.assertEqual(len(P.LAG_COLUMNS), 44)
        for cols in (P.feature_columns(s, "city") for s in P.FEATURE_SETS):
            self.assertFalse(set(cols) & {"HeatLevelTomorrow", "HeatIndex_Max_Tomorrow", "HeatLevelToday", "Date", "Month"})
        with self.assertRaises(ValueError):
            P.feature_columns("nonsense")
        with self.assertRaises(ValueError):
            P.feature_columns("today", "nonsense")
        with self.assertRaises(ValueError):
            U.assert_no_future_information(P.feature_columns("today") + ["HeatIndex_Max_Tomorrow"])

    def test_output_is_numeric_complete_and_named(self):
        prep = prepared()
        X = prep.X(prep.train, "hi_lags")
        out = P.make_preprocessor("hi_lags").fit_transform(X)
        self.assertEqual(out.shape, (len(X), 16 + 3 + 2 + 3))               # weather + lags + sin/cos + 3 cities
        self.assertFalse(out.isna().any().any())
        self.assertTrue({"DayOfYear_sin", "DayOfYear_cos", "City_Alpha", "HeatIndex_Max_Today_lag1"} <= set(out.columns))
        self.assertFalse({"Date", "Month", "HeatLevelTomorrow", "DayOfYear"} & set(out.columns))
        self.assertTrue(np.allclose(out["Temp_Max"].mean(), 0, atol=1e-9))
        self.assertTrue(np.allclose(out["Temp_Max"].std(ddof=0), 1, atol=1e-9))

    def test_static_options(self):
        prep = prepared()
        n = {s: P.make_preprocessor("today", s).fit_transform(prep.X(prep.train, "today", s)).shape[1]
             for s in P.STATIC_OPTIONS}
        self.assertEqual(n, {"city": 16 + 2 + 3, "coords": 16 + 2 + 3, "none": 16 + 2})

    def test_statistics_come_from_training_rows_only(self):
        prep = prepared()
        pre = P.make_preprocessor("today").fit(prep.X(prep.train, "today"))
        scaler = pre.named_transformers_["numeric"].named_steps["scale"]
        col = list(pre.named_transformers_["numeric"].feature_names_in_).index("Temp_Max")
        self.assertAlmostEqual(scaler.mean_[col], prep.train["Temp_Max"].mean(), places=9)
        self.assertNotAlmostEqual(scaler.mean_[col], pd.concat([prep.train, prep.test])["Temp_Max"].mean(), places=3)
        before = scaler.mean_.copy()
        pre.transform(prep.X(prep.test, "today"))                              # transforming test must not change anything
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_rainfall_is_log_transformed(self):
        prep = prepared()
        X = prep.X(prep.train, "today")
        out = P.make_preprocessor("today").fit_transform(X)
        self.assertLess(abs(out["Rainfall_Total"].skew()), abs(X["Rainfall_Total"].skew()))

    def test_pruned_weather_follows_the_eda_list_and_keeps_only_its_lags(self):
        cols = P.feature_columns("all_lags", "none", weather="lr_pruned")
        self.assertEqual(cols[:7], P.PRUNED_WEATHER)
        self.assertNotIn("DewPoint_Mean_lag1", cols)                      # a dropped variable's lags are dropped too
        self.assertIn("HeatIndex_Max_Today_lag3", cols)
        self.assertIn("Temp_Min", cols)
        self.assertNotIn("Temp_Max_lag2", cols)
        self.assertEqual(len(P.feature_columns("today", "city", weather="lr_pruned")), 7 + 1 + 1)
        prep = prepared()
        out = P.make_preprocessor("today", "none", weather="lr_pruned").fit_transform(
            prep.X(prep.train, "today", "none", weather="lr_pruned"))
        self.assertEqual(out.shape[1], 7 + 2)

    def test_level_today_is_off_by_default_and_ordinal_when_asked(self):
        self.assertNotIn(P.LEVEL_CODE, P.feature_columns("today"))
        self.assertIn(P.LEVEL_CODE, P.feature_columns("today", level_today=True))
        prep = prepared()
        mapped = prep.train["HeatLevelToday"].map(P.LEVEL_CODES)
        self.assertTrue((prep.train[P.LEVEL_CODE] == mapped).all())
        out = P.make_preprocessor("today", level_today=True).fit_transform(prep.X(prep.train, "today", level_today=True))
        self.assertIn(P.LEVEL_CODE, out.columns)

    def test_log_applies_to_rain_and_mean_wind_but_not_to_other_columns(self):
        prep = prepared()
        X = prep.X(prep.train, "all_lags")
        out = P.make_preprocessor("all_lags").fit_transform(X)
        for col in ("Rainfall_Total", "WindSpeed_Mean", "Rainfall_Total_lag1", "WindSpeed_Mean_mean3d"):
            z = (np.log1p(X[col]) - np.log1p(X[col]).mean()) / np.log1p(X[col]).std(ddof=0)
            np.testing.assert_allclose(out[col].to_numpy(), z.to_numpy(), atol=1e-9)
        z = (X["Temp_Max"] - X["Temp_Max"].mean()) / X["Temp_Max"].std(ddof=0)       # a plain column is only scaled
        np.testing.assert_allclose(out["Temp_Max"].to_numpy(), z.to_numpy(), atol=1e-9)

    def test_day_to_day_changes_are_past_only_differences(self):
        prep = prepared()
        row = prep.train.iloc[300]
        y = prep.train[(prep.train["City"] == row["City"]) & (prep.train["Date"] == row["Date"] - pd.Timedelta(days=1))]
        self.assertAlmostEqual(row["Temp_Max_change1d"], row["Temp_Max"] - y["Temp_Max"].iloc[0], places=9)

    def test_cyclic_season_joins_december_and_january(self):
        x = pd.DataFrame({"DayOfYear": [1, 365]})
        s = P._day_of_year_cycle(x)
        self.assertLess(np.linalg.norm(s[0] - s[1]), 0.05)

    def test_unknown_city_and_negative_rain_do_not_crash(self):
        prep = prepared()
        pre = P.make_preprocessor("today").fit(prep.X(prep.train, "today"))
        X = prep.X(prep.test, "today").head(3).copy()
        X["City"] = "Atlantis"
        X["Rainfall_Total"] = -1.0
        out = pre.transform(X)
        self.assertFalse(out.isna().any().any())
        self.assertEqual(float(out.filter(like="City_").to_numpy().sum()), 0.0)

    def test_pipeline_cross_validates_with_a_placeholder_model(self):
        # DummyClassifier is not a model for the project: it only proves the splits and the transformer work together
        prep = prepared()
        pipe = Pipeline([("prep", P.make_preprocessor("hi_lags")), ("clf", DummyClassifier(strategy="most_frequent"))])
        scores = cross_val_score(pipe, prep.X(prep.train, "hi_lags"), prep.y(prep.train), cv=prep.cv_splits)
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


class ClassHandling(unittest.TestCase):
    def test_default_keeps_all_five_labels_and_every_row(self):
        prep = prepared()
        self.assertEqual(prep.target_policy, "five_class")
        self.assertEqual(prep.class_labels, U.CLASS_ORDER)
        self.assertEqual(encode(["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]), [0, 1, 2, 3, 4])
        c = prep.counts
        self.assertEqual(c["rows_in"], c["dropped_for_lag_history"] + c["gap_rows"] + c["train"] + c["test"])  # nothing dropped by class

    def test_three_class_policy_is_only_an_option(self):
        self.assertEqual(encode(["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"], "three_class"),
                         [0, 0, 1, 2, 2])
        prep = prepared(target_policy="three_class")
        self.assertEqual(prep.class_labels, P.POLICY_LABELS["three_class"])

    def test_unknown_label_or_policy_fails(self):
        with self.assertRaises(ValueError):
            P.encode_target(["Lukewarm"])
        with self.assertRaises(ValueError):
            P.encode_target(["Danger"], "seven_class")

    def test_class_weights_are_balanced_over_present_classes(self):
        y = np.array([2] * 90 + [3] * 9 + [0] * 1)
        w = P.class_weights(y)
        self.assertEqual(sorted(w), [0, 2, 3])
        counts = {0: 1, 2: 90, 3: 9}
        for k, v in w.items():
            self.assertAlmostEqual(v * counts[k] * 3, 100.0)
        self.assertEqual(max(P.class_weights(y, cap=10).values()), 10)
        self.assertNotIn(1, w)                                                 # an absent class gets no weight


def encode(labels, policy="five_class"):
    return P.encode_target(labels, policy).tolist()


class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not P.DEFAULT_CSV.exists():
            raise unittest.SkipTest("data/ncr not present")
        cls.prep = P.prepare_dataset(P.DEFAULT_CSV)

    def test_documented_split(self):
        p = self.prep
        self.assertEqual(p.test_start, pd.Timestamp("2023-10-18"))
        self.assertEqual(p.test["Date"].max(), pd.Timestamp("2025-12-28"))
        self.assertEqual(p.train["Date"].min(), pd.Timestamp("2015-01-04"))
        self.assertEqual(p.train["Date"].max(), pd.Timestamp("2023-10-13"))
        self.assertEqual(p.counts, {"rows_in": 64240, "dropped_for_lag_history": 48, "gap_rows": 64, "train": 51280, "test": 12848})
        self.assertEqual(p.facts["not_hazardous_days"], 16)
        self.assertLess(pd.Timestamp(p.facts["last_not_hazardous_day"]), p.test_start)       # no Not Hazardous day in the test period
        self.assertEqual([len(v) for _, v in p.cv_splits], [7904] * 5)

    def test_documented_class_counts(self):
        p = self.prep
        tr, te = np.bincount(p.y(p.train), minlength=5).tolist(), np.bincount(p.y(p.test), minlength=5).tolist()
        self.assertEqual(tr, [158, 7175, 39893, 4054, 0])
        self.assertEqual(te, [0, 1510, 10061, 1277, 0])
        self.assertEqual(sum(tr) + sum(te) + p.counts["gap_rows"] + p.counts["dropped_for_lag_history"], 64240)

    def test_no_future_columns_in_any_feature_set(self):
        for fs in P.FEATURE_SETS:
            for st in P.STATIC_OPTIONS:
                U.assert_no_future_information(P.feature_columns(fs, st))

    def test_committed_processed_files_match_the_code(self):
        committed = P.DEFAULT_OUT
        if not (committed / "split_definition.json").exists():
            self.skipTest("data/ncr/processed not generated yet")
        with tempfile.TemporaryDirectory() as tmp:
            P.write_outputs(self.prep, Path(tmp))
            for name in ("split_definition.json", "preprocessing_report.md"):
                self.assertEqual((Path(tmp) / name).read_bytes(), (committed / name).read_bytes(), name)
            with gzip.open(Path(tmp) / "ncr_model_ready.csv.gz") as a, gzip.open(committed / "ncr_model_ready.csv.gz") as b:
                self.assertEqual(a.read(), b.read())
        info = json.loads((committed / "split_definition.json").read_text())
        self.assertEqual(info["test_start"], "2023-10-18")
        self.assertEqual(info["class_labels"], U.CLASS_ORDER)

    def test_report_makes_no_claim_that_a_class_was_changed(self):
        text = P.render_report(self.prep)
        self.assertIn("No class merged or dropped", text)
        self.assertIn("not applied", text.lower())
        self.assertIn("Open: the group decides", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
