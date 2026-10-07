"""Tests for ncr_modeling_utils.py and ncr_dataset_analysis.py.

The analysis is checked against synthetic panels with *planted* structure (known grid cells, known
lapse-rate effect, known AR(2) dynamics, known class counts), so a pass means the code recovers what
is really in the data - not just that it runs. Synthetic data only; nothing here is real weather.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_ph_heat_index_dataset as m  # noqa: E402
import ncr_dataset_analysis as A  # noqa: E402
import ncr_modeling_utils as U  # noqa: E402


def make_panel(n_days=2200, start="2018-01-01", cells=((14.75, 121.0, (10, 20, 30)), (14.50, 121.0, (5, 15))),
               lapse_per_m=-0.0065, hi_per_t=1.4, phi=(0.6, 0.25), seed=7):
    """Final-schema panel: cities in `cells` share their cell's weather except for lapse-rate temperature shifts."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days)
    doy = dates.dayofyear.to_numpy()
    rows, city_rows = [], []
    for ci, (glat, glon, elevs) in enumerate(cells):
        e = rng.normal(0, 2.0, n_days)
        a = np.zeros(n_days)
        for t in range(2, n_days):
            a[t] = phi[0] * a[t - 1] + phi[1] * a[t - 2] + e[t]
        base_hi = 40 + 3 * np.sin(2 * np.pi * (doy - 110) / 365.25) + a
        tmax = 28 + 0.25 * base_hi
        rain = np.where(rng.random(n_days) < 0.3, rng.exponential(5, n_days), 0.0)
        shared = {
            "Pressure_Mean": 1010 + rng.normal(0, 2, n_days), "CloudCover_Mean": rng.uniform(20, 90, n_days),
            "WindSpeed_Mean": rng.uniform(5, 15, n_days), "WindSpeed_Max": rng.uniform(15, 25, n_days),
            "WindGust_Max": rng.uniform(25, 40, n_days), "Rainfall_Total": rain,
            "RainHours": (rain > 0).astype(int) * rng.integers(1, 8, n_days),
            "SolarRadiation_Total": rng.uniform(3000, 6000, n_days),
            "Humidity_Min": rng.uniform(40, 60, n_days), "Humidity_Mean": rng.uniform(65, 85, n_days),
            "Humidity_Max": rng.uniform(88, 100, n_days), "DewPoint_Mean": rng.uniform(22, 26, n_days),
        }
        for k, elev in enumerate(elevs):
            dT = lapse_per_m * elev
            city = f"C{ci}_{k}"
            city_rows.append({"City": city, "grid_latitude": glat, "grid_longitude": glon, "elevation_m": float(elev),
                              "latitude": glat + 0.01 * k, "longitude": glon + 0.01 * k})
            d = pd.DataFrame({"Date": dates, "City": city, "Latitude": glat + 0.01 * k, "Longitude": glon + 0.01 * k,
                              "Elevation": float(elev), "Month": dates.month, "DayOfYear": doy})
            d["Temp_Max"] = np.round(tmax + dT, 2)
            d["Temp_Mean"] = np.round(tmax - 4 + dT, 2)
            d["Temp_Min"] = np.round(tmax - 8 + dT, 2)
            for col, v in shared.items():
                d[col] = np.round(v, 2) if col != "RainHours" else v
            d["HeatIndex_Max_Today"] = np.round(base_hi + hi_per_t * dT, 2)
            rows.append(d)
    final, _ = m.build_targets(pd.concat(rows, ignore_index=True))
    return final, pd.DataFrame(city_rows)


class UtilsTests(unittest.TestCase):
    def test_future_guard(self):
        U.assert_no_future_information(U.INPUT_FEATURES)
        for bad in (["HeatIndex_Max_Tomorrow"], ["HeatLevelTomorrow"], ["Temp_Max_next"], ["Rain_lead1"], ["x", "Temp_tomorrow"]):
            with self.assertRaises(ValueError):
                U.assert_no_future_information(bad)

    def test_input_feature_list_matches_the_specification(self):
        spec = ["Temp_Min", "Temp_Mean", "Temp_Max", "Humidity_Min", "Humidity_Mean", "Humidity_Max", "DewPoint_Mean",
                "Pressure_Mean", "CloudCover_Mean", "WindSpeed_Mean", "WindSpeed_Max", "WindGust_Max", "Rainfall_Total",
                "RainHours", "SolarRadiation_Total", "HeatIndex_Max_Today", "Month", "DayOfYear", "City", "Latitude",
                "Longitude", "Elevation"]
        self.assertEqual(U.INPUT_FEATURES, spec)

    @staticmethod
    def _small():
        d = pd.date_range("2020-01-01", periods=10)
        rows = []
        for c, off in (("A", 0.0), ("B", 100.0)):
            for i, day in enumerate(d):
                rows.append({"City": c, "Date": day, "HeatIndex_Max_Today": off + i, "Pressure_Mean": 1000.0 + off + i * 2,
                             **{k: 1.0 for k in U.LAG_SOURCES if k not in ("HeatIndex_Max_Today", "Pressure_Mean")}})
        return pd.DataFrame(rows)

    def test_lag_values_and_no_cross_city_bleed(self):
        out = U.add_lag_features(self._small())
        a = out[out.City == "A"].set_index("Date")
        b = out[out.City == "B"].set_index("Date")
        d = pd.Timestamp("2020-01-06")
        self.assertEqual(a.loc[d, "HeatIndex_Max_Today_lag1"], 4.0)
        self.assertEqual(a.loc[d, "HeatIndex_Max_Today_lag3"], 2.0)
        self.assertEqual(b.loc[d, "HeatIndex_Max_Today_lag2"], 103.0)               # city B's own history, not A's
        self.assertTrue(np.isnan(a.loc["2020-01-01", "HeatIndex_Max_Today_lag1"]))
        self.assertTrue(np.isnan(b.loc["2020-01-03", "HeatIndex_Max_Today_lag3"]))   # needs 3 prior days
        self.assertAlmostEqual(a.loc[d, "HeatIndex_Max_Today_mean3d"], (5 + 4 + 3) / 3)     # today + 2 previous days
        self.assertTrue(np.isnan(a.loc["2020-01-02", "HeatIndex_Max_Today_mean3d"]))
        self.assertEqual(a.loc[d, "Pressure_Mean_change1d"], 2.0)
        self.assertEqual(list(out.columns[-len(U.lag_feature_names()):]), U.lag_feature_names())

    def test_lags_are_calendar_aware_across_gaps(self):
        df = self._small()
        df = df[~((df.City == "A") & (df.Date == pd.Timestamp("2020-01-05")))]       # remove one day
        out = U.add_lag_features(df)
        a = out[out.City == "A"].set_index("Date")
        self.assertTrue(np.isnan(a.loc["2020-01-06", "HeatIndex_Max_Today_lag1"]))     # Jan 5 missing -> NaN, not Jan 4
        self.assertEqual(a.loc["2020-01-06", "HeatIndex_Max_Today_lag2"], 3.0)          # Jan 4 is two days back
        self.assertTrue(np.isnan(a.loc["2020-01-06", "HeatIndex_Max_Today_mean3d"]))

    def test_lag_features_use_only_past_information(self):
        """Scramble everything after day T; features on days <= T must not change."""
        final, _ = make_panel(n_days=120)
        base = U.add_lag_features(final)
        cutoff = pd.Timestamp("2018-03-01")
        changed = final.copy()
        rng = np.random.default_rng(0)
        fut = changed["Date"] > cutoff
        for col in U.LAG_SOURCES:
            changed[col] = changed[col].astype(float)
            changed.loc[fut, col] = rng.normal(500, 50, fut.sum())
        again = U.add_lag_features(changed)
        past = (final["Date"] <= cutoff).to_numpy()
        names = U.lag_feature_names()
        pd.testing.assert_frame_equal(base.loc[past, names].reset_index(drop=True), again.loc[past, names].reset_index(drop=True))
        self.assertFalse(base.loc[~past, names].equals(again.loc[~past, names]))        # (and the test can detect change)

    def test_duplicate_rows_rejected(self):
        d = self._small()
        with self.assertRaises(ValueError):
            U.add_lag_features(pd.concat([d, d.iloc[:1]]))

    def test_chronological_split_by_date_with_embargo(self):
        final, _ = make_panel(n_days=400)
        tr, te = U.chronological_split(final, "2018-12-01", embargo_days=2)
        dtr, dte = final.iloc[tr]["Date"], final.iloc[te]["Date"]
        self.assertLess(dtr.max(), pd.Timestamp("2018-12-01") - pd.Timedelta(days=2))
        self.assertGreaterEqual(dte.min(), pd.Timestamp("2018-12-01"))
        self.assertEqual(set(dtr.unique()) & set(dte.unique()), set())                 # a date is entirely on one side
        for day in dte.unique()[:5]:                                                   # all cities of a date stay together
            self.assertEqual((final["Date"] == day).sum(), (dte == day).sum())
        gap_rows = len(final) - len(tr) - len(te)
        self.assertEqual(gap_rows, 2 * final["City"].nunique())                        # exactly the embargo days dropped
        with self.assertRaises(ValueError):
            U.chronological_split(final, "2030-01-01")

    def test_naive_row_order_split_is_not_chronological(self):
        """Documents the pitfall: the CSV is sorted City-then-Date, so 'first 80% of rows' splits by city."""
        final, _ = make_panel(n_days=300)
        n80 = int(len(final) * 0.8)
        train, test = final.iloc[:n80], final.iloc[n80:]
        self.assertEqual(test["Date"].min(), final["Date"].min())          # the "test" partition starts at the very first date
        self.assertEqual(train["Date"].max(), final["Date"].max())         # while "train" runs to the very last: not chronological
        self.assertGreater(train["Date"].max(), test["Date"].min())
        self.assertNotEqual(set(train["City"]), set(test["City"]))         # the partitions are (mostly) different cities

    def test_expanding_window_folds(self):
        final, _ = make_panel(n_days=1500)
        folds = U.expanding_window_folds(final, n_splits=4, gap_days=2, min_train_days=700)
        self.assertEqual(len(folds), 4)
        prev_test_end = None
        prev_train_n = 0
        for tr, te in folds:
            dtr, dte = final.iloc[tr]["Date"], final.iloc[te]["Date"]
            self.assertLessEqual(dtr.max(), dte.min() - pd.Timedelta(days=3))        # gap of 2 days between
            self.assertEqual(set(dtr.unique()) & set(dte.unique()), set())
            self.assertGreater(len(tr), prev_train_n)                               # expanding window
            if prev_test_end is not None:
                self.assertGreater(dte.min(), prev_test_end)                         # test blocks increase, disjoint
            prev_test_end, prev_train_n = dte.max(), len(tr)
            self.assertGreaterEqual(dtr.nunique(), 700 - 5)
        self.assertEqual(final.iloc[folds[-1][1]]["Date"].max(), final["Date"].max())     # last fold reaches the last date
        with self.assertRaises(ValueError):
            U.expanding_window_folds(final.head(500), n_splits=5, min_train_days=700)


class MetricsTests(unittest.TestCase):
    def test_hand_computed_metrics(self):
        r = A.classification_metrics(list("aabbc"), list("abbba"), labels=["a", "b", "c"])
        pc = r["per_class"]
        self.assertAlmostEqual(pc.loc["a", "precision"], 0.5)
        self.assertAlmostEqual(pc.loc["b", "precision"], 2 / 3)
        self.assertAlmostEqual(pc.loc["c", "recall"], 0.0)
        self.assertAlmostEqual(pc.loc["b", "f1"], 0.8)
        self.assertAlmostEqual(r["macro_f1"], (0.5 + 0.8 + 0.0) / 3)
        self.assertAlmostEqual(r["accuracy"], 0.6)
        self.assertEqual(r["confusion"].to_numpy().tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])

    def test_runs(self):
        d = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10", "2020-01-20", "2020-01-21"]).to_numpy()
        self.assertEqual(A._runs(d), [3, 1, 2])
        self.assertEqual(A._runs(np.array([], dtype="datetime64[ns]")), [])


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.cities = make_panel()
        cls.res = A.run_analysis(cls.df, cls.cities)

    # ---- 1/2: grid series and elevation
    def test_recovers_the_planted_grid_structure(self):
        g = self.res["grid"]
        self.assertEqual(g["n_cities"], 5)
        self.assertEqual(g["n_grid_cells"], 2)
        self.assertEqual(g["n_nonthermal_series"], 2)
        groups = sorted(sorted(x) for x in g["nonthermal_groups"])
        self.assertEqual(groups, [["C0_0", "C0_1", "C0_2"], ["C1_0", "C1_1"]])
        self.assertEqual(g["n_pairs_same_cell"], 3 + 1)
        self.assertEqual(g["n_pairs_diff_cell"], 6)
        self.assertTrue((g["identical_by_column"] == 1.0).all())                  # shared weather is exactly shared
        self.assertGreater(g["effective_series"], 1.7)
        self.assertLess(g["effective_series"], 2.7)                               # ~2 independent series, not 5

    def test_recovers_planted_lapse_rate(self):
        s = self.res["grid"]["lapse_slopes_per_100m"]
        self.assertAlmostEqual(s["Temp_Mean"], -0.65, delta=0.02)
        self.assertAlmostEqual(s["Temp_Max"], -0.65, delta=0.02)
        self.assertAlmostEqual(s["HeatIndex_Max_Today"], -0.65 * 1.4, delta=0.03)
        self.assertAlmostEqual(s["Humidity_Mean (%)"], 0.0, delta=0.05)           # humidity was not adjusted in the fixture

    def test_elevation_verdict_follows_the_size_of_the_effect(self):
        # 5-10 m spacing -> ~0.11 C heat-index shift, yet ~1% of days still flip class because days sitting
        # near a class boundary tip over: that is "modest" under the documented rule, not "negligible".
        g = self.res["grid"]
        self.assertEqual(g["elevation_effect"], "modest")
        self.assertLess(g["same_cell_mean_abs_dHI"], A.MODEST_HI_DIFF_C)
        self.assertGreaterEqual(g["same_cell_class_disagree_today"], A.MODEST_CLASS_DISAGREE)
        self.assertLess(g["same_cell_class_disagree_today"], A.MATERIAL_CLASS_DISAGREE)
        tiny, tc = make_panel(cells=((14.75, 121.0, (5, 6, 7)), (14.5, 121.0, (5, 6))))   # 1-2 m spacing
        self.assertEqual(A.analyze_grid_series(tiny, tc)["elevation_effect"], "negligible")
        big, cities = make_panel(cells=((14.75, 121.0, (0, 80, 160)), (14.5, 121.0, (0, 90))))
        self.assertEqual(A.analyze_grid_series(big, cities)["elevation_effect"], "material")
        same = cities.assign(elevation_m=7.0)
        flat, _ = make_panel(lapse_per_m=0.0, cells=((14.75, 121.0, (7, 7, 7)), (14.5, 121.0, (7, 7))))
        self.assertIn("not testable", A.analyze_grid_series(flat, same)["elevation_effect"])

    def test_cities_in_one_cell_never_disagree_when_no_adjustment(self):
        flat, cities = make_panel(lapse_per_m=0.0)
        g = A.analyze_grid_series(flat, cities)
        self.assertEqual(g["same_cell_mean_abs_dHI"], 0.0)
        self.assertEqual(g["same_cell_class_disagree_today"], 0.0)
        self.assertGreater(g["diff_cell_mean_abs_dHI"], 0.5)                      # different cells really differ

    # ---- 3/4: classes
    def test_class_counts_match_direct_counting(self):
        ov, df = self.res["classes"]["overall"], self.df
        vc = df[U.TARGET].value_counts()
        for cls in U.CLASS_ORDER:
            self.assertEqual(int(ov.loc[cls, "rows"]), int(vc.get(cls, 0)))
        self.assertAlmostEqual(float(ov["percent_rows"].sum()), 100.0, places=6)
        self.assertEqual(list(ov.index), U.CLASS_ORDER)                           # all five classes always reported, none merged
        # independent episode count for Danger (event day = Date + 1)
        days = sorted(set((df.loc[df[U.TARGET] == "Danger", "Date"] + pd.Timedelta(days=1)).tolist()))
        episodes = 1 + sum(1 for a, b in zip(days, days[1:]) if (b - a).days > 1)
        self.assertEqual(int(ov.loc["Danger", "episodes"]), episodes)
        self.assertEqual(int(ov.loc["Danger", "distinct_days"]), len(days))
        self.assertLessEqual(int(ov.loc["Danger", "distinct_days"]), int(ov.loc["Danger", "rows"]))

    def test_rare_classes_are_flagged_not_dropped(self):
        df = self.df.copy()
        df.loc[df[U.NUMERIC_TARGET] >= 52, [U.NUMERIC_TARGET]] = 45.0              # remove Extreme Danger entirely
        df[U.TARGET] = m.classify_heat_level(df[U.NUMERIC_TARGET])
        r = A.analyze_classes(df, self.cities)
        self.assertEqual(r["flags"].loc["Extreme Danger", "status"], "ABSENT")
        self.assertEqual(list(r["overall"].index), U.CLASS_ORDER)                 # still reported
        self.assertEqual(int(r["overall"].loc["Extreme Danger", "rows"]), 0)
        self.assertEqual(int(r["overall"]["rows"].sum()), len(df))                # nothing dropped

    def test_handful_of_events_is_very_few(self):
        df = self.df.copy()
        df[U.NUMERIC_TARGET] = df[U.NUMERIC_TARGET].clip(upper=51.0)
        three_days = df["Date"].drop_duplicates().iloc[[100, 800, 1500]]
        df.loc[df["Date"].isin(three_days), U.NUMERIC_TARGET] = 53.0
        df[U.TARGET] = m.classify_heat_level(df[U.NUMERIC_TARGET])
        r = A.analyze_classes(df, self.cities)
        self.assertEqual(int(r["overall"].loc["Extreme Danger", "episodes"]), 3)
        self.assertEqual(r["flags"].loc["Extreme Danger", "status"], "VERY FEW")

    def test_split_and_folds_are_chronological_and_complete(self):
        c = self.res["classes"]
        n_cities = self.df["City"].nunique()
        self.assertEqual(c["n_train_rows"] + c["n_test_rows"] + c["embargo_days"] * n_cities, len(self.df))
        self.assertLess(pd.Timestamp(c["train_range"][1]), pd.Timestamp(c["test_range"][0]))
        self.assertEqual(len(c["cv"]), 5)
        self.assertTrue((pd.to_datetime(c["cv"]["train_end"]) < pd.to_datetime(c["cv"]["test_start"])).all())
        self.assertTrue(pd.to_datetime(c["cv"]["test_start"]).is_monotonic_increasing)

    def test_danger_section(self):
        d = self.res["danger"]["Danger"]
        n_cities = self.df["City"].nunique()
        self.assertEqual(d["rows"], int((self.df[U.TARGET] == "Danger").sum()))
        self.assertGreaterEqual(d["mean_cities_in_class_on_event_days"], 1.0)
        self.assertLessEqual(d["mean_cities_in_class_on_event_days"], n_cities)

    # ---- 5: leakage
    def test_leakage_simulations(self):
        lk = self.res["leakage"]
        self.assertEqual(lk["chron_test_rows_with_same_date_train_row"], 0.0)
        self.assertEqual(lk["chron_test_rows_with_same_cell_date_train_row"], 0.0)
        self.assertEqual(lk["chron_exact_duplicate_weather_vectors_across_partitions"], 0)
        self.assertGreater(lk["random_split_test_rows_with_same_cell_date_train_row"], 0.8)    # theory ~0.90 here
        self.assertLess(lk["random_split_test_rows_with_same_cell_date_train_row"], 0.97)
        self.assertEqual(lk["leave_one_city_out_rows_with_same_cell_mate_in_train"], 1.0)
        nv = lk["naive_row_order_split"]
        self.assertFalse(nv["chronological"])
        self.assertEqual(nv["test_dates"][0], str(self.df["Date"].min().date()))
        # the fixture has two *independent* cells, so date alone explains only the shared part ...
        self.assertGreater(lk["eta2_date"], 0.4)
        self.assertLess(lk["eta2_date"], 0.8)
        # ... but (cell, date) explains essentially everything: cities within a cell add ~nothing
        self.assertGreater(lk["eta2_cell_date"], 0.99)
        self.assertLess(lk["eta2_city"], 0.01)
        self.assertEqual(lk["n_cell_days"], 2 * self.df["Date"].nunique())
        self.assertEqual(lk["static_identifier_combinations"], 5)

    def test_same_day_duplicate_shares(self):
        lk = self.res["leakage"]
        self.assertEqual(lk["same_day_nonthermal_duplicate_share"], 1.0)           # every city has a same-cell twin
        self.assertEqual(lk["same_day_full_weather_duplicate_share"], 0.0)         # temperature differs by elevation
        self.assertAlmostEqual(lk["distinct_nonthermal_vectors_per_date_mean"], 2.0)

    # ---- 6: lags
    def test_planted_ar2_structure_is_recovered(self):
        p = self.res["lags"]["pacf"]
        self.assertAlmostEqual(p[2], 0.25, delta=0.07)
        self.assertLess(abs(p[3]), 0.07)
        t = self.res["lags"]["table"]
        self.assertGreater(abs(t.loc["HeatIndex_Max_Today_lag1", "partial_corr"]), abs(t.loc["HeatIndex_Max_Today_lag3", "partial_corr"]))
        self.assertTrue(bool(t.loc["HeatIndex_Max_Today_lag1", "useful"]))
        self.assertEqual(set(t.index) <= set(U.lag_feature_names()), True)

    # ---- 7: baselines
    def test_persistence_baseline_matches_manual_computation(self):
        c = self.res["classes"]
        tr, te = U.chronological_split(self.df, c["test_start"], c["embargo_days"])
        test = self.df.iloc[te]
        manual = A.classification_metrics(test[U.TARGET], test["HeatLevelToday"])
        got = self.res["baselines"]["persistence (tomorrow = today's class)"]
        self.assertAlmostEqual(got["macro_f1"], manual["macro_f1"])
        self.assertAlmostEqual(got["accuracy"], float((test[U.TARGET] == test["HeatLevelToday"]).mean()))
        self.assertEqual(len(self.res["baselines"]), 3)

    # ---- 8 + report
    def test_report_renders_with_all_sections_and_recommendations(self):
        md = A.render_markdown(self.res)
        for heading in ("## Key findings", "## 1. How many distinct grid series", "## 2. Does Open-Meteo's elevation",
                        "## 3. Class counts", "## 4. Danger and Extreme Danger", "## 5. Shared grid series and leakage",
                        "## 6. Lag features", "## 7. Reference baselines", "## 8. Recommendations"):
            self.assertIn(heading, md)
        for needle in ("not 5 independent weather stations", "No model has been trained", "NOT chronological", "invalid"):
            self.assertIn(needle, md)
        self.assertGreaterEqual(len(self.res["recommendations"]), 8)
        self.assertEqual(len(A.key_findings(self.res)), 4)

    def test_analysis_never_mutates_or_drops_input(self):
        before = self.df.copy()
        A.run_analysis(self.df, self.cities)
        pd.testing.assert_frame_equal(before, self.df)

    def test_single_city_degenerate_case(self):
        one = self.df[self.df["City"] == "C0_0"]
        cities = self.cities[self.cities["City"] == "C0_0"]
        res = A.run_analysis(one, cities)
        self.assertEqual(res["grid"]["n_grid_cells"], 1)
        self.assertEqual(res["grid"]["effective_series"], 1.0)
        self.assertIn("not testable", res["grid"]["elevation_effect"])
        A.render_markdown(res)

    def test_missing_grid_columns_raise_a_clear_error(self):
        with self.assertRaises(ValueError):
            A.run_analysis(self.df, self.cities.drop(columns=["grid_latitude"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
