"""Tests for build_ph_heat_index_dataset.py.

All API traffic goes to tests/mock_open_meteo.py (synthetic data). Nothing here touches
Open-Meteo. Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_ph_heat_index_dataset as m  # noqa: E402
from mock_open_meteo import MockOpenMeteo  # noqa: E402

try:
    from metpy.calc import heat_index as _metpy_hi
    from metpy.units import units as _units
except Exception:  # pragma: no cover
    _metpy_hi = None


def f2c(f):
    return (np.asarray(f, float) - 32) * 5 / 9


def c2f(c):
    return np.asarray(c, float) * 9 / 5 + 32


class HeatIndexTests(unittest.TestCase):
    # (T degF, RH %, HI degF) confirmed with MetPy's NWS implementation, tolerance 1e-3 degF
    REFERENCE = [(90, 70, 105.922), (100, 40, 109.2556), (96, 60, 116.1312), (86, 80, 99.8007),
                 (95, 90, 146.6031), (110, 10, 104.389), (84, 95, 100.8817), (104, 50, 130.5823)]

    def test_pinned_reference_values(self):
        for t, rh, expected in self.REFERENCE:
            got = float(c2f(m.heat_index_celsius(f2c(t), rh)))
            self.assertAlmostEqual(got, expected, delta=1e-3, msg=f"T={t}F RH={rh}%")

    def test_cool_air_uses_simple_formula(self):
        t_f, rh = 70.0, 50.0
        simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
        self.assertAlmostEqual(float(c2f(m.heat_index_celsius(f2c(t_f), rh))), simple, places=6)

    @unittest.skipIf(_metpy_hi is None, "metpy not installed")
    def test_agrees_with_metpy_outside_switchover_band(self):
        tc, rh = np.meshgrid(np.arange(15, 45.01, 0.1), np.arange(5, 100.01, 1.0))
        mine = m.heat_index_celsius(tc, rh)
        ref = _metpy_hi((tc * 9 / 5 + 32) * _units.degF, rh * _units.percent, mask_undefined=False).to("degC").magnitude
        diff = np.abs(mine - ref)
        # The NWS WPC text and MetPy switch from the simple to the regression formula at slightly
        # different points (the boundary depends on RH too), so they differ only for T of about 25-27 C.
        outside = (tc < 24.5) | (tc >= 27.5)
        self.assertLess(diff[outside].max(), 1e-6)
        self.assertLess(diff.max(), 1.3)

    def test_invalid_or_missing_inputs_are_nan(self):
        out = m.heat_index_celsius([np.nan, 30, 30, 61, -91], [60, np.nan, 101, 60, 60])
        self.assertTrue(np.isnan(out).all())
        self.assertFalse(np.isnan(m.heat_index_celsius(30.0, 100.0)))
        self.assertFalse(np.isnan(m.heat_index_celsius(30.0, 0.0)))

    def test_low_humidity_adjustment_applies(self):
        # T=100F RH=10%: regression minus the dry adjustment must differ from the bare regression
        t, rh = 100.0, 10.0
        bare = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 0.00683783 * t * t
                - 0.05481717 * rh * rh + 0.00122874 * t * t * rh + 0.00085282 * t * rh * rh
                - 0.00000199 * t * t * rh * rh)
        adj = ((13 - rh) / 4) * np.sqrt((17 - abs(t - 95)) / 17)
        self.assertAlmostEqual(float(c2f(m.heat_index_celsius(f2c(t), rh))), bare - adj, places=6)

    def test_high_humidity_adjustment_applies(self):
        t, rh = 82.0, 95.0
        bare = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 0.00683783 * t * t
                - 0.05481717 * rh * rh + 0.00122874 * t * t * rh + 0.00085282 * t * rh * rh
                - 0.00000199 * t * t * rh * rh)
        adj = ((rh - 85) / 10) * ((87 - t) / 5)
        self.assertAlmostEqual(float(c2f(m.heat_index_celsius(f2c(t), rh))), bare + adj, places=6)


class ClassificationTests(unittest.TestCase):
    def test_boundaries_lower_inclusive(self):
        vals = [26.99, 27.0, 32.0, 32.99, 33.0, 41.99, 42.0, 51.99, 52.0, 70.0, -5.0, 32.5]
        want = ["Not Hazardous", "Caution", "Caution", "Caution", "Extreme Caution", "Extreme Caution",
                "Danger", "Danger", "Extreme Danger", "Extreme Danger", "Not Hazardous", "Caution"]
        self.assertEqual(m.classify_heat_level(vals).tolist(), want)

    def test_nan_stays_nan(self):
        out = m.classify_heat_level([np.nan, 30.0])
        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertEqual(out.iloc[1], "Caution")


def make_payload(start="2020-03-01", end="2020-03-02", temp=None, rh=None, rain=None, null_hours=None):
    """Hand-built hourly payload (2 days by default) for aggregation tests."""
    n = (pd.Timestamp(end) - pd.Timestamp(start)).days * 24 + 24
    times = pd.date_range(start, periods=n, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    hourly["temperature_2m"] = list(temp if temp is not None else np.linspace(25, 35, n))
    hourly["relative_humidity_2m"] = list(rh if rh is not None else np.linspace(90, 50, n))
    hourly["dew_point_2m"] = [22.0] * n
    hourly["pressure_msl"] = [1010.0] * n
    hourly["cloud_cover"] = [50.0] * n
    hourly["precipitation"] = list(rain if rain is not None else [0.0] * n)
    hourly["rain"] = list(rain if rain is not None else [0.0] * n)
    hourly["shortwave_radiation"] = [100.0] * n
    hourly["wind_speed_10m"] = [10.0] * n
    hourly["wind_gusts_10m"] = [20.0] * n
    for var, idx in (null_hours or {}).items():
        for i in idx:
            hourly[var][i] = None
    return {"latitude": 14.5, "longitude": 121.0, "elevation": 12.0, "hourly": hourly}


class AggregationTests(unittest.TestCase):
    def test_hand_computed_day(self):
        temp = [20.0 + h for h in range(24)] + [30.0] * 24         # day1: 20..43, day2: constant 30
        rain = [0.0] * 24 + [0.0, 0.05, 0.1, 0.5, 2.0] + [0.0] * 19  # day2: 3 hours >= 0.1 mm
        p = make_payload(temp=temp, rain=rain, rh=[60.0] * 48)
        d = m.aggregate_daily(p, "2020-03-01", "2020-03-02")
        d1, d2 = d.iloc[0], d.iloc[1]
        self.assertEqual((d1.Temp_Min, d1.Temp_Max, d1.Temp_Mean), (20.0, 43.0, 31.5))
        self.assertEqual((d2.Temp_Min, d2.Temp_Max), (30.0, 30.0))
        self.assertEqual(d2.RainHours, 3)
        self.assertAlmostEqual(d2.Rainfall_Total, 2.65, places=6)
        self.assertEqual(d1.RainHours, 0)
        self.assertEqual(d1.SolarRadiation_Total, 2400.0)      # 24 x 100 W/m2
        self.assertEqual((d1.WindSpeed_Mean, d1.WindSpeed_Max, d1.WindGust_Max), (10.0, 10.0, 20.0))
        hourly_hi = m.heat_index_celsius(np.array(temp[:24]), np.full(24, 60.0))
        self.assertAlmostEqual(d1.HeatIndex_Max_Today, round(float(hourly_hi.max()), 2), places=6)
        self.assertEqual(d1.Date, "2020-03-01")

    def test_incomplete_day_yields_nan_only_for_affected_variables(self):
        p = make_payload(null_hours={"wind_gusts_10m": [3], "relative_humidity_2m": [30]})
        d = m.aggregate_daily(p, "2020-03-01", "2020-03-02")
        self.assertTrue(pd.isna(d.loc[0, "WindGust_Max"]))                 # day 1 gust incomplete
        self.assertFalse(pd.isna(d.loc[1, "WindGust_Max"]))
        self.assertTrue(pd.isna(d.loc[1, "HeatIndex_Max_Today"]))          # day 2 humidity hour missing
        self.assertTrue(pd.isna(d.loc[1, "Humidity_Mean"]))
        self.assertFalse(pd.isna(d.loc[0, "HeatIndex_Max_Today"]))
        self.assertFalse(pd.isna(d.loc[0, "Temp_Max"]))
        self.assertEqual(d.loc[1, "_HoursMin"], 23)

    def test_range_mismatch_is_rejected(self):
        with self.assertRaises(m.ApiDataError):
            m.aggregate_daily(make_payload(), "2020-03-02", "2020-03-03")


class TargetTests(unittest.TestCase):
    @staticmethod
    def _daily(dates_by_city):
        rows = []
        for city, dates in dates_by_city.items():
            for i, d in enumerate(dates):
                rows.append({"City": city, "Date": pd.Timestamp(d), "HeatIndex_Max_Today": 30.0 + i + (5 if city == "B" else 0)})
        return pd.DataFrame(rows)

    def test_shift_within_city_and_last_row_dropped(self):
        daily = self._daily({"B": ["2020-01-01", "2020-01-02", "2020-01-03"], "A": ["2020-01-01", "2020-01-02"]})
        out, info = m.build_targets(daily)
        self.assertEqual(info["last_rows_dropped"], 2)
        self.assertEqual(out["City"].tolist(), ["A", "B", "B"])             # sorted by City then Date
        self.assertEqual(out["HeatIndex_Max_Tomorrow"].tolist(), [31.0, 36.0, 37.0])  # never crosses cities
        self.assertEqual(out["HeatLevelTomorrow"].tolist(), ["Caution", "Extreme Caution", "Extreme Caution"])

    def test_gap_never_pairs_with_wrong_day(self):
        daily = self._daily({"A": ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]})
        out, _ = m.build_targets(daily)
        by_date = out.set_index("Date")["HeatIndex_Max_Tomorrow"]
        self.assertEqual(by_date[pd.Timestamp("2020-01-01")], 31.0)
        self.assertTrue(pd.isna(by_date[pd.Timestamp("2020-01-02")]))       # next row is Jan 4, not Jan 3
        self.assertEqual(by_date[pd.Timestamp("2020-01-04")], 33.0)


class CityNameTests(unittest.TestCase):
    def test_clean_city_name(self):
        cases = {"City of Davao": "Davao", "Quezon City": "Quezon", "Island Garden City of Samal": "Samal",
                 "Science City of Muñoz": "Muñoz", "City of Cagayan De Oro": "Cagayan de Oro",
                 "City of San Jose Del Monte": "San Jose del Monte", "City of Sto. Tomas": "Santo Tomas",
                 "Batangas City": "Batangas", "City of Lapu-Lapu": "Lapu-Lapu"}
        for raw, want in cases.items():
            self.assertEqual(m.clean_city_name(raw), want, raw)

    def test_real_city_list_if_network(self):
        """Builds the real city list from the pinned PyPI wheel; skipped when PyPI is unreachable."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                df = m.build_city_list(Path(tmp), m.Logger(None))
            except Exception as exc:  # network / env
                self.skipTest(f"PyPI unreachable: {exc}")
        self.assertEqual(len(df), 149)
        self.assertTrue(df["City"].is_unique)
        self.assertEqual(df["city_class"].value_counts().to_dict(),
                         {"Component City": 111, "Highly Urbanized City": 33, "Independent Component City": 5})
        self.assertTrue(df["latitude"].between(4, 22).all() and df["longitude"].between(116, 128).all())
        self.assertEqual(sorted(c for c in df["City"] if "(" in c),
                         sorted(["Naga (Camarines Sur)", "Naga (Cebu)", "San Carlos (Negros Occidental)",
                                 "San Carlos (Pangasinan)", "San Fernando (La Union)", "San Fernando (Pampanga)",
                                 "Talisay (Cebu)", "Talisay (Negros Occidental)"]))


    def test_committed_reference_list_matches_a_fresh_build(self):
        """reference/ph_city_list_base.csv must be reproducible from the pinned PSGC wheel."""
        ref = pd.read_csv(ROOT / "reference" / "ph_city_list_base.csv", dtype={"city_id": str})
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fresh = m.build_city_list(Path(tmp), m.Logger(None))
            except Exception as exc:
                self.skipTest(f"PyPI unreachable: {exc}")
        pd.testing.assert_frame_equal(ref, fresh, check_dtype=False, check_exact=False, atol=1e-9)


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class RateBudgetTests(unittest.TestCase):
    def test_minute_window_waits_then_proceeds(self):
        c = FakeClock()
        b = m.RateBudget({"minute": 100}, clock=c.now, sleep=c.sleep)
        for _ in range(3):
            b.acquire(30)
            b.record(30)
        b.acquire(30)                       # 90 used + 30 > 100 -> must wait for the first call to age out
        self.assertEqual(len(c.slept), 1)
        self.assertGreater(c.slept[0], 50)
        self.assertLessEqual(c.slept[0], 62)

    def test_day_window_raises_with_wait_time(self):
        c = FakeClock()
        b = m.RateBudget({"day": 100}, clock=c.now, sleep=c.sleep)
        b.acquire(60); b.record(60)
        c.t += 3600
        b.acquire(30); b.record(30)
        with self.assertRaises(m.DailyLimitReached) as ctx:
            b.acquire(30)
        self.assertAlmostEqual(ctx.exception.wait_seconds, 86400 - 3600, delta=1)
        self.assertEqual(c.slept, [])

    def test_usage_persists_across_instances(self):
        c = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            b1 = m.RateBudget({"day": 100}, usage_path=path, clock=c.now, sleep=c.sleep)
            b1.acquire(80); b1.record(80)
            b2 = m.RateBudget({"day": 100}, usage_path=path, clock=c.now, sleep=c.sleep)
            with self.assertRaises(m.DailyLimitReached):
                b2.acquire(30)
            c.t += 86401
            b3 = m.RateBudget({"day": 100}, usage_path=path, clock=c.now, sleep=c.sleep)
            b3.acquire(30)                   # a day later: usage rolled off

    def test_default_budgets_let_the_whole_ncr_job_run_without_an_hourly_pause(self):
        """Regression: with a 4,500/h cap the 4,590-call NCR job stalled for ~51 min at chunk 172/176 in Colab."""
        ref = pd.read_csv(ROOT / "reference" / "ph_city_list_base.csv", dtype={"city_id": str})
        chunks = m.plan_chunks(m.filter_regions(ref, ["NCR"]), m.DEFAULT_START, m.DEFAULT_END)
        total = sum(m.call_weight(c.n_days) for c in chunks) + 1.0                  # + the preflight call
        self.assertEqual(len(chunks), 176)
        self.assertLess(total, m.DEFAULT_BUDGETS["hour"])
        self.assertLess(total, m.DEFAULT_BUDGETS["day"])
        # replay the whole job through the budget with a fake clock: it must never sleep for an hour
        c = FakeClock()
        b = m.RateBudget(dict(m.DEFAULT_BUDGETS), clock=c.now, sleep=c.sleep)
        b.acquire(1.0); b.record(1.0)
        for ch in chunks:
            b.acquire(m.call_weight(ch.n_days)); b.record(m.call_weight(ch.n_days))
            c.t += 0.5                                                              # ~0.5 s per request
        self.assertLess(max(c.slept, default=0), 100)                              # only brief per-minute pauses
        self.assertFalse(any(x >= 3000 for x in c.slept))

    def test_default_budgets_stay_below_the_server_limits(self):
        for k, v in m.DEFAULT_BUDGETS.items():
            self.assertLess(v, m.FREE_LIMITS[k])
            self.assertGreater(v, 0.9 * m.FREE_LIMITS[k])                           # ... but not needlessly far below

    def test_short_waits_are_quiet_and_long_waits_explain_themselves(self):
        c, lines, seen = FakeClock(), [], []
        b = m.RateBudget({"minute": 100}, clock=c.now, sleep=c.sleep, log=lines.append)
        b.on_wait = lambda secs, why: seen.append((round(secs), why))
        for _ in range(3):
            b.acquire(30); b.record(30)
        b.acquire(30)                                                                # ~60 s wait: routine
        self.assertEqual(lines, [])
        self.assertEqual(seen[0][1], "minute")
        c2, lines2 = FakeClock(), []
        b2 = m.RateBudget({"hour": 100}, clock=c2.now, sleep=c2.sleep, log=lines2.append)
        for _ in range(3):
            b2.acquire(30); b2.record(30)
        b2.acquire(30)                                                               # ~1 h wait: must be explained
        self.assertEqual(len(lines2), 1)
        for needle in ("PAUSE", "not an error and not a hang", "--hour-budget 4900", "5,000"):
            self.assertIn(needle, lines2[0])

    def test_call_weight_matches_open_meteo_formula(self):
        self.assertAlmostEqual(m.call_weight(365), 365 / 14, places=9)
        self.assertEqual(m.call_weight(3), 1.0)
        self.assertEqual(m.call_weight(14), 1.0)
        self.assertAlmostEqual(m.call_weight(28), 2.0, places=9)
        self.assertAlmostEqual(m.call_weight(14, n_vars=15), 1.5, places=9)   # documented example


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        self._orig_sleep = m._sleep
        m._sleep = self.sleeps.append

    def tearDown(self):
        m._sleep = self._orig_sleep

    def _client(self, mock, **kw):
        return m.OpenMeteoClient(mock.url, "era5", backoff_base=1.0, **kw)

    def test_happy_path_returns_validated_payload(self):
        with MockOpenMeteo() as mock:
            data = self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-10")
        self.assertEqual(len(data["hourly"]["time"]), 240)
        self.assertEqual(mock.requests[0]["models"], "era5")
        self.assertEqual(mock.requests[0]["tz"], "Asia/Manila")

    def test_retries_server_errors_truncation_and_short_arrays(self):
        with MockOpenMeteo() as mock:
            for kind in ("http500", "http503", "truncated", "short"):
                mock.inject(kind)
            data = self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
            self.assertEqual([r["served"] for r in mock.requests], ["http500", "http503", "truncated", "short", "data"])
        self.assertEqual(len(data["hourly"]["time"]), 120)
        self.assertEqual(len(self.sleeps), 4)

    def test_minutely_and_hourly_429_sleep_then_retry(self):
        with MockOpenMeteo() as mock:
            mock.inject("rate_minutely"); mock.inject("rate_hourly")
            self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
        self.assertGreaterEqual(self.sleeps[0], 65.0)
        self.assertGreaterEqual(self.sleeps[1], 15.0)
        self.assertLessEqual(self.sleeps[1], 3700.0)

    def test_daily_429_raises_without_sleeping(self):
        with MockOpenMeteo() as mock:
            mock.inject("rate_daily")
            with self.assertRaises(m.DailyLimitReached):
                self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
        self.assertEqual(self.sleeps, [])

    def test_bad_request_is_fatal_and_not_retried(self):
        with MockOpenMeteo() as mock:
            mock.inject("bad_request")
            with self.assertRaises(m.FatalApiError):
                self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
            self.assertEqual(len(mock.requests), 1)

    def test_wrong_timezone_is_fatal(self):
        with MockOpenMeteo() as mock:
            mock.inject("wrong_tz")
            with self.assertRaises(m.FatalApiError):
                self._client(mock).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")

    def test_gives_up_after_max_retries(self):
        with MockOpenMeteo() as mock:
            mock.inject("http500", 5)
            with self.assertRaises(m.FatalApiError):
                self._client(mock, max_retries=3).fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
            self.assertEqual(len(mock.requests), 3)

    def test_connection_refused_is_retried_then_fatal(self):
        client = m.OpenMeteoClient("http://127.0.0.1:1/v1/archive", "era5", max_retries=2, backoff_base=0.1)
        with self.assertRaises(m.FatalApiError):
            client.fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
        self.assertEqual(len(self.sleeps), 1)

    def test_proxy_policy_denial_fails_fast_with_clear_message(self):
        import requests

        class DeniedSession:
            headers = {}
            calls = 0

            def get(self, *a, **k):
                DeniedSession.calls += 1
                raise requests.exceptions.ProxyError(OSError("Tunnel connection failed: 403 Forbidden"))

        client = m.OpenMeteoClient("https://archive-api.open-meteo.com/v1/archive", "era5", session=DeniedSession())
        with self.assertRaises(m.FatalApiError) as ctx:
            client.fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-05")
        self.assertEqual(DeniedSession.calls, 1)                 # no pointless retries
        self.assertIn("archive-api.open-meteo.com", str(ctx.exception))
        self.assertIn("network policy", str(ctx.exception))
        self.assertEqual(self.sleeps, [])

    def test_preflight_flags_variable_with_no_data(self):
        city = pd.Series({"City": "Manila", "latitude": 14.6, "longitude": 121.0})
        with MockOpenMeteo(null_vars={"wind_gusts_10m"}) as mock:
            with self.assertRaises(m.FatalApiError) as ctx:
                m.preflight(self._client(mock), city, "2015-01-01", m.Logger(None))
        self.assertIn("wind_gusts_10m", str(ctx.exception))

    def test_local_budget_stops_before_hitting_the_server(self):
        with MockOpenMeteo() as mock:
            budget = m.RateBudget({"day": 28})
            client = self._client(mock, budget=budget)
            client.fetch_hourly(14.6, 121.0, "2020-01-01", "2020-12-31")          # weight 26.14 (leap year)
            client.fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-14")          # weight 1 -> 27.14 <= 28
            with self.assertRaises(m.DailyLimitReached):
                client.fetch_hourly(14.6, 121.0, "2020-01-01", "2020-01-14")      # 28.14 > 28 -> refused locally
            self.assertEqual(len(mock.requests), 2)       # the refused call never reached the server


def seed_city_list(work: Path, n: int = 4, regions: list[str] | None = None) -> pd.DataFrame:
    """Offline stand-in for the PSGC-derived city list (same schema as build_city_list)."""
    coords = [(14.60, 120.98), (10.32, 123.88), (7.12, 125.55), (16.41, 120.59), (9.80, 118.75), (6.99, 122.09)]
    rows = []
    for i in range(n):
        lat, lon = coords[i % len(coords)]
        rows.append({"city_id": f"99{i:08d}", "City": f"TestCity{i}", "psgc_name": f"City of TestCity{i}",
                     "province_or_group": "Testprov", "region": (regions or ["National Capital Region (NCR)"])[i % len(regions or [1])],
                     "city_class": "Component City",
                     "population_2024": 100000 + i, "latitude": lat + 0.01 * i, "longitude": lon + 0.01 * i,
                     "coord_method": "population_weighted_barangay_centroids", "n_barangays": 10,
                     "n_barangays_used": 10, "psgc_area_centroid_latitude": lat, "psgc_area_centroid_longitude": lon,
                     "shift_vs_area_centroid_km": 1.0})
    df = pd.DataFrame(rows)
    work.mkdir(parents=True, exist_ok=True)
    df.to_csv(work / "city_list_base.csv", index=False)
    return df


class PipelineTests(unittest.TestCase):
    START, END = "2015-01-01", "2016-03-15"      # 2 chunks per city, small and fast

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name) / "work"
        self.out = Path(self._tmp.name) / "out"
        self.sleeps = []
        self._orig_sleep = m._sleep
        m._sleep = self.sleeps.append

    def tearDown(self):
        m._sleep = self._orig_sleep
        self._tmp.cleanup()

    def run_main(self, mock, *extra, n_cities=4, start=None, end=None):
        argv = ["--work-dir", str(self.work), "--out-dir", str(self.out), "--api-url", mock.url,
                "--start", start or self.START, "--end", end or self.END, "--backoff-base", "0.001",
                "--minute-budget", "0", "--hour-budget", "0", "--day-budget", "0", *extra]
        return m.main(argv)

    def test_end_to_end_outputs_and_invariants(self):
        seed_city_list(self.work, 4)
        with MockOpenMeteo() as mock:
            rc = self.run_main(mock, "--max-cities", "4")
            n_requests = len(mock.requests)
        self.assertEqual(rc, 0)
        for name in ("ph_heat_index_next_day.csv", "ph_heat_index_data_dictionary.csv",
                     "ph_heat_index_city_list.csv", "dataset_documentation.md"):
            self.assertTrue((self.out / name).exists(), name)
        self.assertEqual(n_requests, 1 + 4 * 2)                                  # preflight + 4 cities x 2 year chunks
        df = pd.read_csv(self.out / "ph_heat_index_next_day.csv", parse_dates=["Date"])
        self.assertEqual(list(df.columns), m.FINAL_COLUMNS)
        n_days = (pd.Timestamp(self.END) - pd.Timestamp(self.START)).days + 1
        self.assertEqual(df.shape, (4 * (n_days - 1), len(m.FINAL_COLUMNS)))
        self.assertEqual(df["Date"].max(), pd.Timestamp(self.END) - pd.Timedelta(days=1))   # last day dropped
        self.assertTrue(df.sort_values(["City", "Date"]).reset_index(drop=True).equals(df))
        self.assertEqual(int(df.isna().sum().sum()), 0)
        self.assertFalse(df.duplicated(["City", "Date"]).any())
        # tomorrow == next day's today, exactly, within each city
        nxt = df.groupby("City")["HeatIndex_Max_Today"].shift(-1)
        ok = nxt.notna()
        self.assertTrue((df.loc[ok, "HeatIndex_Max_Tomorrow"] == nxt[ok]).all())
        self.assertEqual(int((~ok).sum()), 4)
        # labels re-derivable from numbers
        self.assertEqual(df["HeatLevelTomorrow"].tolist(), m.classify_heat_level(df["HeatIndex_Max_Tomorrow"]).tolist())
        # only the two target columns look ahead
        self.assertEqual([c for c in df.columns if "Tomorrow" in c], ["HeatIndex_Max_Tomorrow", "HeatLevelTomorrow"])
        self.assertTrue((df["Month"] == df["Date"].dt.month).all())
        # companion files
        dd = pd.read_csv(self.out / "ph_heat_index_data_dictionary.csv")
        self.assertEqual(dd["column"].tolist(), m.FINAL_COLUMNS)
        cl = pd.read_csv(self.out / "ph_heat_index_city_list.csv")
        self.assertTrue({"elevation_m", "grid_latitude", "grid_longitude"} <= set(cl.columns))
        self.assertEqual(len(cl), 4)
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        for needle in ("archive-api.open-meteo.com/v1/archive", "Asia/Manila", "not PAGASA station observations",
                       "derived from the numeric heat index", "Rothfusz", "Known limitations",
                       "temperature_2m", "Extreme Danger"):
            self.assertIn(needle, doc)

    def test_raw_retention_default_and_api_key_option(self):
        import gzip
        seed_city_list(self.work, 1)
        with MockOpenMeteo() as mock:
            rc = self.run_main(mock, "--max-cities", "1", "--api-key", "secret123")
            reqs = list(mock.requests)
        self.assertEqual(rc, 0)
        self.assertTrue(all(r["apikey"] == "secret123" for r in reqs))             # key sent on every call
        raws = sorted((self.work / "raw").rglob("*.json.gz"))
        self.assertEqual(len(raws), 2)                                              # one per city-year chunk
        with gzip.open(raws[0], "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(len(payload["hourly"]["time"]), 365 * 24)
        self.assertFalse((self.work / "api_usage_log.json").exists())               # key => free-tier usage log untouched

    REGIONS = ["National Capital Region (NCR)", "Region VII (Central Visayas)",
               "National Capital Region (NCR)", "Region XI (Davao Region)"]

    def test_region_filter_selects_only_matching_cities_and_labels_scope(self):
        seed_city_list(self.work, 4, regions=self.REGIONS)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--region", "NCR"), 0)
            n_requests = len(mock.data_requests())
        self.assertEqual(n_requests, 1 + 2 * 2)                      # preflight + 2 NCR cities x 2 year chunks
        df = pd.read_csv(self.out / "ph_heat_index_next_day.csv")
        self.assertEqual(sorted(df["City"].unique()), ["TestCity0", "TestCity2"])
        cl = pd.read_csv(self.out / "ph_heat_index_city_list.csv")
        self.assertEqual(set(cl["region"]), {"National Capital Region (NCR)"})
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        self.assertIn("Scope: National Capital Region (NCR) only", doc)
        self.assertIn("2 of the 4", doc)
        self.assertNotIn("SUBSET build", doc)                  # an intentional region scope is not a partial build
        self.assertNotIn("full official city list", doc)

    def test_default_scope_is_ncr_and_all_is_explicit(self):
        seed_city_list(self.work, 4, regions=self.REGIONS)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock), 0)                                  # no --region given
        self.assertEqual(sorted(pd.read_csv(self.out / "ph_heat_index_next_day.csv")["City"].unique()),
                         ["TestCity0", "TestCity2"])
        with MockOpenMeteo() as mock2:
            self.assertEqual(self.run_main(mock2, "--region", "all"), 0)
        self.assertEqual(pd.read_csv(self.out / "ph_heat_index_next_day.csv")["City"].nunique(), 4)
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        self.assertIn("outside the HeatCast NCR project scope", doc)

    def test_region_alias_and_multiple_regions(self):
        cities = seed_city_list(self.work, 4, regions=self.REGIONS)
        self.assertEqual(len(m.filter_regions(cities, ["Metro Manila"])), 2)
        self.assertEqual(len(m.filter_regions(cities, ["ncr", "davao"])), 3)
        self.assertEqual(len(m.filter_regions(cities, ["ncr,central visayas"])), 3)
        self.assertEqual(len(m.filter_regions(cities, ["Region VII (Central Visayas)"])), 1)   # parentheses are literal

    def test_unknown_region_fails_fast_listing_choices(self):
        seed_city_list(self.work, 4, regions=self.REGIONS)
        with MockOpenMeteo() as mock:
            rc = self.run_main(mock, "--region", "Atlantis")
            self.assertEqual(mock.requests, [])
        self.assertEqual(rc, 1)
        log = (self.work / "run_log.txt").read_text(encoding="utf-8")
        self.assertIn("Available regions", log)
        self.assertIn("National Capital Region (NCR)", log)
        self.assertFalse((self.out / "ph_heat_index_next_day.csv").exists())

    def test_region_run_reuses_cache_from_a_full_run(self):
        seed_city_list(self.work, 4, regions=self.REGIONS)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--region", "all", "--max-cities", "4"), 0)
        with MockOpenMeteo() as mock2:
            self.assertEqual(self.run_main(mock2, "--region", "NCR"), 0)
            self.assertEqual(mock2.requests, [])
        self.assertEqual(pd.read_csv(self.out / "ph_heat_index_next_day.csv")["City"].nunique(), 2)

    def test_region_with_max_cities_still_flags_subset(self):
        seed_city_list(self.work, 4, regions=self.REGIONS)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--region", "NCR", "--max-cities", "1"), 0)
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        self.assertIn("SUBSET build", doc)
        self.assertIn("1 of 2 cities", doc)

    def test_analysis_report_is_written_and_documented(self):
        seed_city_list(self.work, 4)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock), 0)
        report = (self.out / "ncr_dataset_analysis.md").read_text(encoding="utf-8")
        for heading in ("## Key findings", "## 1. How many distinct grid series", "## 3. Class counts",
                        "## 5. Shared grid series and leakage", "## 8. Recommendations"):
            self.assertIn(heading, report)
        self.assertIn("No model has been trained", report)
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        self.assertIn("## Shared grid series - read before modelling", doc)
        self.assertIn("not 4 independent weather stations", doc)
        self.assertIn("gridded ERA5 reanalysis estimates for representative city coordinates", doc)
        self.assertIn("HeatCast NCR", doc)

    def test_analysis_failure_is_loud_but_keeps_the_dataset(self):
        import ncr_dataset_analysis as NA
        seed_city_list(self.work, 2)
        orig = NA.run_analysis
        NA.run_analysis = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with MockOpenMeteo() as mock:
                rc = self.run_main(mock)
        finally:
            NA.run_analysis = orig
        self.assertEqual(rc, 1)
        self.assertTrue((self.out / "ph_heat_index_next_day.csv").exists())           # valid dataset is not discarded
        self.assertIn("the analysis failed", (self.work / "run_log.txt").read_text(encoding="utf-8"))

    def test_very_short_range_does_not_break_the_analysis(self):
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            rc = self.run_main(mock, start="2018-03-01", end="2018-03-20")
        self.assertEqual(rc, 0)                                   # 19 days, no lag rows, absent classes: must still report
        report = (self.out / "ncr_dataset_analysis.md").read_text(encoding="utf-8")
        self.assertIn("## Key findings", report)
        self.assertIn("ABSENT", report)                           # absent classes are flagged, not hidden
        self.assertNotIn("2,018", report)                         # labels such as years are not formatted as quantities

    def test_real_ncr_city_list_has_the_16_psgc_cities(self):
        ref = pd.read_csv(ROOT / "reference" / "ph_city_list_base.csv")
        ncr = ref[ref["region"].str.contains("NCR")]
        self.assertEqual(sorted(ncr["City"]), sorted([
            "Caloocan", "Las Piñas", "Makati", "Malabon", "Mandaluyong", "Manila", "Marikina", "Muntinlupa", "Navotas",
            "Parañaque", "Pasay", "Pasig", "Quezon", "San Juan", "Taguig", "Valenzuela"]))
        self.assertEqual(len(m.filter_regions(ref, ["NCR"])), 16)                     # Pateros is a municipality, not a city

    def test_no_keep_raw_flag(self):
        seed_city_list(self.work, 1)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "1", "--no-keep-raw"), 0)
        self.assertFalse((self.work / "raw").exists())

    def test_rebuild_from_raw_costs_no_api_calls(self):
        """Deleting the aggregated cache, or bumping AGG_VERSION, must not need the API again."""
        import shutil
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
        reference = (self.out / "ph_heat_index_next_day.csv").read_bytes()
        shutil.rmtree(self.work / "chunks")
        orig_agg = m.AGG_VERSION
        try:
            m.AGG_VERSION = "999"                                      # new aggregation logic => new chunk cache key
            with MockOpenMeteo() as mock2:
                self.assertEqual(self.run_main(mock2, "--max-cities", "2"), 0)
                self.assertEqual(mock2.requests, [])                    # not even a preflight call
        finally:
            m.AGG_VERSION = orig_agg
        self.assertEqual((self.out / "ph_heat_index_next_day.csv").read_bytes(), reference)

    def test_corrupt_raw_file_falls_back_to_api_for_that_chunk_only(self):
        import shutil
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
        shutil.rmtree(self.work / "chunks")
        victim = sorted((self.work / "raw").rglob("*.json.gz"))[2]
        victim.write_bytes(victim.read_bytes()[:200])                   # truncated gzip
        with MockOpenMeteo() as mock2:
            self.assertEqual(self.run_main(mock2, "--max-cities", "2"), 0)
            self.assertEqual(len(mock2.data_requests()), 1)             # only the unreadable chunk (no preflight: raw exists)

    def test_subset_build_is_labelled_in_docs_and_city_list(self):
        seed_city_list(self.work, 4)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
        doc = (self.out / "dataset_documentation.md").read_text(encoding="utf-8")
        self.assertIn("SUBSET build", doc)
        self.assertIn("2 of 4 cities", doc)
        self.assertEqual(len(pd.read_csv(self.out / "ph_heat_index_city_list.csv")), 2)

    def test_rerun_with_full_cache_makes_no_api_calls(self):
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
            first = (self.out / "ph_heat_index_next_day.csv").read_bytes()
            before = len(mock.requests)
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
            self.assertEqual(len(mock.requests), before)
        self.assertEqual((self.out / "ph_heat_index_next_day.csv").read_bytes(), first)    # deterministic

    def test_resume_after_daily_limit_never_refetches(self):
        seed_city_list(self.work, 3)                      # 3 cities x 2 chunks = 6 chunks
        w1 = m.call_weight(365) + 1.0                     # budget: preflight(1) + exactly 1 full-year chunk
        with MockOpenMeteo(daily_weight_limit=w1) as mock:
            rc1 = self.run_main(mock, "--max-cities", "3", "--skip-preflight")
            served_1 = [r for r in mock.requests if r["served"] == "data"]
        self.assertEqual(len(served_1), 1)                # quota covered exactly one full-year chunk
        self.assertEqual(rc1, 2)                          # incomplete -> exit code 2
        cached = sorted((self.work / "chunks").rglob("*.csv"))
        self.assertEqual(len(cached), 1)
        with MockOpenMeteo() as mock2:                    # next day: unlimited quota
            rc2 = self.run_main(mock2, "--max-cities", "3", "--skip-preflight")
            served_2 = [(r["lat"], r["start"]) for r in mock2.requests if r["served"] == "data"]
        self.assertEqual(rc2, 0)
        self.assertEqual(len(served_2), 6 - len(served_1))   # only the missing chunks were fetched
        df = pd.read_csv(self.out / "ph_heat_index_next_day.csv")
        self.assertEqual(df["City"].nunique(), 3)

    def test_incomplete_run_writes_nothing_without_allow_partial(self):
        seed_city_list(self.work, 3)
        # no --max-cities => partial output not allowed
        with MockOpenMeteo(daily_weight_limit=m.call_weight(365) + 1) as mock:
            rc = self.run_main(mock, "--skip-preflight")
        self.assertEqual(rc, 2)
        self.assertFalse((self.out / "ph_heat_index_next_day.csv").exists())

    def test_corrupt_cache_chunk_is_rebuilt_from_raw_without_api(self):
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
        reference = (self.out / "ph_heat_index_next_day.csv").read_bytes()
        victim = sorted((self.work / "chunks").rglob("*.csv"))[1]
        victim.write_text(victim.read_text()[:300])                      # truncate mid-file
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2"), 0)
            self.assertEqual(mock.requests, [])                          # repaired from retained raw data
        self.assertEqual((self.out / "ph_heat_index_next_day.csv").read_bytes(), reference)

    def test_corrupt_cache_chunk_without_raw_is_refetched_alone(self):
        import shutil
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2", "--no-keep-raw"), 0)
        victim = sorted((self.work / "chunks").rglob("*.csv"))[1]
        victim.write_text(victim.read_text()[:300])
        with MockOpenMeteo() as mock:
            self.assertEqual(self.run_main(mock, "--max-cities", "2", "--no-keep-raw"), 0)
            self.assertEqual(len(mock.data_requests()), 1 + 1)           # preflight + exactly the damaged chunk

    def test_transient_faults_during_collection_do_not_change_result(self):
        seed_city_list(self.work, 2)
        with MockOpenMeteo() as clean:
            self.assertEqual(self.run_main(clean, "--max-cities", "2"), 0)
        reference = (self.out / "ph_heat_index_next_day.csv").read_bytes()
        (self.out / "ph_heat_index_next_day.csv").unlink()
        import shutil
        shutil.rmtree(self.work / "chunks")
        with MockOpenMeteo() as flaky:
            for kind in ("http500", "truncated", "rate_minutely", "short", "http503"):
                flaky.inject(kind)
            self.assertEqual(self.run_main(flaky, "--max-cities", "2"), 0)
        self.assertEqual((self.out / "ph_heat_index_next_day.csv").read_bytes(), reference)

    def test_preflight_failure_stops_run_before_any_chunk(self):
        seed_city_list(self.work, 2)
        with MockOpenMeteo(null_vars={"cloud_cover"}) as mock:
            rc = self.run_main(mock, "--max-cities", "2")
            self.assertEqual(len(mock.requests), 1)
        self.assertEqual(rc, 1)
        self.assertFalse((self.work / "chunks").exists() and any((self.work / "chunks").rglob("*.csv")))

    def test_model_change_uses_a_separate_cache(self):
        self.assertNotEqual(m.cache_key("era5"), m.cache_key("era5_seamless"))
        self.assertEqual(m.cache_key("era5"), m.cache_key("era5"))

    def test_chunk_plan_is_calendar_years_clipped_to_range(self):
        cities = pd.DataFrame({"city_id": ["1"], "City": ["A"], "latitude": [14.0], "longitude": [121.0]})
        plan = m.plan_chunks(cities, "2015-01-01", "2025-12-29")
        self.assertEqual(len(plan), 11)
        self.assertEqual((plan[0].start, plan[0].end), ("2015-01-01", "2015-12-31"))
        self.assertEqual((plan[-1].start, plan[-1].end), ("2025-01-01", "2025-12-29"))
        self.assertEqual(sum(c.n_days for c in plan), 4016)
        self.assertEqual(plan[1].n_days, 366)                                  # 2016 leap year


class ValidatorTests(unittest.TestCase):
    """validate_dataset must catch deliberately broken data (run on a clean build, then corrupt it)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        seed_city_list(tmp / "w", 3)
        orig = m._sleep
        m._sleep = lambda s: None
        try:
            with MockOpenMeteo() as mock:
                rc = m.main(["--work-dir", str(tmp / "w"), "--out-dir", str(tmp / "o"), "--api-url", mock.url,
                             "--start", "2018-01-01", "--end", "2018-06-30", "--max-cities", "3",
                             "--minute-budget", "0", "--hour-budget", "0", "--day-budget", "0"])
        finally:
            m._sleep = orig
        assert rc == 0
        cls.df = pd.read_csv(tmp / "o" / "ph_heat_index_next_day.csv", parse_dates=["Date"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def check(self, df):
        return m.validate_dataset(df, "2018-01-01", "2018-06-30")

    def test_clean_data_passes(self):
        r = self.check(self.df)
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["checks"]["tomorrow_equals_next_day_today"]["mismatches_shift"], 0)
        self.assertEqual(r["checks"]["tomorrow_equals_next_day_today"]["mismatches_merge"], 0)

    def test_detects_missing_values(self):
        d = self.df.copy(); d.loc[5, "Temp_Max"] = np.nan
        self.assertTrue(any("missing" in e for e in self.check(d)["errors"]))

    def test_detects_duplicates(self):
        d = pd.concat([self.df, self.df.iloc[[10]]], ignore_index=True)
        self.assertTrue(any("duplicates" in e for e in self.check(d)["errors"]))

    def test_detects_gap(self):
        d = self.df.drop(index=20).reset_index(drop=True)
        self.assertTrue(any("gaps" in e for e in self.check(d)["errors"]))

    def test_detects_impossible_values(self):
        d = self.df.copy()
        d.loc[3, "Humidity_Max"] = 130.0
        d.loc[4, "Rainfall_Total"] = -1.0
        d.loc[6, "Temp_Min"] = d.loc[6, "Temp_Max"] + 2
        errs = " ".join(self.check(d)["errors"])
        self.assertIn("Humidity_Max", errs)
        self.assertIn("Rainfall_Total", errs)
        self.assertIn("Temp_Min<=Temp_Mean<=Temp_Max", errs)

    def test_detects_broken_target_shift(self):
        d = self.df.copy(); d.loc[50, "HeatIndex_Max_Tomorrow"] += 0.01
        r = self.check(d)
        self.assertGreater(r["checks"]["tomorrow_equals_next_day_today"]["mismatches_shift"], 0)
        self.assertTrue(any("HeatIndex_Max_Tomorrow !=" in e for e in r["errors"]))

    def test_detects_label_disagreement(self):
        d = self.df.copy()
        d.loc[7, "HeatLevelTomorrow"] = "Extreme Danger" if d.loc[7, "HeatLevelTomorrow"] != "Extreme Danger" else "Caution"
        self.assertTrue(any("HeatLevelTomorrow disagrees" in e for e in self.check(d)["errors"]))

    def test_detects_unsorted_and_extra_future_columns(self):
        d = self.df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        self.assertTrue(any("not sorted" in e for e in self.check(d)["errors"]))
        d2 = self.df.copy(); d2["Temp_Max_Tomorrow"] = d2["Temp_Max"].shift(-1)
        errs = " ".join(self.check(d2)["errors"])
        self.assertIn("unexpected next-day columns", errs)

    def test_validate_only_cli(self):
        path = Path(self._tmp.name) / "o" / "ph_heat_index_next_day.csv"
        cwd_before = set(Path.cwd().iterdir())
        self.assertEqual(m.main(["--validate-only", str(path), "--start", "2018-01-01", "--end", "2018-06-30"]), 0)
        self.assertEqual(set(Path.cwd().iterdir()), cwd_before)                  # read-only: no cache dir created
        bad = self.df.copy(); bad.loc[0, "Humidity_Max"] = 150.0
        bad_path = Path(self._tmp.name) / "bad.csv"; bad.to_csv(bad_path, index=False)
        self.assertEqual(m.main(["--validate-only", str(bad_path), "--start", "2018-01-01", "--end", "2018-06-30"]), 1)

    def test_class_distribution_sums_to_100(self):
        dist = self.check(self.df)["checks"]["class_distribution"]["HeatLevelTomorrow"]
        self.assertAlmostEqual(float(dist["percent"].sum()), 100.0, delta=0.05)
        self.assertEqual(int(dist["count"].sum()), len(self.df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
