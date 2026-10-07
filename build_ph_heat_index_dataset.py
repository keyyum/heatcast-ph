#!/usr/bin/env python3
"""
Build a reproducible Philippine next-day heat-index ML dataset.

Source   : Open-Meteo Historical Weather API (ERA5 reanalysis), hourly data
Period   : 2015-01-01 .. 2025-12-29, timezone Asia/Manila
Locations: the 149 Philippine cities of the PSA PSGC (see build_city_list)
Output   : one row per city-day with same-day weather aggregates, the NWS
           heat index computed from hourly temperature + humidity, and the
           next-day heat index / heat level as targets.

Usage (terminal or Colab):
    python build_ph_heat_index_dataset.py --work-dir ph_cache --out-dir out

The run is resumable: every completed (city, year) chunk is cached in
``--work-dir`` and re-runs skip it. The free Open-Meteo tier allows
10,000 weighted calls/day and this job costs ~42,700, so expect the first run
to stop gracefully on the daily budget; simply re-run it the next day.

Exit codes: 0 = dataset built and validated, 1 = error / validation failure,
2 = incomplete (API budget reached or --max-cities subset not allowed to
write without --allow-partial).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:  # tqdm is preinstalled in Colab; degrade gracefully elsewhere
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SCRIPT_VERSION = "1.0.0"
# Bump when aggregation / heat-index logic changes: invalidates the chunk cache.
AGG_VERSION = "1"

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
API_URL = "https://archive-api.open-meteo.com/v1/archive"
CUSTOMER_API_URL = "https://customer-archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Asia/Manila"
UTC_OFFSET_SECONDS = 8 * 3600  # Philippines: UTC+8, no DST
DEFAULT_MODEL = "era5"  # pinned: best_match silently switches to IFS from 2017
CELL_SELECTION = "land"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-29"
HOURS_PER_DAY = 24
RAIN_HOUR_THRESHOLD_MM = 0.1  # an hour counts as a "rain hour" at >= 0.1 mm

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
    "rain",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
]
EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "pressure_msl": "hPa",
    "cloud_cover": "%",
    "precipitation": "mm",
    "rain": "mm",
    "shortwave_radiation": "W/m²",
    "wind_speed_10m": "km/h",
    "wind_gusts_10m": "km/h",
}

# Free-tier limits are 600/min, 5,000/hour, 10,000/day (weighted calls).
# Client-side budgets stay below them. 0 / None disables a window.
DEFAULT_BUDGETS = {"minute": 500.0, "hour": 4500.0, "day": 9500.0}

# City list source: pinned PyPI wheel of the community PSGC package.
PSGC_VERSION = "2026.4.13.0"
PSGC_WHEEL_SHA256 = "e9662b69f3313d90089896c567d9b82be2452358d16f3eb561a9f44b5212a778"
PSGC_PYPI_JSON = f"https://pypi.org/pypi/psgc/{PSGC_VERSION}/json"

HEAT_LEVELS = ["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]
# Lower bound inclusive: [-inf,27) [27,33) [33,42) [42,52) [52,inf)
HEAT_BINS = [-math.inf, 27.0, 33.0, 42.0, 52.0, math.inf]

FINAL_COLUMNS = [
    "Date", "City", "Latitude", "Longitude", "Elevation", "Month", "DayOfYear",
    "Temp_Min", "Temp_Mean", "Temp_Max",
    "Humidity_Min", "Humidity_Mean", "Humidity_Max",
    "DewPoint_Mean", "Pressure_Mean", "CloudCover_Mean",
    "WindSpeed_Mean", "WindSpeed_Max", "WindGust_Max",
    "Rainfall_Total", "RainHours", "SolarRadiation_Total",
    "HeatIndex_Max_Today", "HeatLevelToday",
    "HeatIndex_Max_Tomorrow", "HeatLevelTomorrow",
]
# (output column, hourly source variable, aggregation)
AGG_SPEC = [
    ("Temp_Min", "temperature_2m", "min"),
    ("Temp_Mean", "temperature_2m", "mean"),
    ("Temp_Max", "temperature_2m", "max"),
    ("Humidity_Min", "relative_humidity_2m", "min"),
    ("Humidity_Mean", "relative_humidity_2m", "mean"),
    ("Humidity_Max", "relative_humidity_2m", "max"),
    ("DewPoint_Mean", "dew_point_2m", "mean"),
    ("Pressure_Mean", "pressure_msl", "mean"),
    ("CloudCover_Mean", "cloud_cover", "mean"),
    ("WindSpeed_Mean", "wind_speed_10m", "mean"),
    ("WindSpeed_Max", "wind_speed_10m", "max"),
    ("WindGust_Max", "wind_gusts_10m", "max"),
    ("Rainfall_Total", "rain", "sum"),
    ("SolarRadiation_Total", "shortwave_radiation", "sum"),
]
DAILY_FEATURE_COLUMNS = [c for c, _, _ in AGG_SPEC[:12]] + ["Rainfall_Total", "RainHours", "SolarRadiation_Total"]
ROUND_DP = {"SolarRadiation_Total": 1}  # everything else 2 dp
CHUNK_COLUMNS = (
    ["Date"] + [c for c, _, _ in AGG_SPEC] + ["RainHours", "HeatIndex_Max_Today"]
    + ["_HoursMin", "_PrecipTotal", "_GridLat", "_GridLon", "_ApiElevation"]
)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
class Logger:
    """Print (tqdm-safe) and append to a log file."""

    def __init__(self, path: Path | None = None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        if tqdm is not None:
            tqdm.write(msg)
        else:  # pragma: no cover
            print(msg, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}\n")


def _sleep(seconds: float) -> None:
    """Indirection so tests can patch sleeping."""
    time.sleep(max(0.0, seconds))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --------------------------------------------------------------------------- #
# Heat index (NWS / Rothfusz regression) and heat-level classification
# --------------------------------------------------------------------------- #
def heat_index_celsius(temp_c, rh_pct) -> np.ndarray:
    """NWS heat index in °C from air temperature (°C) and relative humidity (%).

    Algorithm (NWS WPC "The Heat Index Equation", Rothfusz 1990):
      1. Steadman simple formula (T in °F):
            HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)
      2. If the mean of that value and T is >= 80 °F, use the full Rothfusz
         regression and apply the two NWS adjustments:
            RH < 13 %, 80 <= T <= 112 °F : subtract ((13-RH)/4)*sqrt((17-|T-95|)/17)
            RH > 85 %, 80 <= T <=  87 °F : add      ((RH-85)/10)*((87-T)/5)
      3. Otherwise keep the simple-formula value.
    Result is converted back to °C. Inputs outside physical bounds
    (T not in [-90, 60] °C, RH not in [0, 100] %) or missing -> NaN.
    """
    t_c = np.asarray(temp_c, dtype="float64")
    rh = np.asarray(rh_pct, dtype="float64")
    valid = (t_c >= -90.0) & (t_c <= 60.0) & (rh >= 0.0) & (rh <= 100.0)  # NaN -> False
    t = t_c * 9.0 / 5.0 + 32.0
    with np.errstate(invalid="ignore"):
        simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
        regression = (
            -42.379
            + 2.04901523 * t
            + 10.14333127 * rh
            - 0.22475541 * t * rh
            - 0.00683783 * t * t
            - 0.05481717 * rh * rh
            + 0.00122874 * t * t * rh
            + 0.00085282 * t * rh * rh
            - 0.00000199 * t * t * rh * rh
        )
        radicand = np.clip((17.0 - np.abs(t - 95.0)) / 17.0, 0.0, None)
        adj_dry = np.where((rh < 13.0) & (t >= 80.0) & (t <= 112.0), -((13.0 - rh) / 4.0) * np.sqrt(radicand), 0.0)
        adj_humid = np.where((rh > 85.0) & (t >= 80.0) & (t <= 87.0), ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0), 0.0)
        hi_f = np.where((simple + t) / 2.0 >= 80.0, regression + adj_dry + adj_humid, simple)
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return np.where(valid, hi_c, np.nan)


def classify_heat_level(hi_c) -> pd.Series:
    """Heat level from a (rounded) °C heat index; NaN stays NaN. Lower bounds inclusive.

    The index of a pandas input is preserved so the result can be assigned back safely."""
    index = hi_c.index if isinstance(hi_c, pd.Series) else None
    s = pd.Series(np.asarray(hi_c, dtype="float64"), index=index)
    return pd.cut(s, bins=HEAT_BINS, labels=HEAT_LEVELS, right=False).astype(object)


# --------------------------------------------------------------------------- #
# City list: official PSGC cities + population-weighted coordinates
# --------------------------------------------------------------------------- #
_NAME_OVERRIDES = {"Cagayan De Oro": "Cagayan de Oro", "San Jose Del Monte": "San Jose del Monte", "Sto. Tomas": "Santo Tomas"}


def clean_city_name(psgc_name: str) -> str:
    """'City of Davao' -> 'Davao', 'Quezon City' -> 'Quezon', 'Island Garden City of Samal' -> 'Samal'."""
    name = re.sub(r"^.*?\bCity of\s+", "", psgc_name)
    name = re.sub(r"\s+City$", "", name).strip()
    return _NAME_OVERRIDES.get(name, name)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180.0
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def fetch_psgc_wheel(cache_dir: Path, log: Logger) -> Path:
    """Download the pinned psgc wheel from PyPI and verify its SHA-256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"psgc-{PSGC_VERSION}-py3-none-any.whl"

    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    if path.exists() and sha256(path) == PSGC_WHEEL_SHA256:
        return path
    log(f"Downloading psgc=={PSGC_VERSION} from PyPI (city list source)...")
    meta = requests.get(PSGC_PYPI_JSON, timeout=60)
    meta.raise_for_status()
    wheels = [u for u in meta.json()["urls"] if u["packagetype"] == "bdist_wheel"]
    if not wheels:
        raise RuntimeError(f"No wheel found for psgc=={PSGC_VERSION}")
    tmp = path.with_name(path.name + ".tmp")
    with requests.get(wheels[0]["url"], timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    if sha256(tmp) != PSGC_WHEEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("psgc wheel SHA-256 mismatch - refusing to use it as a city-list source")
    os.replace(tmp, path)
    return path


def build_city_list(cache_dir: Path, log: Logger) -> pd.DataFrame:
    """Return one row per Philippine city (PSGC geographic level 'City', n=149).

    Coordinates are the population-weighted mean of the city's barangay centroids
    (2024 census population; barangay centroids from OCHA/HDX 2023 boundaries as
    packaged in psgc). This approximates where people live, unlike the package's
    area-weighted city centroid, which can sit tens of km from the urban core in
    large cities (e.g. Davao).  The wheel is only read as data, never imported.
    """
    wheel = fetch_psgc_wheel(cache_dir, log)
    with zipfile.ZipFile(wheel) as zf:
        load = lambda n: json.loads(zf.read(f"psgc/data/core/{n}.json"))  # noqa: E731
        all_cities, provinces, regions, barangays = (load(n) for n in ("cities", "provinces", "regions", "barangays"))
    prov_name = {p["psgc_code"]: p["name"] for p in provinces}
    reg_name = {r["psgc_code"]: r["name"] for r in regions}
    submuns = [c for c in all_cities if c["geographic_level"] == "SubMun"]
    by_city: dict[str, list] = {}
    for b in barangays:
        by_city.setdefault(b["city_code"], []).append((b["population"], b["coordinate"]["latitude"],
                                                       b["coordinate"]["longitude"], b["coordinate_source"]))
    del barangays

    rows = []
    for c in all_cities:
        if c["geographic_level"] != "City":
            continue
        codes = {c["psgc_code"]} | {s["psgc_code"] for s in submuns if s["psgc_code"][:5] == c["psgc_code"][:5]}
        brgys = [b for k in codes for b in by_city.get(k, [])]
        usable = [b for b in brgys if b[3] != "fallback_unverified" and b[0] and b[0] > 0]
        area_lat, area_lon = c["coordinate"]["latitude"], c["coordinate"]["longitude"]
        if usable:
            wsum = float(sum(b[0] for b in usable))
            lat = sum(b[0] * b[1] for b in usable) / wsum
            lon = sum(b[0] * b[2] for b in usable) / wsum
            method = "population_weighted_barangay_centroids"
        else:  # not expected for the pinned release; kept as an explicit fallback
            lat, lon, method = area_lat, area_lon, "psgc_area_weighted_centroid_fallback"
        rows.append({
            "city_id": c["psgc_code"],
            "psgc_name": c["name"],
            "province_or_group": prov_name.get(c["province_code"], ""),
            "region": reg_name.get(c["region_code"], ""),
            "city_class": c.get("city_class", ""),
            "population_2024": c.get("population"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "coord_method": method,
            "n_barangays": len(brgys),
            "n_barangays_used": len(usable),
            "psgc_area_centroid_latitude": round(area_lat, 5),
            "psgc_area_centroid_longitude": round(area_lon, 5),
            "shift_vs_area_centroid_km": round(haversine_km(lat, lon, area_lat, area_lon), 2),
        })
    df = pd.DataFrame(rows).sort_values("city_id").reset_index(drop=True)

    names = df["psgc_name"].map(clean_city_name)
    dup = names.duplicated(keep=False)
    df.insert(1, "City", [f"{n} ({p})" if d else n for n, p, d in zip(names, df["province_or_group"], dup)])
    if df["City"].duplicated().any():
        raise RuntimeError(f"City keys not unique: {df.loc[df['City'].duplicated(keep=False), 'City'].tolist()}")
    if len(df) != 149:
        log(f"WARNING: expected 149 PSGC cities, found {len(df)}")
    return df


# --------------------------------------------------------------------------- #
# Rate budget + Open-Meteo client
# --------------------------------------------------------------------------- #
class DailyLimitReached(RuntimeError):
    """Daily call budget exhausted (local budget or server 429)."""

    def __init__(self, msg: str, wait_seconds: float | None = None):
        super().__init__(msg)
        self.wait_seconds = wait_seconds


class FatalApiError(RuntimeError):
    """Non-retryable API / configuration error."""


class ApiDataError(RuntimeError):
    """Retryable: response was 200 but malformed / truncated."""


def call_weight(n_days: int, n_vars: int = len(HOURLY_VARS), n_models: int = 1) -> float:
    """Open-Meteo weighted call count: max(1, (days/14) * (vars*models/10)) per location."""
    var_w = n_vars * max(n_models, 1) / 10.0
    return max(1.0, max(var_w, (n_days / 14.0) * var_w))


class RateBudget:
    """Sliding-window client-side budget (weighted calls) persisted across runs."""

    WINDOWS = (("minute", 60), ("hour", 3600), ("day", 86400))

    def __init__(self, limits: dict | None, usage_path: Path | None = None, log: Logger | None = None,
                 clock=time.time, sleep=None):
        self.limits = {k: float(v) for k, v in (limits or {}).items() if v}
        self.usage_path = usage_path
        self.log = log or (lambda m: None)
        self.clock = clock
        self.sleep = sleep or _sleep
        self.events: deque = deque()
        if usage_path and usage_path.exists():
            try:
                now = self.clock()
                self.events = deque((t, w) for t, w in json.loads(usage_path.read_text()) if t > now - 86400)
            except (ValueError, OSError):
                self.events = deque()

    def _prune(self, now: float) -> None:
        while self.events and self.events[0][0] <= now - 86400:
            self.events.popleft()

    def used(self, window_s: int) -> float:
        now = self.clock()
        return sum(w for t, w in self.events if t > now - window_s)

    def acquire(self, weight: float) -> None:
        """Block until `weight` fits every window; raise DailyLimitReached for the day window."""
        while True:
            now = self.clock()
            self._prune(now)
            wait, why = 0.0, ""
            for name, window in self.WINDOWS:
                limit = self.limits.get(name)
                if not limit:
                    continue
                if weight > limit:
                    raise ValueError(f"call weight {weight:.1f} exceeds the {name} budget {limit:.0f}")
                used = sum(w for t, w in self.events if t > now - window)
                if used + weight <= limit + 1e-9:
                    continue
                need, acc, free_in = used + weight - limit, 0.0, float(window)
                for t, w in self.events:
                    if t <= now - window:
                        continue
                    acc += w
                    if acc >= need - 1e-9:
                        free_in = t + window - now
                        break
                if name == "day":
                    raise DailyLimitReached(
                        f"daily budget {limit:.0f} weighted calls reached ({used:.0f} used in the last 24h)",
                        wait_seconds=free_in)
                if free_in > wait:
                    wait, why = free_in, name
            if wait <= 0:
                return
            self.log(f"  rate budget: waiting {wait + 1:.0f}s for the {why} window to free up")
            self.sleep(wait + 1.0)

    def record(self, weight: float) -> None:
        self.events.append((self.clock(), weight))
        if self.usage_path:
            try:
                atomic_write_text(self.usage_path, json.dumps(list(self.events)))
            except OSError:
                pass


class OpenMeteoClient:
    def __init__(self, api_url: str, model: str, api_key: str | None = None, budget: RateBudget | None = None,
                 max_retries: int = 8, backoff_base: float = 5.0, backoff_cap: float = 300.0,
                 timeout=(15, 180), session: requests.Session | None = None, log: Logger | None = None):
        self.api_url, self.model, self.api_key = api_url, model, api_key
        self.budget = budget or RateBudget(None)
        self.max_retries, self.backoff_base, self.backoff_cap = max_retries, backoff_base, backoff_cap
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": f"heatcast-ph-dataset-builder/{SCRIPT_VERSION}"})
        self.log = log or (lambda m: None)
        self.calls_made = 0
        self.weight_spent = 0.0

    # ---- helpers ---------------------------------------------------------
    def _params(self, lat: float, lon: float, start: str, end: str) -> dict:
        p = {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY_VARS), "timezone": TIMEZONE, "models": self.model,
            "cell_selection": CELL_SELECTION, "temperature_unit": "celsius", "wind_speed_unit": "kmh",
            "precipitation_unit": "mm", "timeformat": "iso8601",
        }
        if self.api_key:
            p["apikey"] = self.api_key
        return p

    @staticmethod
    def _reason(resp: requests.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("reason"):
                return str(body["reason"])
        except ValueError:
            pass
        return (resp.text or "")[:200].strip() or f"HTTP {resp.status_code}"

    def _backoff(self, attempt: int, floor: float = 0.0) -> float:
        delay = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
        return max(floor, delay * (0.5 + random.random() / 2))

    @staticmethod
    def _validate(data: dict, start: str, end: str) -> None:
        if not isinstance(data, dict) or "hourly" not in data:
            raise ApiDataError(f"response has no 'hourly' block (keys={list(data)[:6] if isinstance(data, dict) else type(data)})")
        if data.get("utc_offset_seconds") != UTC_OFFSET_SECONDS or data.get("timezone") != TIMEZONE:
            raise FatalApiError(f"unexpected timezone in response: {data.get('timezone')!r} / "
                                f"offset {data.get('utc_offset_seconds')!r} (expected {TIMEZONE}, +8h)")
        hourly = data["hourly"]
        n_days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
        n_expected = n_days * HOURS_PER_DAY
        times = hourly.get("time")
        if not isinstance(times, list) or len(times) != n_expected:
            raise ApiDataError(f"expected {n_expected} hourly timestamps, got {len(times) if isinstance(times, list) else times!r}")
        if times[0] != f"{start}T00:00" or times[-1] != f"{end}T23:00":
            raise ApiDataError(f"timestamp range {times[0]}..{times[-1]} != {start}T00:00..{end}T23:00")
        for v in HOURLY_VARS:
            arr = hourly.get(v)
            if not isinstance(arr, list) or len(arr) != n_expected:
                raise ApiDataError(f"variable {v!r} missing or wrong length")
        units = data.get("hourly_units", {})
        if "C" not in str(units.get("temperature_2m", "")):
            raise FatalApiError(f"temperature_2m unit is {units.get('temperature_2m')!r}, expected Celsius")

    # ---- public ----------------------------------------------------------
    def fetch_hourly(self, lat: float, lon: float, start: str, end: str) -> dict:
        n_days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
        weight = call_weight(n_days)
        params = self._params(lat, lon, start, end)
        attempt, rate_waits = 0, 0
        while True:
            attempt += 1
            self.budget.acquire(weight)  # may sleep or raise DailyLimitReached
            self.budget.record(weight)
            self.calls_made += 1
            self.weight_spent += weight
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                if re.search(r"\b(403|407)\b", str(exc)):  # policy denial: retrying cannot help
                    host = re.sub(r"^https?://([^/]+).*$", r"\1", self.api_url)
                    raise FatalApiError(
                        f"a proxy/firewall refused the connection to {host} (HTTP 403/407 on CONNECT). The host is "
                        "probably blocked by the network policy of this environment; allow outbound HTTPS to it "
                        "(and to pypi.org / files.pythonhosted.org for the city list) and re-run.") from exc
                if attempt >= self.max_retries:
                    raise FatalApiError(f"proxy error after {attempt} attempts: {exc}") from exc
                wait = self._backoff(attempt)
                self.log(f"  proxy error; retry {attempt}/{self.max_retries} in {wait:.0f}s")
                _sleep(wait)
                continue
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise FatalApiError(f"network error after {attempt} attempts: {type(exc).__name__}: {str(exc)[:200]}") from exc
                wait = self._backoff(attempt)
                self.log(f"  network error ({type(exc).__name__}); retry {attempt}/{self.max_retries} in {wait:.0f}s")
                _sleep(wait)
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    self._validate(data, start, end)
                    return data
                except (ValueError, ApiDataError) as exc:
                    if attempt >= self.max_retries:
                        raise FatalApiError(f"malformed response after {attempt} attempts: {exc}") from exc
                    wait = self._backoff(attempt)
                    self.log(f"  malformed response ({exc}); retry {attempt}/{self.max_retries} in {wait:.0f}s")
                    _sleep(wait)
                    continue

            reason = self._reason(resp)
            if resp.status_code == 429:
                low = reason.lower()
                if "daily" in low:
                    raise DailyLimitReached(f"server: {reason}")
                retry_after = 0.0
                try:
                    retry_after = float(resp.headers.get("Retry-After", 0))
                except ValueError:
                    pass
                rate_waits += 1
                if rate_waits > 8:
                    raise FatalApiError(f"still rate limited after {rate_waits} waits: {reason}")
                if "hourly" in low:  # wait for the top of the next hour
                    now = dt.datetime.now(dt.timezone.utc)
                    until_next_hour = 3600 - (now.minute * 60 + now.second)
                    wait = max(retry_after, min(3700.0, until_next_hour + 15.0))
                elif "minut" in low:
                    wait = max(retry_after, 65.0)
                else:
                    wait = self._backoff(rate_waits, floor=retry_after)
                self.log(f"  429 rate limit ({reason}); sleeping {fmt_duration(wait)}")
                _sleep(wait)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 408:
                raise FatalApiError(f"HTTP {resp.status_code}: {reason}")
            if attempt >= self.max_retries:
                raise FatalApiError(f"HTTP {resp.status_code} after {attempt} attempts: {reason}")
            wait = self._backoff(attempt, floor=0.0)
            self.log(f"  HTTP {resp.status_code} ({reason[:80]}); retry {attempt}/{self.max_retries} in {wait:.0f}s")
            _sleep(wait)


# --------------------------------------------------------------------------- #
# Hourly -> daily aggregation
# --------------------------------------------------------------------------- #
def aggregate_daily(payload: dict, start: str, end: str, rain_threshold: float = RAIN_HOUR_THRESHOLD_MM) -> pd.DataFrame:
    """Aggregate one chunk of hourly data to city-days. A day's value is NaN unless
    all 24 hourly values of the source variable are present."""
    h = payload["hourly"]
    times = pd.to_datetime(h["time"], format="%Y-%m-%dT%H:%M")
    df = pd.DataFrame({v: np.array(h[v], dtype="float64") for v in HOURLY_VARS})
    df["heat_index"] = heat_index_celsius(df["temperature_2m"].to_numpy(), df["relative_humidity_2m"].to_numpy())
    date = pd.Series(times.floor("D"), name="date")
    g = df.groupby(date.to_numpy(), sort=True)
    cnt = g[HOURLY_VARS + ["heat_index"]].count()
    full = lambda var: cnt[var] == HOURS_PER_DAY  # noqa: E731

    out = pd.DataFrame(index=cnt.index)
    for col, var, fn in AGG_SPEC:
        out[col] = getattr(g[var], fn)().where(full(var))
    out["RainHours"] = (df["rain"] >= rain_threshold).groupby(date.to_numpy()).sum().where(full("rain"))
    out["HeatIndex_Max_Today"] = g["heat_index"].max().where(full("heat_index"))
    out["_HoursMin"] = cnt[HOURLY_VARS].min(axis=1).astype(int)
    out["_PrecipTotal"] = g["precipitation"].sum().where(full("precipitation"))
    for col in out.columns:
        if col == "_HoursMin":
            continue
        out[col] = out[col].round(ROUND_DP.get(col, 2))
    out.insert(0, "Date", pd.DatetimeIndex(out.index).strftime("%Y-%m-%d"))
    out = out.reset_index(drop=True)

    expected = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    if out["Date"].tolist() != expected or (cnt[HOURLY_VARS[0]].index.size != len(expected)):
        raise ApiDataError("daily index does not match requested range")
    out["_GridLat"] = float(payload.get("latitude", np.nan))
    out["_GridLon"] = float(payload.get("longitude", np.nan))
    out["_ApiElevation"] = float(payload.get("elevation", np.nan))
    return out[CHUNK_COLUMNS]


# --------------------------------------------------------------------------- #
# Chunk planning + cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    city_id: str
    city: str
    lat: float
    lon: float
    start: str
    end: str

    @property
    def n_days(self) -> int:
        return (dt.date.fromisoformat(self.end) - dt.date.fromisoformat(self.start)).days + 1


def cache_key(model: str) -> str:
    spec = {"model": model, "vars": HOURLY_VARS, "tz": TIMEZONE, "cells": CELL_SELECTION,
            "rain_thr": RAIN_HOUR_THRESHOLD_MM, "agg": AGG_VERSION}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def raw_key(model: str) -> str:
    """Raw API responses depend only on what was requested, not on how we aggregate them."""
    spec = {"model": model, "vars": HOURLY_VARS, "tz": TIMEZONE, "cells": CELL_SELECTION}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def plan_chunks(cities: pd.DataFrame, start: str, end: str) -> list[Chunk]:
    """One chunk per city per calendar year (clipped to [start, end])."""
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    pieces = []
    for year in range(s.year, e.year + 1):
        a, b = max(s, dt.date(year, 1, 1)), min(e, dt.date(year, 12, 31))
        pieces.append((a.isoformat(), b.isoformat()))
    return [Chunk(r.city_id, r.City, r.latitude, r.longitude, a, b)
            for r in cities.itertuples(index=False) for a, b in pieces]


def chunk_path(work_dir: Path, key: str, ch: Chunk) -> Path:
    return work_dir / "chunks" / key / f"{ch.city_id}_{ch.start}_{ch.end}.csv"


def raw_path(work_dir: Path, rkey: str, ch: Chunk) -> Path:
    return work_dir / "raw" / rkey / f"{ch.city_id}_{ch.start}_{ch.end}.json.gz"


def save_raw(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)


def load_raw(path: Path, ch: Chunk) -> dict | None:
    """Return a validated retained API response, or None (and remove the file) if unusable."""
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        OpenMeteoClient._validate(payload, ch.start, ch.end)
        return payload
    except (OSError, EOFError, ValueError, ApiDataError, FatalApiError):
        path.unlink(missing_ok=True)
        return None


def chunk_is_complete(path: Path, ch: Chunk) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, usecols=["Date"])
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return False
    return len(df) == ch.n_days and df["Date"].iloc[0] == ch.start and df["Date"].iloc[-1] == ch.end


class _NullBar:
    def update(self, *a, **k): ...
    def set_postfix_str(self, *a, **k): ...
    def close(self): ...


def collect_chunks(chunks: list[Chunk], client: OpenMeteoClient, work_dir: Path, key: str, log: Logger,
                   keep_raw: bool = True) -> tuple[str, int]:
    """Make every chunk's daily file exist: reuse the cache, rebuild from retained raw
    responses (no API call), otherwise fetch. Returns (status, n_fetched) with status in
    {'complete', 'daily_limit'}."""
    rkey = raw_key(client.model)
    pending = []
    for ch in chunks:
        p = chunk_path(work_dir, key, ch)
        if chunk_is_complete(p, ch):
            continue
        p.unlink(missing_ok=True)  # discard corrupt / partial file
        pending.append(ch)
    n_cached = len(chunks) - len(pending)

    rebuilt, to_fetch = 0, []
    for ch in pending:  # free path first: re-aggregate retained raw responses
        payload = load_raw(raw_path(work_dir, rkey, ch), ch)
        if payload is None:
            to_fetch.append(ch)
            continue
        atomic_write_csv(aggregate_daily(payload, ch.start, ch.end), chunk_path(work_dir, key, ch))
        rebuilt += 1
    est_weight = sum(call_weight(c.n_days) for c in to_fetch)
    day_budget = client.budget.limits.get("day")
    log(f"Chunks: {len(chunks)} total | {n_cached} cached | {rebuilt} rebuilt from retained raw data | "
        f"{len(to_fetch)} to fetch (~{est_weight:,.0f} weighted API calls)")
    if day_budget and to_fetch:
        log(f"  client daily budget {day_budget:,.0f} calls -> about {math.ceil(est_weight / day_budget)} day(s) of runs on the free tier")
    if not to_fetch:
        return "complete", 0

    bar = tqdm(total=len(to_fetch), unit="chunk", desc="Fetching", dynamic_ncols=True, mininterval=1.0) if tqdm else _NullBar()
    t0, fetched = time.time(), 0
    try:
        for ch in to_fetch:
            bar.set_postfix_str(f"{ch.city[:22]} {ch.start[:4]}")
            try:
                payload = client.fetch_hourly(ch.lat, ch.lon, ch.start, ch.end)
            except DailyLimitReached as exc:
                bar.close()
                log(f"\nSTOPPED: {exc}.")
                if exc.wait_seconds:
                    log(f"  Budget frees up in about {fmt_duration(exc.wait_seconds)}.")
                log(f"  {fetched} chunk(s) fetched this run; {len(to_fetch) - fetched} remaining. "
                    "Everything fetched is cached - re-run the same command later to resume.")
                return "daily_limit", fetched
            if keep_raw:  # written first: a crash before aggregation loses nothing
                save_raw(raw_path(work_dir, rkey, ch), payload)
            atomic_write_csv(aggregate_daily(payload, ch.start, ch.end), chunk_path(work_dir, key, ch))
            fetched += 1
            bar.update(1)
            if tqdm is None and fetched % 10 == 0:  # pragma: no cover
                rate = (time.time() - t0) / fetched
                log(f"  {fetched}/{len(to_fetch)} chunks, ETA {fmt_duration(rate * (len(to_fetch) - fetched))}")
    finally:
        bar.close()
    return "complete", fetched


def preflight(client: OpenMeteoClient, city: pd.Series, start: str, log: Logger) -> None:
    """One cheap request (3 days, weighted cost 1) to verify the model returns every variable."""
    end = (dt.date.fromisoformat(start) + dt.timedelta(days=2)).isoformat()
    log(f"Preflight: {city['City']} {start}..{end}, model={client.model}")
    data = client.fetch_hourly(float(city["latitude"]), float(city["longitude"]), start, end)
    empty = [v for v in HOURLY_VARS if all(x is None or (isinstance(x, float) and math.isnan(x)) for x in data["hourly"][v])]
    if empty:
        raise FatalApiError(
            f"model {client.model!r} returned no data for: {', '.join(empty)}. "
            "The documentation lists per-model variable availability; pick another model with --model "
            "(for example era5_seamless) or drop the variable.")
    glat, glon = data.get("latitude"), data.get("longitude")
    if glat is None or abs(glat - float(city["latitude"])) > 0.5 or abs(glon - float(city["longitude"])) > 0.5:
        raise FatalApiError(f"API grid cell ({glat}, {glon}) is far from the requested location")
    odd = {v: data.get("hourly_units", {}).get(v) for v in HOURLY_VARS
           if data.get("hourly_units", {}).get(v) != EXPECTED_UNITS[v]}
    if odd:
        log(f"  note: unit strings differ from expected (informational): {odd}")
    log(f"  OK: all {len(HOURLY_VARS)} variables present; grid cell ({glat}, {glon}), elevation {data.get('elevation')} m")


# --------------------------------------------------------------------------- #
# Assemble + targets
# --------------------------------------------------------------------------- #
def load_city_daily(work_dir: Path, key: str, chunks: list[Chunk]) -> pd.DataFrame:
    frames = [pd.read_csv(chunk_path(work_dir, key, ch)) for ch in sorted(chunks, key=lambda c: c.start)]
    return pd.concat(frames, ignore_index=True)


def build_targets(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create next-day targets within each city and drop each city's last row.

    The shift is calendar-aware: tomorrow's value is only attached when the next row
    is exactly Date + 1 day, so a gap can never pair a row with a wrong day.
    """
    df = daily.sort_values(["City", "Date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("City", sort=False)
    next_date = g["Date"].shift(-1)
    next_hi = g["HeatIndex_Max_Today"].shift(-1)
    is_last = next_date.isna()
    contiguous = (next_date - df["Date"]) == pd.Timedelta(days=1)
    df["HeatIndex_Max_Tomorrow"] = next_hi.where(contiguous)
    info = {"rows_before": len(df), "last_rows_dropped": int(is_last.sum()), "n_cities": int(df["City"].nunique())}
    df = df.loc[~is_last].reset_index(drop=True)
    df["HeatLevelToday"] = classify_heat_level(df["HeatIndex_Max_Today"])
    df["HeatLevelTomorrow"] = classify_heat_level(df["HeatIndex_Max_Tomorrow"])
    return df, info


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# column -> (hard_min, hard_max) ; outside = impossible (error)
HARD_RANGES = {
    "Latitude": (4.0, 22.0), "Longitude": (116.0, 128.0), "Elevation": (-100.0, 3500.0),
    "Month": (1, 12), "DayOfYear": (1, 366),
    "Temp_Min": (0.0, 50.0), "Temp_Mean": (0.0, 50.0), "Temp_Max": (0.0, 50.0),
    "Humidity_Min": (0.0, 100.0), "Humidity_Mean": (0.0, 100.0), "Humidity_Max": (0.0, 100.0),
    "DewPoint_Mean": (-5.0, 40.0), "Pressure_Mean": (870.0, 1090.0), "CloudCover_Mean": (0.0, 100.0),
    "WindSpeed_Mean": (0.0, 400.0), "WindSpeed_Max": (0.0, 400.0), "WindGust_Max": (0.0, 500.0),
    "Rainfall_Total": (0.0, 1200.0), "RainHours": (0, 24), "SolarRadiation_Total": (0.0, 12000.0),
    "HeatIndex_Max_Today": (10.0, 80.0), "HeatIndex_Max_Tomorrow": (10.0, 80.0),
}
# softer plausibility bands: outside = warning only
SOFT_RANGES = {
    "Temp_Max": (15.0, 42.5), "Pressure_Mean": (950.0, 1040.0), "Rainfall_Total": (0.0, 500.0),
    "HeatIndex_Max_Today": (15.0, 65.0), "HeatIndex_Max_Tomorrow": (15.0, 65.0), "Elevation": (0.0, 3000.0),
    "SolarRadiation_Total": (0.0, 10000.0),
}


def validate_dataset(df: pd.DataFrame, start: str, end: str, aux: pd.DataFrame | None = None) -> dict:
    """Run all data-quality checks on a final (sorted) dataset.

    `aux` optionally carries helper columns from the build (e.g. _PrecipTotal)."""
    errors, warnings, checks = [], [], {}
    n_cities = df["City"].nunique()
    expected_days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days  # last day dropped
    checks["shape"] = df.shape
    checks["date_min"], checks["date_max"] = str(df["Date"].min().date()), str(df["Date"].max().date())
    checks["n_cities"] = int(n_cities)

    # ---- columns / leakage guard
    if list(df.columns) != FINAL_COLUMNS:
        errors.append(f"columns differ from specification: {list(df.columns)}")
    future_cols = [c for c in df.columns if "tomorrow" in c.lower()]
    if sorted(future_cols) != ["HeatIndex_Max_Tomorrow", "HeatLevelTomorrow"]:
        errors.append(f"unexpected next-day columns: {future_cols}")

    # ---- missing
    miss = df.isna().sum()
    checks["missing_total"] = int(miss.sum())
    checks["missing_by_column"] = {k: int(v) for k, v in miss.items() if v}
    if miss.sum():
        errors.append(f"{int(miss.sum())} missing values: {checks['missing_by_column']}")

    # ---- duplicates
    dup_key = int(df.duplicated(["City", "Date"]).sum())
    dup_row = int(df.duplicated().sum())
    checks["duplicate_city_date"], checks["duplicate_rows"] = dup_key, dup_row
    if dup_key or dup_row:
        errors.append(f"duplicates: {dup_key} City/Date, {dup_row} full rows")

    # ---- sorting
    is_sorted = df[["City", "Date"]].equals(df[["City", "Date"]].sort_values(["City", "Date"], kind="mergesort").reset_index(drop=True))
    checks["sorted_by_city_date"] = bool(is_sorted)
    if not is_sorted:
        errors.append("rows are not sorted by City, Date")

    # ---- impossible / implausible values
    impossible, implausible = {}, {}
    for col, (lo, hi) in HARD_RANGES.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            impossible[col] = n
    for col, (lo, hi) in SOFT_RANGES.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n and col not in impossible:
            implausible[col] = n
    order_rules = {
        "Temp_Min<=Temp_Mean<=Temp_Max": ~((df["Temp_Min"] <= df["Temp_Mean"]) & (df["Temp_Mean"] <= df["Temp_Max"])),
        "Humidity_Min<=Mean<=Max": ~((df["Humidity_Min"] <= df["Humidity_Mean"]) & (df["Humidity_Mean"] <= df["Humidity_Max"])),
        "WindSpeed_Mean<=WindSpeed_Max": df["WindSpeed_Mean"] > df["WindSpeed_Max"],
        "DewPoint_Mean<=Temp_Mean+0.1": df["DewPoint_Mean"] > df["Temp_Mean"] + 0.1,
        "Month==Date.month": df["Month"] != df["Date"].dt.month,
        "DayOfYear==Date.dayofyear": df["DayOfYear"] != df["Date"].dt.dayofyear,
        "RainHours integer": df["RainHours"] != df["RainHours"].round(),
        # any rain hour (>= threshold) forces a daily total >= threshold; the reverse need not hold (drizzle)
        "RainHours>0 implies Rainfall_Total>=threshold": (df["RainHours"] > 0) & (df["Rainfall_Total"] < RAIN_HOUR_THRESHOLD_MM - 1e-9),
    }
    for name, bad in order_rules.items():
        if int(bad.sum()):
            impossible[name] = int(bad.sum())
    gust_lt_wind = int((df["WindGust_Max"] < df["WindSpeed_Max"] - 0.01).sum())
    if gust_lt_wind:
        implausible["WindGust_Max<WindSpeed_Max"] = gust_lt_wind
    if aux is not None and "_PrecipTotal" in aux:
        gap = (aux["_PrecipTotal"] - aux["Rainfall_Total"]).abs()
        n = int((gap > 0.5).sum())
        checks["max_abs_precipitation_minus_rain_mm"] = round(float(gap.max()), 3)
        if n:
            implausible["|precipitation-rain|>0.5mm"] = n
    checks["impossible_values"], checks["implausible_values"] = impossible, implausible
    if impossible:
        errors.append(f"impossible values: {impossible}")
    if implausible:
        warnings.append(f"unusual (but possible) values: {implausible}")

    # ---- city/date gaps
    gaps = {}
    per_city = df.groupby("City", sort=False)["Date"]
    bad_len = per_city.size()[per_city.size() != expected_days]
    bad_first = per_city.min()[per_city.min() != pd.Timestamp(start)]
    bad_last = per_city.max()[per_city.max() != pd.Timestamp(end) - pd.Timedelta(days=1)]
    step = df.groupby("City", sort=False)["Date"].diff().dropna()
    n_breaks = int((step != pd.Timedelta(days=1)).sum())
    if len(bad_len):
        gaps["cities_with_wrong_row_count"] = bad_len.to_dict()
    if len(bad_first):
        gaps["cities_with_wrong_first_date"] = {k: str(v.date()) for k, v in bad_first.items()}
    if len(bad_last):
        gaps["cities_with_wrong_last_date"] = {k: str(v.date()) for k, v in bad_last.items()}
    if n_breaks:
        gaps["non_consecutive_steps"] = n_breaks
    checks["expected_rows_per_city"] = expected_days
    checks["gaps"] = gaps
    if gaps:
        errors.append(f"city/date gaps: {gaps}")

    # ---- target integrity: tomorrow == following day's today, same city (exact)
    nxt = df.groupby("City", sort=False)["HeatIndex_Max_Today"].shift(-1)
    nxt_date = df.groupby("City", sort=False)["Date"].shift(-1)
    has_next = nxt_date.notna() & ((nxt_date - df["Date"]) == pd.Timedelta(days=1))
    mismatch_shift = int((df.loc[has_next, "HeatIndex_Max_Tomorrow"] != nxt[has_next]).sum())
    # independent method: self-merge on (City, Date + 1 day)
    right = df[["City", "Date", "HeatIndex_Max_Today"]].copy()
    right["Date"] = right["Date"] - pd.Timedelta(days=1)
    merged = df[["City", "Date", "HeatIndex_Max_Tomorrow"]].merge(right, on=["City", "Date"], how="left", suffixes=("", "_next"))
    has_m = merged["HeatIndex_Max_Today"].notna()
    mismatch_merge = int((merged.loc[has_m, "HeatIndex_Max_Tomorrow"] != merged.loc[has_m, "HeatIndex_Max_Today"]).sum())
    checks["tomorrow_equals_next_day_today"] = {
        "rows_checked_shift": int(has_next.sum()), "mismatches_shift": mismatch_shift,
        "rows_checked_merge": int(has_m.sum()), "mismatches_merge": mismatch_merge,
        "rows_without_following_row_in_dataset": int((~has_m).sum()),
    }
    if mismatch_shift or mismatch_merge:
        errors.append(f"HeatIndex_Max_Tomorrow != next-day HeatIndex_Max_Today in {max(mismatch_shift, mismatch_merge)} rows")
    # the very last day of each city (only its tomorrow target) is the one allowed absentee
    if int((~has_m).sum()) != n_cities:
        warnings.append(f"{int((~has_m).sum())} rows have no following-day row in the dataset (expected {n_cities}: each city's last kept day)")

    # ---- label integrity
    for lvl, num in (("HeatLevelToday", "HeatIndex_Max_Today"), ("HeatLevelTomorrow", "HeatIndex_Max_Tomorrow")):
        rederived = classify_heat_level(df[num].to_numpy()).fillna("<NA>").to_numpy()
        bad = int((rederived != df[lvl].astype(object).fillna("<NA>").to_numpy()).sum())
        checks[f"{lvl}_mismatches"] = bad
        if bad:
            errors.append(f"{lvl} disagrees with {num} in {bad} rows")
        unknown = set(df[lvl].dropna().unique()) - set(HEAT_LEVELS)
        if unknown:
            errors.append(f"{lvl} has unknown categories {unknown}")

    # ---- class distribution
    dist = {}
    for lvl in ("HeatLevelTomorrow", "HeatLevelToday"):
        vc = df[lvl].value_counts().reindex(HEAT_LEVELS, fill_value=0)
        dist[lvl] = pd.DataFrame({"count": vc, "percent": (vc / vc.sum() * 100).round(2)})
    checks["class_distribution"] = dist
    return {"errors": errors, "warnings": warnings, "checks": checks}


def print_validation(report: dict, df: pd.DataFrame, out=print) -> None:
    c = report["checks"]
    out("\n" + "=" * 78 + "\nDATASET SUMMARY\n" + "=" * 78)
    out(f"Shape        : {c['shape'][0]:,} rows x {c['shape'][1]} columns")
    out(f"Date range   : {c['date_min']} .. {c['date_max']} (last day per city dropped: tomorrow unavailable)")
    out(f"Cities       : {c['n_cities']}   (expected {c['expected_rows_per_city']:,} rows per city)")
    out(f"Missing      : {c['missing_total']}   Duplicates (City+Date): {c['duplicate_city_date']}   "
        f"Gaps: {'none' if not c['gaps'] else c['gaps']}")
    out(f"Impossible   : {c['impossible_values'] or 'none'}")
    out(f"Implausible  : {c['implausible_values'] or 'none'}")
    t = c["tomorrow_equals_next_day_today"]
    out(f"Target check : HeatIndex_Max_Tomorrow == next day's HeatIndex_Max_Today (same city): "
        f"{t['rows_checked_shift']:,} rows via shift, {t['rows_checked_merge']:,} via merge -> "
        f"{t['mismatches_shift']} / {t['mismatches_merge']} mismatches")
    for lvl, title in (("HeatLevelTomorrow", "TARGET class distribution - HeatLevelTomorrow"),
                       ("HeatLevelToday", "Reference - HeatLevelToday")):
        out(f"\n{title}")
        d = c["class_distribution"][lvl]
        for name, row in d.iterrows():
            out(f"  {name:<16}{int(row['count']):>10,}  {row['percent']:>6.2f}%")
        out(f"  {'TOTAL':<16}{int(d['count'].sum()):>10,}  100.00%")
    out("\nSummary statistics (numeric columns)")
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:,.2f}".format):
        out(df.select_dtypes("number").describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())


# --------------------------------------------------------------------------- #
# Data dictionary + documentation
# --------------------------------------------------------------------------- #
_DICT = [
    ("Date", "identifier", "YYYY-MM-DD", "Local calendar date (Asia/Manila).", "Date of the hourly timestamps aggregated."),
    ("City", "identifier", "", "Unique city key (PSGC city name without 'City of'/'City'; 'Name (Province)' where the name is shared).", "Philippine Standard Geographic Code city list; see ph_heat_index_city_list.csv."),
    ("Latitude", "static feature", "degrees N", "Latitude requested from the API (population-weighted city center).", "Population-weighted mean of barangay centroids."),
    ("Longitude", "static feature", "degrees E", "Longitude requested from the API (population-weighted city center).", "Population-weighted mean of barangay centroids."),
    ("Elevation", "static feature", "m", "Elevation returned by Open-Meteo for the requested coordinates (90 m DEM used for its statistical downscaling).", "Response field `elevation`."),
    ("Month", "calendar feature", "1-12", "Calendar month of Date.", "Date.month"),
    ("DayOfYear", "calendar feature", "1-366", "Day of year of Date.", "Date.dayofyear"),
    ("Temp_Min", "feature (same day)", "°C", "Minimum hourly 2 m air temperature.", "min of hourly temperature_2m"),
    ("Temp_Mean", "feature (same day)", "°C", "Mean hourly 2 m air temperature.", "mean of hourly temperature_2m"),
    ("Temp_Max", "feature (same day)", "°C", "Maximum hourly 2 m air temperature.", "max of hourly temperature_2m"),
    ("Humidity_Min", "feature (same day)", "%", "Minimum hourly relative humidity at 2 m.", "min of hourly relative_humidity_2m"),
    ("Humidity_Mean", "feature (same day)", "%", "Mean hourly relative humidity at 2 m.", "mean of hourly relative_humidity_2m"),
    ("Humidity_Max", "feature (same day)", "%", "Maximum hourly relative humidity at 2 m.", "max of hourly relative_humidity_2m"),
    ("DewPoint_Mean", "feature (same day)", "°C", "Mean hourly 2 m dew point.", "mean of hourly dew_point_2m"),
    ("Pressure_Mean", "feature (same day)", "hPa", "Mean hourly mean-sea-level pressure.", "mean of hourly pressure_msl"),
    ("CloudCover_Mean", "feature (same day)", "%", "Mean hourly total cloud cover.", "mean of hourly cloud_cover"),
    ("WindSpeed_Mean", "feature (same day)", "km/h", "Mean hourly 10 m wind speed.", "mean of hourly wind_speed_10m"),
    ("WindSpeed_Max", "feature (same day)", "km/h", "Maximum hourly 10 m wind speed.", "max of hourly wind_speed_10m"),
    ("WindGust_Max", "feature (same day)", "km/h", "Maximum hourly 10 m wind gust.", "max of hourly wind_gusts_10m"),
    ("Rainfall_Total", "feature (same day)", "mm", "Total rain over the day.", "sum of hourly rain"),
    ("RainHours", "feature (same day)", "hours (0-24)", f"Number of hours with rain >= {RAIN_HOUR_THRESHOLD_MM} mm.", f"count of hourly rain >= {RAIN_HOUR_THRESHOLD_MM}"),
    ("SolarRadiation_Total", "feature (same day)", "Wh/m²", "Daily total shortwave (global horizontal) radiation. Divide by 277.78 for MJ/m².", "sum of hourly shortwave_radiation (W/m², preceding-hour mean => Wh/m² per hour)"),
    ("HeatIndex_Max_Today", "feature (same day, derived)", "°C", "Maximum of the 24 hourly NWS heat-index values of Date.", "max over hours of heat_index(temperature_2m, relative_humidity_2m)"),
    ("HeatLevelToday", "feature (same day, derived)", "category", "Heat level of HeatIndex_Max_Today.", "Category rule in dataset_documentation.md"),
    ("HeatIndex_Max_Tomorrow", "TARGET (regression)", "°C", "HeatIndex_Max_Today of the next calendar day in the same city.", "shift(-1) within City, only when the next row is Date + 1 day"),
    ("HeatLevelTomorrow", "TARGET (classification)", "category", "Heat level of HeatIndex_Max_Tomorrow. Deterministic bins of the numeric target, not an independent label.", "Category rule in dataset_documentation.md"),
]


def _dtype_name(series: pd.Series) -> str:
    """pandas-version-independent type names (object/str -> string)."""
    d = str(series.dtype)
    if d.startswith("datetime64"):
        return "date"
    return "string" if d in ("object", "str", "string") else d


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, role, unit, desc, deriv in _DICT:
        rows.append({"column": col, "dtype": _dtype_name(df[col]) if col in df else "", "role": role, "unit": unit,
                     "description": desc, "derivation": deriv,
                     "allowed_values": " | ".join(HEAT_LEVELS) if col.startswith("HeatLevel") else ""})
    if [r["column"] for r in rows] != FINAL_COLUMNS:
        raise RuntimeError("data dictionary and FINAL_COLUMNS are out of sync")
    return pd.DataFrame(rows)


def write_documentation(path: Path, ctx: dict) -> None:
    c = ctx["report"]["checks"]
    dist = c["class_distribution"]["HeatLevelTomorrow"]
    dist_tbl = "\n".join(f"| {k} | {int(r['count']):,} | {r['percent']:.2f}% |" for k, r in dist.iterrows())
    cities = ctx["cities"]
    n_cells = cities[["grid_latitude", "grid_longitude"]].drop_duplicates().shape[0]
    cls_counts = cities["city_class"].value_counts().to_dict()
    warn = "\n".join(f"- {w}" for w in ctx["report"]["warnings"]) or "- none"
    subset_note = ""
    selection_note = ", i.e. the full official city list of that release."
    if ctx["is_subset"]:
        selection_note = f"; a subset of the {ctx['n_cities_total']} cities in that release."
    if ctx["is_subset"]:
        subset_note = ("\n> **NOTE: this is a SUBSET build** (`--max-cities` / `--allow-partial`): "
                       f"{c['n_cities']} of {ctx['n_cities_total']} cities.\n")
    md = f"""# Philippine Next-Day Heat-Index Dataset

Generated {ctx['generated_at']} by `build_ph_heat_index_dataset.py` v{SCRIPT_VERSION}
(Python {ctx['py']}, pandas {pd.__version__}, numpy {np.__version__}).
{subset_note}
## What this is

One row per city-day for **{c['n_cities']} Philippine cities**, {c['date_min']} to {c['date_max']}
({c['shape'][0]:,} rows x {c['shape'][1]} columns). Inputs are same-day weather aggregates; the
targets are the **next day's** maximum heat index (numeric) and its heat level (class).
Files: `ph_heat_index_next_day.csv`, `ph_heat_index_data_dictionary.csv`, `ph_heat_index_city_list.csv`.

## Data source

- **Open-Meteo Historical Weather API** (<https://open-meteo.com/en/docs/historical-weather-api>), reanalysis data.
- **Endpoint:** `{ctx['api_url']}`
- **Model pinned to `{ctx['model']}`** (ERA5, 0.25 degree, about 25 km). The API default `best_match` blends ECMWF IFS
  (9 km, 2017 onward) with ERA5; that would change the underlying model inside the 2015-2025 window, so it is not used.
  `cell_selection={CELL_SELECTION}` (land grid cell with similar elevation to the request).
- **Timezone:** `{TIMEZONE}` (UTC+8, no daylight saving). All timestamps are local; days are local calendar days.
- **Period requested:** {ctx['start']} to {ctx['end']} (inclusive); data retrieved between {ctx['fetched_between']} (cache-file timestamps, approximate).
- **Units requested:** temperature °C, wind km/h, precipitation mm (API explicit parameters).
- **Hourly variables requested:** {', '.join(f'`{v}`' for v in HOURLY_VARS)}.
- **Attribution:** weather data by Open-Meteo.com (CC BY 4.0); contains modified Copernicus Climate Change Service
  (ERA5) information. The free API is for non-commercial use.

> **ERA5 is gridded reanalysis data, not PAGASA station observations.** Each value is a model estimate for a
> ~25 km grid cell, adjusted by Open-Meteo for the elevation difference between the cell and the requested point
> (90 m elevation model; see the Open-Meteo documentation). Heat-index values here are
> **not** PAGASA-reported station heat indices and can differ from them, especially in coastal or mountain cities.

## City list

- **Source:** Philippine Standard Geographic Code (PSGC), PSA Q1 2026 release, as packaged by the community
  `psgc` Python package **v{PSGC_VERSION}** from PyPI (MIT licence; not affiliated with the PSA). The wheel is downloaded
  from PyPI and verified against SHA-256 `{PSGC_WHEEL_SHA256}`; it is read as data, never imported.
- **Selection:** PSGC records with geographic level *City* - **{len(cities)} cities**
  ({', '.join(f'{v} {k}' for k, v in cls_counts.items())}){selection_note}
  Municipalities are not included.
- **Coordinates:** population-weighted mean of the city's barangay centroids (2024 census population; barangay
  centroids from OCHA/HDX 2023 administrative boundaries via `psgc`, CC BY-IGO). Barangays flagged
  `fallback_unverified` are excluded. This approximates the populated area; the package's own area-weighted
  city centroid can lie far from the urban core (up to {ctx['max_shift_km']:.0f} km here, see `shift_vs_area_centroid_km`).
- **Elevation:** taken from the Open-Meteo response for the requested coordinates (the PSGC data has no elevation).
- **Grid cells:** the {len(cities)} cities map to **{n_cells} distinct ERA5 grid cells** (`grid_latitude`/`grid_longitude`
  in the city list). Cities sharing a cell receive nearly identical weather; only the elevation downscaling differs.
- **City keys:** PSGC name without "City of"/"City"; four names occur twice (Naga, San Carlos, San Fernando, Talisay)
  and are written `Name (Province)`. Official names and PSGC codes are in the city list.

## Collection method

1. Build the city list (above).
2. For each city and calendar year, request the 10 hourly variables (one request per city-year; whole-year
   requests cost ~26 weighted API calls each; ~{ctx['total_weight']:,.0f} weighted calls for the cities in this build).
3. Validate each response (timestamp count/range, all variables present, timezone Asia/Manila, Celsius).
4. Aggregate to daily values immediately and cache the result per city-year (atomic file writes), so re-runs resume
   instead of starting over. The raw hourly responses are also retained (gzip, `raw/` in the work directory) so that
   any change to the aggregation can be re-applied without calling the API again (`--no-keep-raw` disables this).
5. Retries with exponential backoff on network/5xx errors; `429` handling distinguishes minutely (wait ~65 s),
   hourly (wait for the next hour) and daily limits (stop gracefully, resume later). A persisted client-side budget
   (500/min, 4,500/h, 9,500/day weighted calls) keeps the free tier from being exceeded.
6. Assemble all cached chunks, create targets, validate, write outputs.

## Heat index

Computed **per hour** from `temperature_2m` and `relative_humidity_2m` with the NOAA/NWS algorithm
(Rothfusz 1990, NWS Southern Region Technical Attachment SR 90-23; NWS WPC, "The Heat Index Equation",
<https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml>). Open-Meteo's `apparent_temperature` is **not** used.
With T in °F and RH in %:

1. Simple (Steadman) estimate: `HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)`.
2. If `(HI + T) / 2 >= 80`, use the Rothfusz regression:
   `HI = -42.379 + 2.04901523 T + 10.14333127 RH - 0.22475541 T RH - 0.00683783 T^2 - 0.05481717 RH^2
   + 0.00122874 T^2 RH + 0.00085282 T RH^2 - 0.00000199 T^2 RH^2`, then
   - if RH < 13 % and 80 <= T <= 112: subtract `((13 - RH) / 4) * sqrt((17 - |T - 95|) / 17)`;
   - if RH > 85 % and 80 <= T <= 87: add `((RH - 85) / 10) * ((87 - T) / 5)`.
3. Otherwise keep the simple estimate. The result is converted to °C.

An hourly value is **valid** when temperature and humidity are present and physically possible
(-90 to 60 °C, 0 to 100 %). `HeatIndex_Max_Today` is the maximum of the day's valid hourly values; it is only produced
when all 24 hours are valid (ERA5 is gap-free, so this is the normal case). A day with any invalid hour is excluded
and reported rather than computed from fewer hours, which could silently miss the peak.
The implementation was cross-checked against MetPy's `heat_index` (`tests/test_pipeline.py`, needs `pip install metpy`):
the two are identical (to 1e-6 °C) except where air temperature is roughly 25-27 °C, because the NWS WPC text and
MetPy switch from the simple formula to the regression at slightly different points; there the difference reaches
about 1.25 °C (around the 27 °C class boundary). It only matters on cool days whose maximum temperature stays below
~27 °C (e.g. upland cities in the cool season).

## Aggregation method

Daily values are computed from the 24 local hourly values of each date. **A daily value is only produced when all 24
hourly values of its source variable are present** (otherwise it is missing and the row is reported/dropped).

| Column | Hourly source | Aggregation |
|---|---|---|
| Temp_Min / Mean / Max | temperature_2m | min / mean / max |
| Humidity_Min / Mean / Max | relative_humidity_2m | min / mean / max |
| DewPoint_Mean | dew_point_2m | mean |
| Pressure_Mean | pressure_msl | mean |
| CloudCover_Mean | cloud_cover | mean |
| WindSpeed_Mean / Max | wind_speed_10m | mean / max |
| WindGust_Max | wind_gusts_10m | max |
| Rainfall_Total | rain | sum (mm) |
| RainHours | rain | count of hours with rain >= {RAIN_HOUR_THRESHOLD_MM} mm |
| SolarRadiation_Total | shortwave_radiation | sum of hourly W/m² (= Wh/m² per day) |
| HeatIndex_Max_Today | hourly heat index | max |

`precipitation` is collected and used only as a consistency check against `rain` (max daily absolute difference
{c.get('max_abs_precipitation_minus_rain_mm', 'n/a')} mm); it is not an output column. Values are rounded to 2 decimals
(solar total 1) **before** targets are created, so the numeric targets are exactly equal to the next day's feature.

## Target creation

Within each city (sorted by Date): `HeatIndex_Max_Tomorrow` = `HeatIndex_Max_Today` shifted by -1 day, attached
only when the next row is exactly the next calendar day. The **last row of every city is dropped** because tomorrow
is unavailable ({ctx['info']['last_rows_dropped']} rows). Only today's data appear in the input columns; no tomorrow
weather feature exists. `HeatLevelToday` / `HeatLevelTomorrow` are the categories below applied to
`HeatIndex_Max_Today` / `HeatIndex_Max_Tomorrow`. Rows are sorted by City, then Date.

## Heat-level categories

Thresholds as specified for this dataset (they follow the PAGASA heat-index classification). They are written for
whole °C, which leaves gaps for fractional values (e.g. 32.5), so **the lower bound of each class is inclusive and
the class extends up to the next lower bound**, applied to the stored 2-decimal value:

| HeatLevel | Rule on heat index (°C) |
|---|---|
| Not Hazardous | < 27 |
| Caution | 27 <= HI < 33 |
| Extreme Caution | 33 <= HI < 42 |
| Danger | 42 <= HI < 52 |
| Extreme Danger | >= 52 |

**The classification labels are derived from the numeric heat index** (`HeatLevelTomorrow` is a deterministic binning
of `HeatIndex_Max_Tomorrow`, the quantity a regression model would predict) - they are not independent observations
and not PAGASA-issued advisories. A model that predicts the number and then bins it inherits any error near class
boundaries; the exact numeric target is therefore kept.

### Class distribution of `HeatLevelTomorrow` (this build)

| Class | Rows | Share |
|---|---|---|
{dist_tbl}

## Validation (this build)

- Rows: {c['shape'][0]:,}; cities: {c['n_cities']}; expected rows per city: {c['expected_rows_per_city']:,}.
- Missing values: {c['missing_total']}; duplicate (City, Date): {c['duplicate_city_date']}; city/date gaps: {'none' if not c['gaps'] else c['gaps']}.
- Impossible values: {c['impossible_values'] or 'none'}.
- `HeatIndex_Max_Tomorrow` equals the following day's `HeatIndex_Max_Today` (same city, exact equality) in all
  {c['tomorrow_equals_next_day_today']['rows_checked_shift']:,} checkable rows ({c['tomorrow_equals_next_day_today']['mismatches_shift']} mismatches).
- Warnings:
{warn}

## Known limitations

- **Reanalysis, not observations.** ~25 km grid cells; no station data. Reanalysis humidity and temperature can be
  biased versus PAGASA stations, so absolute heat-index levels may be off, particularly for coastal and mountain cities.
- **Hourly instantaneous values.** The daily heat-index maximum is the maximum of 24 hourly values; the true
  sub-hourly peak and PAGASA's station-based reporting practice may differ.
- **Heat-index formula range.** The NWS regression is fitted for shade, light wind, T >= 80 °F and RH >= 40 %; it is
  an approximation outside that range and does not account for direct sun or wind. It is not a physiological model.
- **Few independent locations.** {len(cities)} cities share {n_cells} grid cells, so cities are strongly correlated
  in space. Use time-based splits and group by grid cell/region; random row splits leak information between
  neighbouring cities and consecutive days.
- **Temporal dependence.** Consecutive days are strongly autocorrelated; use chronological validation.
- **City coordinates are model inputs, not city-hall points.** Population-weighted centers can fall outside
  built-up areas for dispersed or multi-centre cities, and `cell_selection=land` may move to a neighbouring cell.
- **City definition.** The list is the PSGC Q1 2026 city set, including cities converted from municipalities after
  2015; the weather series is physical and unaffected, but "city" status was not constant over 2015-2025.
- **Class balance is determined by climate**, not by design; see the distribution above. Rare classes
  (Extreme Danger, Not Hazardous) may have few rows.
- **Revisions.** ERA5 is occasionally corrected upstream (ERA5T to ERA5); a re-download later may differ slightly.
  The chunk cache pins the data actually used.
- **Rain definition.** `RainHours` uses a 0.1 mm/h threshold, a modelling choice.
- **Solar units.** `SolarRadiation_Total` is in Wh/m² (hourly values are preceding-hour means).
- **Free-tier terms.** The free Open-Meteo API is for non-commercial use and rate-limited.

## Reproduce

```bash
pip install requests pandas numpy tqdm
python build_ph_heat_index_dataset.py --work-dir ph_cache --out-dir out   # re-run until it finishes
```

The free tier needs several days of runs (budget {DEFAULT_BUDGETS['day']:,.0f} weighted calls/day); cached chunks are
reused. In Google Colab use `ph_heat_index_colab.ipynb` with Google Drive as the cache location.
"""
    atomic_write_text(path, md)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default="ph_heat_index_cache", help="cache directory (use Google Drive in Colab)")
    p.add_argument("--out-dir", default=".", help="where the four output files are written")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Open-Meteo reanalysis model (default era5)")
    p.add_argument("--api-url", default=None, help="override endpoint (testing)")
    p.add_argument("--api-key", default=os.environ.get("OPEN_METEO_API_KEY"), help="commercial key -> customer endpoint, no client budgets")
    p.add_argument("--max-cities", type=int, default=None, help="only the first N cities (smoke test); implies --allow-partial")
    p.add_argument("--allow-partial", action="store_true", help="write outputs from cities whose chunks are all complete")
    p.add_argument("--no-keep-raw", dest="keep_raw", action="store_false",
                   help="do not retain raw hourly API responses (default: keep them, gzip, so aggregation changes never need the API again)")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--minute-budget", type=float, default=DEFAULT_BUDGETS["minute"])
    p.add_argument("--hour-budget", type=float, default=DEFAULT_BUDGETS["hour"])
    p.add_argument("--day-budget", type=float, default=DEFAULT_BUDGETS["day"], help="0 = unlimited")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--backoff-base", type=float, default=5.0)
    p.add_argument("--validate-only", metavar="CSV", help="validate an existing dataset CSV and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.validate_only:  # read-only: no directories or logs are created
        log = Logger(None)
        df = pd.read_csv(args.validate_only, parse_dates=["Date"])
        report = validate_dataset(df, args.start, args.end)
        print_validation(report, df, log)
        for w in report["warnings"]:
            log(f"WARNING: {w}")
        for e in report["errors"]:
            log(f"ERROR: {e}")
        log("VALIDATION: " + ("FAILED" if report["errors"] else "PASSED"))
        return 1 if report["errors"] else 0

    work_dir, out_dir = Path(args.work_dir).resolve(), Path(args.out_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(work_dir / "run_log.txt")

    log(f"=== ph heat index build v{SCRIPT_VERSION} | {args.start}..{args.end} | model={args.model} ===")
    # ---- city list
    base_path = work_dir / "city_list_base.csv"
    if base_path.exists():
        cities = pd.read_csv(base_path, dtype={"city_id": str})
    else:
        cities = build_city_list(work_dir, log)
        atomic_write_csv(cities, base_path)
    n_total = len(cities)
    if args.max_cities:
        cities = cities.head(args.max_cities).reset_index(drop=True)
        args.allow_partial = True
    log(f"City list: {len(cities)} of {n_total} PSGC cities selected")

    key = cache_key(args.model)
    chunks = plan_chunks(cities, args.start, args.end)

    # ---- API client
    use_key = bool(args.api_key)
    limits = None if use_key else {"minute": args.minute_budget, "hour": args.hour_budget, "day": args.day_budget}
    # customer-key calls do not count against the free tier, so they are not recorded in the free-tier usage log
    budget = RateBudget(limits, usage_path=None if use_key else work_dir / "api_usage_log.json", log=log)
    api_url = args.api_url or (CUSTOMER_API_URL if use_key else API_URL)
    client = OpenMeteoClient(api_url, args.model, api_key=args.api_key, budget=budget,
                             max_retries=args.max_retries, backoff_base=args.backoff_base, log=log)
    log(f"Endpoint: {api_url}" + ("  (customer key, client budgets off)" if use_key else ""))

    try:
        rkey = raw_key(args.model)
        n_need_api = sum(not chunk_is_complete(chunk_path(work_dir, key, c), c) and not raw_path(work_dir, rkey, c).exists()
                         for c in chunks)
        if n_need_api and not args.skip_preflight:
            preflight(client, cities.iloc[0], args.start, log)
        status, _ = collect_chunks(chunks, client, work_dir, key, log, keep_raw=args.keep_raw)
    except FatalApiError as exc:
        log(f"FATAL: {exc}")
        return 1
    except KeyboardInterrupt:
        log("Interrupted. Completed chunks are cached; re-run to resume.")
        return 2
    except DailyLimitReached as exc:  # raised by preflight
        log(f"STOPPED before fetching: {exc}. Re-run later; nothing is lost.")
        return 2

    # ---- which cities are complete?
    by_city: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_city.setdefault(c.city_id, []).append(c)
    complete_ids = [cid for cid, cs in by_city.items()
                    if all(chunk_is_complete(chunk_path(work_dir, key, c), c) for c in cs)]
    if len(complete_ids) < len(cities) and not args.allow_partial:
        log(f"\nINCOMPLETE: {len(complete_ids)}/{len(cities)} cities fully cached; not writing final outputs "
            "(pass --allow-partial to build from the completed cities). Re-run to continue.")
        return 2
    if not complete_ids:
        log("No complete cities yet - nothing to assemble.")
        return 2
    cities = cities[cities["city_id"].isin(complete_ids)].reset_index(drop=True)
    is_subset = len(cities) < n_total

    # ---- assemble
    log("Assembling dataset from cached chunks...")
    frames, meta_rows = [], []
    for r in cities.itertuples(index=False):
        d = load_city_daily(work_dir, key, by_city[r.city_id])
        elev = d["_ApiElevation"].median()
        if d["_ApiElevation"].nunique() > 1:
            log(f"  note: {r.City} reported several elevations {sorted(d['_ApiElevation'].unique())}; using median")
        meta_rows.append({"city_id": r.city_id, "elevation_m": float(elev),
                          "grid_latitude": float(d["_GridLat"].iloc[0]), "grid_longitude": float(d["_GridLon"].iloc[0])})
        d.insert(1, "City", r.City)
        d["Latitude"], d["Longitude"], d["Elevation"] = r.latitude, r.longitude, float(elev)
        frames.append(d)
    daily = pd.concat(frames, ignore_index=True)
    del frames
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Month"], daily["DayOfYear"] = daily["Date"].dt.month, daily["Date"].dt.dayofyear

    # missing-value policy: incomplete days are dropped, loudly, with a safety cap
    inputs_ok = daily[DAILY_FEATURE_COLUMNS + ["HeatIndex_Max_Today"]].notna().all(axis=1)
    n_incomplete = int((~inputs_ok).sum())
    if n_incomplete:
        log(f"WARNING: {n_incomplete} city-days have incomplete hourly data (will be reported as gaps)")
        if n_incomplete / len(daily) > 0.005:
            log("FATAL: more than 0.5% of city-days are incomplete - refusing to continue; inspect the cache.")
            return 1
        daily = daily.loc[inputs_ok].reset_index(drop=True)

    final, info = build_targets(daily)
    aux = final[["_PrecipTotal"]].copy()
    final["RainHours"] = final["RainHours"].astype("int64")
    final = final[FINAL_COLUMNS]
    if info["last_rows_dropped"] != info["n_cities"]:
        log(f"ERROR: dropped {info['last_rows_dropped']} last rows for {info['n_cities']} cities")
        return 1

    # ---- write -> re-read -> validate the *saved* file
    ds_path = out_dir / "ph_heat_index_next_day.csv"
    tmp_path = ds_path.with_name(ds_path.name + ".tmp")
    final.to_csv(tmp_path, index=False, date_format="%Y-%m-%d")
    reread = pd.read_csv(tmp_path, parse_dates=["Date"])
    aux_re = aux.reset_index(drop=True)
    report = validate_dataset(reread, args.start, args.end, aux=pd.concat([aux_re, reread[["Rainfall_Total"]]], axis=1))
    print_validation(report, reread, log)
    for w in report["warnings"]:
        log(f"WARNING: {w}")
    if report["errors"]:
        bad = ds_path.with_name("ph_heat_index_next_day.FAILED_VALIDATION.csv")
        os.replace(tmp_path, bad)
        for e in report["errors"]:
            log(f"ERROR: {e}")
        log(f"Validation FAILED. Dataset kept for inspection at {bad}; final file not written.")
        return 1
    os.replace(tmp_path, ds_path)

    # ---- companion files
    city_out = cities.merge(pd.DataFrame(meta_rows), on="city_id", how="left")
    city_out.rename(columns={"latitude": "latitude", "longitude": "longitude"}, inplace=True)
    city_path = out_dir / "ph_heat_index_city_list.csv"
    atomic_write_csv(city_out, city_path)
    dict_path = out_dir / "ph_heat_index_data_dictionary.csv"
    atomic_write_csv(build_data_dictionary(reread), dict_path)
    chunk_files = [chunk_path(work_dir, key, c) for c in chunks if c.city_id in set(cities["city_id"])]
    mtimes = [dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) for p in chunk_files if p.exists()]
    ctx = {
        "report": report, "cities": city_out, "info": info, "is_subset": is_subset, "n_cities_total": n_total,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "py": sys.version.split()[0], "api_url": API_URL, "model": args.model, "start": args.start, "end": args.end,
        "fetched_between": f"{min(mtimes):%Y-%m-%d} and {max(mtimes):%Y-%m-%d}" if mtimes else "n/a",
        "max_shift_km": float(city_out["shift_vs_area_centroid_km"].max()),
        "total_weight": sum(call_weight(c.n_days) for c in chunks),
    }
    doc_path = out_dir / "dataset_documentation.md"
    write_documentation(doc_path, ctx)

    log("\n" + "=" * 78 + "\nOUTPUT FILES\n" + "=" * 78)
    for p in (ds_path, dict_path, city_path, doc_path):
        log(f"  {p}  ({p.stat().st_size / 1e6:,.2f} MB)")
    c = report["checks"]
    log(f"\nVALIDATION: {'PASSED' if not report['errors'] else 'FAILED'} | {c['shape'][0]:,} rows, {c['n_cities']} cities, "
        f"{c['date_min']}..{c['date_max']} | missing={c['missing_total']} dup={c['duplicate_city_date']} "
        f"gaps={'none' if not c['gaps'] else 'YES'} impossible={'none' if not c['impossible_values'] else 'YES'} | "
        f"tomorrow==next-day-today mismatches={c['tomorrow_equals_next_day_today']['mismatches_shift']}")
    if is_subset:
        log(f"NOTE: SUBSET build ({len(cities)} of {n_total} cities) - not the full dataset.")
    if status != "complete":
        log("NOTE: the API budget stopped this run before every requested chunk was fetched; re-run to continue.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
