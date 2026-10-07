"""Local stand-in for the Open-Meteo archive endpoint.  FOR TESTING ONLY.

The data it serves are SYNTHETIC (seasonal + diurnal cycles plus noise, snapped to a
0.25 degree grid like ERA5). They exist so the pipeline's fetching, caching, retry,
aggregation and validation logic can be exercised without network access. Nothing
produced from this server may be mistaken for real weather data.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

HOURLY_UNITS = {
    "time": "iso8601", "temperature_2m": "°C", "relative_humidity_2m": "%", "dew_point_2m": "°C",
    "pressure_msl": "hPa", "cloud_cover": "%", "precipitation": "mm", "rain": "mm",
    "shortwave_radiation": "W/m²", "wind_speed_10m": "km/h", "wind_gusts_10m": "km/h",
}
SERIES_START = pd.Timestamp("2015-01-01 00:00")
SERIES_END = pd.Timestamp("2025-12-31 23:00")


def call_weight(n_days: int, n_vars: int = 10) -> float:
    var_w = n_vars / 10.0
    return max(1.0, max(var_w, (n_days / 14.0) * var_w))


class MockOpenMeteo:
    """Threaded HTTP server. `inject(kind, count)` queues faults served before real data."""

    FAULTS = {"http500", "http503", "rate_minutely", "rate_hourly", "rate_daily",
              "truncated", "short", "wrong_tz", "bad_request"}

    def __init__(self, daily_weight_limit: float | None = None, null_vars: set[str] | None = None):
        self.requests: list[dict] = []
        self.weight_served = 0.0
        self.daily_weight_limit = daily_weight_limit
        self.null_vars = set(null_vars or ())
        self._faults: deque[str] = deque()
        self._cells: dict[tuple[float, float], dict] = {}
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):  # noqa: N802
                outer._handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    # ---- lifecycle -------------------------------------------------------
    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/archive"

    def inject(self, kind: str, count: int = 1) -> None:
        assert kind in self.FAULTS, kind
        with self._lock:
            self._faults.extend([kind] * count)

    def data_requests(self) -> list[dict]:
        """Requests that were not part of a fault response."""
        return [r for r in self.requests if r["served"] == "data"]

    # ---- synthetic data --------------------------------------------------
    def _cell(self, glat: float, glon: float) -> dict:
        key = (glat, glon)
        with self._lock:
            if key in self._cells:
                return self._cells[key]
        idx = pd.date_range(SERIES_START, SERIES_END, freq="h")
        n_days = len(idx) // 24
        rng = np.random.default_rng(int(round(glat * 100)) * 100_003 + int(round(glon * 100)))
        doy = idx.dayofyear.values
        hour = idx.hour.values
        day = np.arange(len(idx)) // 24
        seasonal = 2.2 * np.sin(2 * np.pi * (doy - 110) / 365.25)          # hottest Apr-May
        diurnal = 3.8 * np.sin(2 * np.pi * (hour - 9) / 24)                  # peak ~15:00
        heatwave = np.where(rng.random(n_days) < 0.03, rng.uniform(2.5, 5.0, n_days), 0.0)[day]
        daily_noise = rng.normal(0, 1.1, n_days)[day]
        temp = 28.0 - 0.25 * abs(glat - 12) + seasonal + diurnal + daily_noise + heatwave
        wet = rng.random(n_days) < (0.35 + 0.25 * np.sin(2 * np.pi * (doy[::24] - 200) / 365.25))
        rh_base = np.where(wet, 88.0, 72.0)[day] - 2.2 * (temp - 28.0) + rng.normal(0, 2.5, len(idx))
        rh = np.clip(rh_base, 35.0, 100.0)
        a = 17.27 * temp / (237.7 + temp) + np.log(rh / 100.0)
        dew = 237.7 * a / (17.27 - a)
        pressure = 1010 + 3 * np.sin(2 * np.pi * (hour - 4) / 12) + rng.normal(0, 1.5, n_days)[day]
        cloud = np.clip(np.where(wet, 80.0, 40.0)[day] + rng.normal(0, 15, len(idx)), 0, 100)
        rain = np.where(wet[day] & (rng.random(len(idx)) < 0.25), rng.exponential(1.8, len(idx)), 0.0)
        sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 950 * (1 - 0.6 * cloud / 100)
        wind = np.clip(8 + 5 * rng.random(len(idx)) + 2 * np.sin(2 * np.pi * (hour - 8) / 24), 0.5, None)
        gust = wind * (1.4 + 0.5 * rng.random(len(idx)))
        cell = {
            "elevation": float(abs(hash((glat, glon))) % 800),
            "values": {
                "temperature_2m": temp, "relative_humidity_2m": rh, "dew_point_2m": dew,
                "pressure_msl": pressure, "cloud_cover": cloud, "precipitation": rain, "rain": rain,
                "shortwave_radiation": sun, "wind_speed_10m": wind, "wind_gusts_10m": gust,
            },
        }
        with self._lock:
            self._cells[key] = cell
        return cell

    def _payload(self, q: dict) -> dict:
        lat, lon = float(q["latitude"][0]), float(q["longitude"][0])
        glat, glon = round(lat * 4) / 4, round(lon * 4) / 4
        start, end = pd.Timestamp(q["start_date"][0]), pd.Timestamp(q["end_date"][0]) + pd.Timedelta(hours=23)
        times = pd.date_range(start, end, freq="h")
        lo = int((times[0] - SERIES_START) / pd.Timedelta(hours=1))
        cell = self._cell(glat, glon)
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for v in q["hourly"][0].split(","):
            if v in self.null_vars:
                hourly[v] = [None] * len(times)
            else:
                hourly[v] = np.round(cell["values"][v][lo:lo + len(times)], 2).tolist()
        return {"latitude": glat, "longitude": glon, "generationtime_ms": 1.0, "utc_offset_seconds": 28800,
                "timezone": "Asia/Manila", "timezone_abbreviation": "PST", "elevation": cell["elevation"],
                "hourly_units": {k: HOURLY_UNITS[k] for k in ["time"] + q["hourly"][0].split(",")},
                "hourly": hourly}

    # ---- HTTP ------------------------------------------------------------
    def _send(self, h: BaseHTTPRequestHandler, status: int, body: str | dict, extra_headers: dict | None = None):
        raw = (body if isinstance(body, str) else json.dumps(body)).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json; charset=utf-8")
        h.send_header("Content-Length", str(len(raw)))
        for k, v in (extra_headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(raw)

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        q = parse_qs(urlparse(h.path).query)
        rec = {"lat": q.get("latitude", [None])[0], "lon": q.get("longitude", [None])[0],
               "start": q.get("start_date", [None])[0], "end": q.get("end_date", [None])[0],
               "models": q.get("models", [None])[0], "tz": q.get("timezone", [None])[0],
               "apikey": q.get("apikey", [None])[0], "served": "data"}
        with self._lock:
            self.requests.append(rec)
            fault = self._faults.popleft() if self._faults else None
        try:
            if fault:
                rec["served"] = fault
                if fault == "http500":
                    return self._send(h, 500, {"error": True, "reason": "Internal server error"})
                if fault == "http503":
                    return self._send(h, 503, "Service Unavailable")
                if fault == "rate_minutely":
                    return self._send(h, 429, {"error": True, "reason": "Minutely API request limit exceeded. Please try again in one minute."})
                if fault == "rate_hourly":
                    return self._send(h, 429, {"error": True, "reason": "Hourly API request limit exceeded. Please try again in the next hour."})
                if fault == "rate_daily":
                    return self._send(h, 429, {"error": True, "reason": "Daily API request limit exceeded. Please try again tomorrow."})
                if fault == "bad_request":
                    return self._send(h, 400, {"error": True, "reason": "Cannot initialize SurfaceVariable from invalid String value foo."})
                payload = self._payload(q)
                if fault == "truncated":
                    return self._send(h, 200, json.dumps(payload)[:5000])
                if fault == "short":
                    payload["hourly"] = {k: v[:-5] for k, v in payload["hourly"].items()}
                    return self._send(h, 200, payload)
                if fault == "wrong_tz":
                    payload["timezone"], payload["utc_offset_seconds"] = "GMT", 0
                    return self._send(h, 200, payload)
            n_days = (dt.date.fromisoformat(rec["end"]) - dt.date.fromisoformat(rec["start"])).days + 1
            weight = call_weight(n_days, len(q["hourly"][0].split(",")))
            with self._lock:
                if self.daily_weight_limit is not None and self.weight_served + weight > self.daily_weight_limit:
                    rec["served"] = "rate_daily"
                    over = True
                else:
                    self.weight_served += weight
                    over = False
            if over:
                return self._send(h, 429, {"error": True, "reason": "Daily API request limit exceeded. Please try again tomorrow."})
            return self._send(h, 200, self._payload(q))
        except (BrokenPipeError, ConnectionResetError):  # client gave up
            pass
