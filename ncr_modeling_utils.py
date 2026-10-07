"""Leak-safe helpers for modelling the HeatCast NCR dataset.

No model is trained here. These helpers only prepare data:

* ``chronological_split`` / ``expanding_window_folds`` split **by calendar date**, never by row.
  This matters because ``ph_heat_index_next_day.csv`` is sorted by City then Date, so a naive
  row-order split (``train_test_split(shuffle=False)``, ``TimeSeriesSplit`` on the raw frame)
  would split by *city*, not by time. Splitting by date also keeps all cities of one date on the
  same side, which matters because nearby NCR cities share ERA5 grid data.
* ``add_lag_features`` builds 1-3 day lags using only past information.
* ``assert_no_future_information`` refuses feature lists that contain next-day columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "HeatLevelTomorrow"
NUMERIC_TARGET = "HeatIndex_Max_Tomorrow"
CLASS_ORDER = ["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]

DAILY_FEATURES = [
    "Temp_Min", "Temp_Mean", "Temp_Max", "Humidity_Min", "Humidity_Mean", "Humidity_Max",
    "DewPoint_Mean", "Pressure_Mean", "CloudCover_Mean", "WindSpeed_Mean", "WindSpeed_Max",
    "WindGust_Max", "Rainfall_Total", "RainHours", "SolarRadiation_Total", "HeatIndex_Max_Today",
]
CALENDAR_FEATURES = ["Month", "DayOfYear"]
# City/Latitude/Longitude/Elevation identify a city; in NCR several cities share one ERA5 grid
# cell, so these are identifiers rather than independent meteorological information.
STATIC_FEATURES = ["City", "Latitude", "Longitude", "Elevation"]
INPUT_FEATURES = DAILY_FEATURES + CALENDAR_FEATURES + STATIC_FEATURES
# HeatLevelToday is a deterministic binning of HeatIndex_Max_Today (redundant), so it is not listed.
FORBIDDEN_FEATURES = {NUMERIC_TARGET, TARGET}

# Weather variables whose recent history is investigated as lag features.
LAG_SOURCES = [
    "HeatIndex_Max_Today", "Temp_Max", "Humidity_Mean", "DewPoint_Mean", "Pressure_Mean",
    "CloudCover_Mean", "Rainfall_Total", "RainHours", "SolarRadiation_Total", "WindSpeed_Mean",
]
DEFAULT_LAGS = (1, 2, 3)


def assert_no_future_information(columns) -> None:
    """Raise if a candidate feature list contains the targets or anything that looks like tomorrow."""
    bad = [c for c in columns
           if c in FORBIDDEN_FEATURES or any(t in str(c).lower() for t in ("tomorrow", "_lead", "_next", "future"))]
    if bad:
        raise ValueError(f"future information in feature list: {bad}")


# --------------------------------------------------------------------------- #
# Lag features (past information only)
# --------------------------------------------------------------------------- #
def lag_feature_names(columns=LAG_SOURCES, lags=DEFAULT_LAGS, rolling=(3,), deltas=("Pressure_Mean",)) -> list[str]:
    names = [f"{c}_lag{k}" for c in columns for k in lags]
    names += [f"{c}_mean{w}d" for c in columns for w in rolling]
    names += [f"{c}_change1d" for c in deltas if c in columns]
    return names


def add_lag_features(df: pd.DataFrame, columns=LAG_SOURCES, lags=DEFAULT_LAGS, rolling=(3,),
                     deltas=("Pressure_Mean",)) -> pd.DataFrame:
    """Return a copy of ``df`` with past-only features added, computed within each City.

    * ``<col>_lag<k>``   value ``k`` calendar days before the row's Date.
    * ``<col>_mean<w>d`` mean of the last ``w`` days *including today* (today is known at forecast time).
    * ``<col>_change1d`` today minus yesterday (for the columns in ``deltas``).

    Calendar-aware: a lag is NaN unless the row exactly ``k`` days earlier exists, so gaps never pair a
    row with the wrong day. Each city's first ``max(lags)`` days therefore have NaN lags; drop or
    impute them explicitly when modelling. Nothing here looks forward in time.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    idx_cols = ["City", "Date"]
    if out.duplicated(idx_cols).any():
        raise ValueError("duplicate City/Date rows; cannot build lag features")
    base = out.set_index(idx_cols)
    new = {}
    for k in sorted(set(lags) | {1}):
        shifted = base[list(columns)].copy()
        shifted.index = pd.MultiIndex.from_arrays(
            [shifted.index.get_level_values("City"), shifted.index.get_level_values("Date") + pd.Timedelta(days=k)],
            names=idx_cols)
        shifted = shifted.reindex(base.index)
        for c in columns:
            if k in lags:
                new[f"{c}_lag{k}"] = shifted[c].to_numpy()
            if k == 1 and c in deltas:
                new[f"{c}_change1d"] = (base[c] - shifted[c]).to_numpy()
    for w in rolling:
        acc = base[list(columns)].copy()
        for k in range(1, w):
            s = base[list(columns)].copy()
            s.index = pd.MultiIndex.from_arrays(
                [s.index.get_level_values("City"), s.index.get_level_values("Date") + pd.Timedelta(days=k)],
                names=idx_cols)
            acc = acc + s.reindex(base.index)
        for c in columns:
            new[f"{c}_mean{w}d"] = (acc[c] / w).to_numpy()
    lagged = pd.DataFrame(new, index=out.index)
    names = lag_feature_names(columns, lags, rolling, deltas)
    return pd.concat([out, lagged[names]], axis=1)


# --------------------------------------------------------------------------- #
# Chronological splitting (by date, never by row)
# --------------------------------------------------------------------------- #
def chronological_split(df: pd.DataFrame, test_start, embargo_days: int = 1, date_col: str = "Date"):
    """Positional index arrays ``(train_idx, test_idx)``.

    test  = rows with Date >= ``test_start``
    train = rows with Date <  ``test_start`` minus ``embargo_days`` days

    A training row on day d is labelled with day d+1's heat index, so the last training labels are
    computed from the first test day's weather (the test *inputs*, not the test targets). That is
    harmless for a 1-day horizon, but an embargo of 1-3 days is cheap insurance and is essential if
    rolling features are added. All cities of one date always fall on the same side.
    """
    dates = pd.to_datetime(df[date_col]).to_numpy()
    t0 = np.datetime64(pd.Timestamp(test_start))
    train = np.flatnonzero(dates < t0 - np.timedelta64(int(embargo_days), "D"))
    test = np.flatnonzero(dates >= t0)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("split leaves an empty partition")
    return train, test


def expanding_window_folds(df: pd.DataFrame, n_splits: int = 5, gap_days: int = 1, min_train_days: int = 730,
                           date_col: str = "Date"):
    """Expanding-window time-series CV folds over *unique dates*.

    The most recent ``n_splits`` equal blocks of dates are the successive test blocks; fold ``i`` trains on
    every date before its test block minus ``gap_days``. Returns a list of ``(train_idx, test_idx)`` positional
    arrays. Every fold is strictly chronological and keeps each date entirely on one side.
    """
    dates = pd.to_datetime(df[date_col])
    uniq = np.sort(dates.unique())
    n = len(uniq)
    block = (n - min_train_days - gap_days) // n_splits
    if block < 1:
        raise ValueError(f"not enough dates ({n}) for {n_splits} folds with min_train_days={min_train_days}")
    dvals = dates.to_numpy()
    folds = []
    for i in range(n_splits):
        lo = n - (n_splits - i) * block
        hi = lo + block if i < n_splits - 1 else n
        test_dates = (uniq[lo], uniq[hi - 1])
        train_end = uniq[lo] - np.timedelta64(int(gap_days), "D")
        train = np.flatnonzero(dvals < train_end)
        test = np.flatnonzero((dvals >= test_dates[0]) & (dvals <= test_dates[1]))
        folds.append((train, test))
    return folds
