"""Shared, leak-safe preprocessing for the three HeatCast NCR models (project Section 4).

No model is trained here. This module fixes everything the three algorithms must share:

* the data checks (missing values, duplicates, impossible values, label consistency);
* the chronological train / test split and the time-aware cross-validation folds;
* the past-only lag features;
* the scikit-learn transformer that encodes and scales the inputs. It is fitted on the
  training rows only (inside every training fold when used in a ``Pipeline``);
* class weights computed from training rows.

Typical use in a model notebook::

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    import ncr_preprocessing as P

    prep = P.prepare_dataset("data/ncr/output/ph_heat_index_next_day.csv")
    X, y = prep.X(prep.train, "hi_lags"), prep.y(prep.train)
    model = Pipeline([("prep", P.make_preprocessor("hi_lags")), ("clf", LogisticRegression(max_iter=2000))])
    # cross_validate(model, X, y, cv=prep.cv_splits, scoring="f1_macro")   # prep.test stays untouched

Run ``python ncr_preprocessing.py`` to write the model-ready data, the split definition and the report
to ``data/ncr/processed/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import ncr_modeling_utils as U

RANDOM_STATE = 42                 # use this value in every model that has a random component
TEST_FRACTION = 0.20              # course rule: 80 % training / 20 % testing, here by calendar date
N_SPLITS = 5                      # course rule: preferably five-fold cross-validation
MIN_TRAIN_DAYS = 730              # the first CV fold trains on at least two years
MAX_LAG = max(U.DEFAULT_LAGS)     # 3 days of history
HORIZON_DAYS = 1                  # the target is the next day
# A training row on day d uses inputs from d-MAX_LAG..d and its label is day d+1. With this gap no training
# input or label day falls inside the input window of the first test row (and likewise between CV folds).
DEFAULT_GAP_DAYS = MAX_LAG + HORIZON_DAYS

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "ncr" / "output" / "ph_heat_index_next_day.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "ncr" / "processed"

WEATHER_FEATURES = list(U.DAILY_FEATURES)          # 16 daily summaries of the forecast day (incl. heat index)
# Day-to-day changes (EDA 3.4 asks to test them) for the variables that drive the heat index.
DELTA_COLUMNS = ("Pressure_Mean", "Temp_Max", "DewPoint_Mean", "WindSpeed_Mean")
LAG_COLUMNS = U.lag_feature_names(deltas=DELTA_COLUMNS)   # 44 candidate past-only features
FEATURE_SETS = {
    "today": [],                                   # today's weather only: the simplest app input form
    "hi_lags": [f"HeatIndex_Max_Today_lag{k}" for k in U.DEFAULT_LAGS],   # + the heat index of the last 3 days
    "all_lags": list(LAG_COLUMNS),                 # + every candidate lag / 3-day mean / change (research only)
}
# The weather columns the EDA (workspace 3.6) keeps for Logistic Regression after removing collinear ones (all VIF < 6).
# Humidity_Mean is optional there and is left out. The two tree models use the full set.
PRUNED_WEATHER = ["HeatIndex_Max_Today", "Temp_Min", "SolarRadiation_Total", "WindSpeed_Mean", "Pressure_Mean",
                  "CloudCover_Mean", "Rainfall_Total"]
WEATHER_SETS = {"full": WEATHER_FEATURES, "lr_pruned": PRUNED_WEATHER}
LOG_COLUMN_PREFIXES = ("Rainfall_Total", "WindSpeed_Mean")   # right-skewed (EDA 3.7); log1p also covers their lags
LEVEL_CODE = "HeatLevelToday_Code"                 # ordinal 0..4 of today's heat level (optional input, see level_today)
LEVEL_CODES = dict(zip(U.CLASS_ORDER, range(5)))
STATIC_OPTIONS = ("city", "coords", "none")        # how a city is told to the model (see make_preprocessor)

# The class policy is a group decision. "five_class" keeps every label exactly as built. "three_class" is
# only an option that the group may choose later; nothing in this module applies it by default.
TARGET_POLICIES = {
    "five_class": dict(zip(U.CLASS_ORDER, range(5))),
    "three_class": {"Not Hazardous": 0, "Caution": 0, "Extreme Caution": 1, "Danger": 2, "Extreme Danger": 2},
}
POLICY_LABELS = {
    "five_class": list(U.CLASS_ORDER),
    "three_class": ["Caution or below", "Extreme Caution", "Danger or above"],
}

PLAUSIBLE_RANGES = {   # generous physical limits for Metro Manila; anything outside is an error, not an extreme
    "Temp_Min": (10, 45), "Temp_Mean": (10, 45), "Temp_Max": (10, 45),
    "Humidity_Min": (0, 100), "Humidity_Mean": (0, 100), "Humidity_Max": (0, 100),
    "DewPoint_Mean": (0, 40), "Pressure_Mean": (950, 1050), "CloudCover_Mean": (0, 100),
    "WindSpeed_Mean": (0, 200), "WindSpeed_Max": (0, 200), "WindGust_Max": (0, 300),
    "Rainfall_Total": (0, 1000), "RainHours": (0, 24), "SolarRadiation_Total": (0, 12000),
    "HeatIndex_Max_Today": (10, 70), "HeatIndex_Max_Tomorrow": (10, 70),
}
ORDER_RULES = {        # column pairs that must satisfy left <= right
    "Temp_Min <= Temp_Mean": ("Temp_Min", "Temp_Mean"), "Temp_Mean <= Temp_Max": ("Temp_Mean", "Temp_Max"),
    "Humidity_Min <= Humidity_Mean": ("Humidity_Min", "Humidity_Mean"),
    "Humidity_Mean <= Humidity_Max": ("Humidity_Mean", "Humidity_Max"),
    "WindSpeed_Mean <= WindSpeed_Max": ("WindSpeed_Mean", "WindSpeed_Max"),
    "WindSpeed_Max <= WindGust_Max": ("WindSpeed_Max", "WindGust_Max"),
    "DewPoint_Mean <= Temp_Max": ("DewPoint_Mean", "Temp_Max"),
}


# --------------------------------------------------------------------------- #
# Data checks
# --------------------------------------------------------------------------- #
def data_quality_checks(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, impossible values and label consistency of the built dataset.

    ``problems`` lists the checks that must stop the pipeline; ``warnings`` lists the ones worth a look."""
    import build_ph_heat_index_dataset as B

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    problems, warnings = [], []
    out: dict = {"rows": int(len(d)), "columns": int(d.shape[1]), "cities": int(d["City"].nunique())}

    out["missing_cells"] = int(d.isna().sum().sum())
    if out["missing_cells"]:
        warnings.append(f"{out['missing_cells']} missing cells (a training-only median imputer is in the pipeline)")
    out["duplicate_rows"] = int(d.duplicated().sum())
    out["duplicate_city_date"] = int(d.duplicated(["City", "Date"]).sum())
    if out["duplicate_rows"] or out["duplicate_city_date"]:
        problems.append("duplicate rows or duplicate City/Date pairs")

    srt = d.sort_values(["City", "Date"])
    step = srt.groupby("City")["Date"].diff().dropna()
    out["date_gaps"] = int((step != pd.Timedelta(days=1)).sum())
    if out["date_gaps"]:
        warnings.append(f"{out['date_gaps']} gaps in the daily series (lags across a gap are left empty)")
    days = d.groupby("City")["Date"].nunique()
    out["days_per_city"] = [int(days.min()), int(days.max())]

    out["out_of_range"] = {c: int((~d[c].between(lo, hi)).sum()) for c, (lo, hi) in PLAUSIBLE_RANGES.items() if c in d}
    out["order_violations"] = {name: int((d[a] > d[b] + 1e-9).sum()) for name, (a, b) in ORDER_RULES.items()}
    if any(out["out_of_range"].values()) or any(out["order_violations"].values()):
        problems.append("impossible values (outside physical limits or min/mean/max out of order)")

    out["label_mismatch_today"] = int((B.classify_heat_level(d["HeatIndex_Max_Today"]).astype(str)
                                       != d["HeatLevelToday"].astype(str)).sum())
    out["label_mismatch_tomorrow"] = int((B.classify_heat_level(d["HeatIndex_Max_Tomorrow"]).astype(str)
                                          != d["HeatLevelTomorrow"].astype(str)).sum())
    nxt = srt.groupby("City")["HeatIndex_Max_Today"].shift(-1)
    consecutive = srt.groupby("City")["Date"].shift(-1) == srt["Date"] + pd.Timedelta(days=1)
    out["tomorrow_shift_mismatch"] = int(((nxt - srt["HeatIndex_Max_Tomorrow"]).abs() > 1e-9)[consecutive].sum())
    if out["label_mismatch_today"] or out["label_mismatch_tomorrow"] or out["tomorrow_shift_mismatch"]:
        problems.append("labels do not match the heat index, or tomorrow's value is not the next day's value")
    unknown = sorted(set(d["HeatLevelTomorrow"].astype(str)) - set(U.CLASS_ORDER))
    if unknown:
        problems.append(f"unknown class labels {unknown}")
    out["problems"], out["warnings"] = problems, warnings
    return out


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def encode_target(labels, policy: str = "five_class") -> np.ndarray:
    """Integer class codes. ``five_class`` always uses the fixed order of ``U.CLASS_ORDER`` so that
    code 0..4 mean the same thing in every split, even when a class is absent from one of them."""
    if policy not in TARGET_POLICIES:
        raise ValueError(f"unknown target policy {policy!r}; choose from {sorted(TARGET_POLICIES)}")
    s = pd.Series(np.asarray(labels, dtype=object))
    codes = s.map(TARGET_POLICIES[policy])
    if codes.isna().any():
        raise ValueError(f"labels outside the class list: {sorted(set(s[codes.isna()].astype(str)))}")
    return codes.to_numpy(dtype=int)


def class_weights(y, cap: float | None = None) -> dict[int, float]:
    """'Balanced' weights ``n / (k * n_c)`` over the classes that occur in ``y``. Call it on training
    labels only (or a training fold). ``cap`` limits the largest weight, useful for a class with a handful of days."""
    y = np.asarray(y)
    codes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(codes) * counts)
    if cap is not None:
        w = np.minimum(w, cap)
    return {int(c): float(x) for c, x in zip(codes, w)}


# --------------------------------------------------------------------------- #
# The shared transformer
# --------------------------------------------------------------------------- #
_LAG_SOURCE = re.compile(r"^(.*)_(?:lag\d+|mean\d+d|change1d)$")


def feature_columns(feature_set: str = "today", static: str = "city", weather: str = "full",
                    level_today: bool = False) -> list[str]:
    """Raw input columns a feature set needs (this is also the form the app has to collect).

    ``weather="lr_pruned"`` keeps only the weather columns of EDA section 3.6 (and the lags of those columns);
    ``level_today=True`` adds ``HeatLevelToday_Code``, today's heat level as an ordinal 0..4 (EDA 3.4 keeps it; it is a
    binning of ``HeatIndex_Max_Today``, so it adds little and is off by default).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    if static not in STATIC_OPTIONS:
        raise ValueError(f"unknown static option {static!r}; choose from {STATIC_OPTIONS}")
    if weather not in WEATHER_SETS:
        raise ValueError(f"unknown weather set {weather!r}; choose from {sorted(WEATHER_SETS)}")
    base = list(WEATHER_SETS[weather])
    lags = [c for c in FEATURE_SETS[feature_set] if _LAG_SOURCE.match(c).group(1) in base]
    cols = base + lags + ([LEVEL_CODE] if level_today else []) + ["DayOfYear"]
    cols += {"city": ["City"], "coords": ["Latitude", "Longitude", "Elevation"], "none": []}[static]
    U.assert_no_future_information(cols)
    return cols


def _log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))      # rainfall cannot be negative; the clip protects against bad app input


def _day_of_year_cycle(x):
    angle = 2 * np.pi * np.asarray(x, dtype=float).reshape(-1) / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _day_of_year_names(_transformer, _input_features):
    return np.array(["DayOfYear_sin", "DayOfYear_cos"])


def make_preprocessor(feature_set: str = "today", static: str = "city", weather: str = "full",
                      level_today: bool = False) -> ColumnTransformer:
    """The one transformer all three models use. It learns its statistics only when ``fit`` is called, so put it
    in a ``Pipeline`` and every cross-validation fold re-fits it on that fold's training rows.

    * numeric weather and lag columns: median imputation (a safety net, the data has no gaps) then standard scaling;
    * rainfall and mean wind speed columns: ``log1p`` first, because both are right-skewed (EDA 3.7). The tree models are
      unaffected by a monotone transform, so one pipeline serves all three algorithms;
    * ``DayOfYear``: two cyclic columns (sin, cos) so that 31 December sits next to 1 January;
    * ``static``: ``"city"`` one-hot encodes the city (16 columns, it carries the same information as latitude,
      longitude and elevation); ``"coords"`` uses the three numbers instead; ``"none"`` leaves the city out;
    * ``Month`` is not used (``DayOfYear`` carries it); ``Date`` and the targets are never inputs.
    """
    cols = feature_columns(feature_set, static, weather, level_today)
    numeric = [c for c in cols if c not in ("DayOfYear", "City", "Latitude", "Longitude", "Elevation")]
    logged = [c for c in numeric if c.startswith(LOG_COLUMN_PREFIXES)]
    plain = [c for c in numeric if c not in logged]
    parts = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), plain),
        ("logged", Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), logged),
        ("season", FunctionTransformer(_day_of_year_cycle, feature_names_out=_day_of_year_names), ["DayOfYear"]),
    ]
    if static == "city":
        parts.append(("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City"]))
    elif static == "coords":
        parts.append(("coords", StandardScaler(), ["Latitude", "Longitude", "Elevation"]))
    ct = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


# --------------------------------------------------------------------------- #
# Prepared data
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Result of :func:`prepare_dataset`. ``train`` and ``test`` are sorted by Date then City."""
    train: pd.DataFrame
    test: pd.DataFrame
    cv_splits: list
    test_start: pd.Timestamp
    gap_days: int
    target_policy: str
    quality: dict
    counts: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    @property
    def class_labels(self) -> list[str]:
        return POLICY_LABELS[self.target_policy]

    def X(self, frame: pd.DataFrame, feature_set: str = "today", static: str = "city", weather: str = "full",
          level_today: bool = False) -> pd.DataFrame:
        cols = feature_columns(feature_set, static, weather, level_today)
        out = frame[cols].copy()
        if "City" in out:
            out["City"] = out["City"].astype(object)
        return out

    def y(self, frame: pd.DataFrame) -> np.ndarray:
        return encode_target(frame[U.TARGET], self.target_policy)

    def class_weights(self, cap: float | None = None) -> dict[int, float]:
        return class_weights(self.y(self.train), cap)


def prepare_dataset(source, test_fraction: float = TEST_FRACTION, gap_days: int = DEFAULT_GAP_DAYS,
                    n_splits: int = N_SPLITS, min_train_days: int = MIN_TRAIN_DAYS,
                    target_policy: str = "five_class") -> Prepared:
    """Check the data, add the past-only lag features and split by calendar date.

    ``source`` is a path to ``ph_heat_index_next_day.csv`` or a DataFrame. Nothing is fitted here, so nothing
    can leak: the lag features only look backwards, and the statistics of the scaler are learned later, from
    training rows only. The test partition is returned but must stay unused until the final evaluation.
    """
    df = pd.read_csv(source, parse_dates=["Date"]) if not isinstance(source, pd.DataFrame) else source.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    quality = data_quality_checks(df)
    if quality["problems"]:
        raise ValueError("data checks failed: " + "; ".join(quality["problems"]))

    dates = np.sort(df["Date"].unique())
    test_start = pd.Timestamp(dates[int(round(len(dates) * (1 - test_fraction)))])

    lagged = U.add_lag_features(df, deltas=DELTA_COLUMNS)
    n_before = len(lagged)
    lagged = lagged.dropna(subset=LAG_COLUMNS)            # each city's first 3 days have no full history
    dropped_lag_rows = n_before - len(lagged)
    lagged[LEVEL_CODE] = lagged["HeatLevelToday"].map(LEVEL_CODES).astype(int)
    lagged = lagged.sort_values(["Date", "City"], kind="mergesort").reset_index(drop=True)

    train_idx, test_idx = U.chronological_split(lagged, test_start, embargo_days=gap_days)
    train = lagged.iloc[train_idx].reset_index(drop=True)
    test = lagged.iloc[test_idx].reset_index(drop=True)
    cv = U.expanding_window_folds(train, n_splits=n_splits, gap_days=gap_days, min_train_days=min_train_days)
    encode_target(lagged[U.TARGET], target_policy)        # fail early on an unknown policy or label
    counts = {"rows_in": n_before, "dropped_for_lag_history": dropped_lag_rows,
              "gap_rows": len(lagged) - len(train) - len(test), "train": len(train), "test": len(test)}
    nh = lagged.loc[lagged[U.TARGET] == "Not Hazardous", "Date"]
    facts = {"not_hazardous_days": int(nh.nunique()),
             # the label is tomorrow's level, so the heat day itself is one day after the forecast date
             "last_not_hazardous_day": (nh.max() + pd.Timedelta(days=1)).date().isoformat() if len(nh) else None}
    return Prepared(train, test, cv, test_start, gap_days, target_policy, quality, counts, facts)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame, fmt: str = "{:,.2f}", index: bool = True) -> str:
    t = df.reset_index() if index else df

    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else fmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    head = "| " + " | ".join(str(c) for c in t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False))


def _partial_corr(x, y, z) -> float:
    zz = np.column_stack([np.ones(len(z)), z])
    rx = x - zz @ np.linalg.lstsq(zz, x, rcond=None)[0]
    ry = y - zz @ np.linalg.lstsq(zz, y, rcond=None)[0]
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def class_count_table(prep: Prepared) -> pd.DataFrame:
    """Rows per class in the training set, each validation block and the test set."""
    labels = prep.class_labels
    cols = {"train (all)": prep.y(prep.train)}
    for i, (tr, va) in enumerate(prep.cv_splits, 1):
        y = prep.y(prep.train.iloc[va])
        cols[f"fold {i} valid."] = y
    cols["test"] = prep.y(prep.test)
    data = {name: [int((y == k).sum()) for k in range(len(labels))] for name, y in cols.items()}
    return pd.DataFrame(data, index=pd.Index(labels, name="class"))


def fold_table(prep: Prepared) -> pd.DataFrame:
    rows = []
    for i, (tr, va) in enumerate(prep.cv_splits, 1):
        a, b = prep.train.iloc[tr], prep.train.iloc[va]
        rows.append({"fold": i, "train from": a["Date"].min().date().isoformat(), "train to": a["Date"].max().date().isoformat(),
                     "train rows": len(tr), "valid. from": b["Date"].min().date().isoformat(),
                     "valid. to": b["Date"].max().date().isoformat(), "valid. rows": len(va)})
    return pd.DataFrame(rows).set_index("fold")


def outlier_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in WEATHER_FEATURES:
        x = train[c]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        far = int(((x < q1 - 3 * (q3 - q1)) | (x > q3 + 3 * (q3 - q1))).sum())
        rows.append({"column": c, "min": x.min(), "median": x.median(), "99th pct": x.quantile(0.99), "max": x.max(),
                     "skewness": x.skew(), "rows beyond 3 x IQR": far})
    return pd.DataFrame(rows).set_index("column")


def storm_evidence(train: pd.DataFrame, n: int = 20) -> dict:
    """Do the extreme days look like storms (they come in spells) or like sensor glitches (isolated spikes)?
    For the ``n`` rainiest and the ``n`` windiest training dates, look at the neighbouring days."""
    daily = train.groupby("Date")[["Rainfall_Total", "WindGust_Max"]].max()
    day = pd.Timedelta(days=1)

    def neighbours(col):
        top = daily[col].nlargest(n).index
        return np.array([max(daily[col].get(d - day, 0.0), daily[col].get(d + day, 0.0)) for d in top])

    rain, gust = neighbours("Rainfall_Total"), neighbours("WindGust_Max")
    return {"n": n, "rain_spell": int((rain >= 10).sum()), "rain_typical": float(daily["Rainfall_Total"].median()),
            "gust_min_neighbour": float(gust.min()), "gust_typical": float(daily["WindGust_Max"].median())}


def lag_screening(train: pd.DataFrame) -> pd.DataFrame:
    """Linear screening of the lag features on TRAINING rows only: partial correlation with tomorrow's heat index
    after removing today's heat index and the season. It informs the choice of feature set, it does not fix it."""
    angle = 2 * np.pi * train["DayOfYear"].to_numpy() / 365.25
    ctrl = np.column_stack([train["HeatIndex_Max_Today"].to_numpy(), np.sin(angle), np.cos(angle)])
    y = train[U.NUMERIC_TARGET].to_numpy()
    rows = [{"feature": c, "partial corr.": _partial_corr(train[c].to_numpy(), y, ctrl)} for c in LAG_COLUMNS]
    t = pd.DataFrame(rows).set_index("feature")
    t["useful (|r| >= 0.05)"] = t["partial corr."].abs() >= 0.05
    return t.reindex(t["partial corr."].abs().sort_values(ascending=False).index)


def correlated_pairs(train: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr = train[WEATHER_FEATURES].corr().abs()
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(WEATHER_FEATURES) for b in WEATHER_FEATURES[i + 1:]
             if corr.loc[a, b] >= threshold]
    return pd.DataFrame(pairs, columns=["feature A", "feature B", "|r| (training rows)"]).sort_values(
        "|r| (training rows)", ascending=False).set_index("feature A")


def split_definition(prep: Prepared) -> dict:
    labels = prep.class_labels
    return {
        "random_state": RANDOM_STATE, "target": U.TARGET, "target_policy": prep.target_policy, "class_labels": labels,
        "test_fraction": TEST_FRACTION, "test_start": prep.test_start.date().isoformat(),
        "gap_days": prep.gap_days, "n_splits": len(prep.cv_splits), "min_train_days": MIN_TRAIN_DAYS,
        "train_dates": [prep.train["Date"].min().date().isoformat(), prep.train["Date"].max().date().isoformat()],
        "test_dates": [prep.test["Date"].min().date().isoformat(), prep.test["Date"].max().date().isoformat()],
        "rows": prep.counts,
        "folds": [{"fold": i, "train_dates": [prep.train.iloc[tr]["Date"].min().date().isoformat(),
                                              prep.train.iloc[tr]["Date"].max().date().isoformat()],
                   "valid_dates": [prep.train.iloc[va]["Date"].min().date().isoformat(),
                                   prep.train.iloc[va]["Date"].max().date().isoformat()],
                   "train_rows": int(len(tr)), "valid_rows": int(len(va))} for i, (tr, va) in enumerate(prep.cv_splits, 1)],
        "feature_sets": {k: feature_columns(k, "city") for k in FEATURE_SETS},
        "weather_sets": {k: list(v) for k, v in WEATHER_SETS.items()},
        "level_today_column": LEVEL_CODE,
        "log1p_column_prefixes": list(LOG_COLUMN_PREFIXES),
        "static_options": list(STATIC_OPTIONS),
        "class_weights_balanced_train": {labels[k]: round(v, 3) for k, v in prep.class_weights().items()},
    }


def render_report(prep: Prepared) -> str:
    q, tr, te = prep.quality, prep.train, prep.test
    labels = prep.class_labels
    counts = class_count_table(prep)
    weights = prep.class_weights()
    y_tr = prep.y(tr)
    absent = [labels[k] for k in range(len(labels)) if (y_tr == k).sum() == 0]
    test_absent = [labels[k] for k in range(len(labels)) if (prep.y(te) == k).sum() == 0]
    lag = lag_screening(tr)
    useful = lag[lag["useful (|r| >= 0.05)"]]
    corr = correlated_pairs(tr)
    out_t = outlier_table(tr)
    c = prep.counts
    d = lambda ts: ts.date().isoformat()  # noqa: E731

    def n_cols(feature_set, **kw):
        return len(make_preprocessor(feature_set, **kw).fit(prep.X(tr, feature_set, **kw)).get_feature_names_out())

    n_pre, n_hi, n_all = n_cols("today"), n_cols("hi_lags"), n_cols("all_lags")
    n_pruned, n_level = n_cols("today", weather="lr_pruned"), n_cols("today", level_today=True)
    skew = lambda col: (float(tr[col].skew()), float(np.log1p(tr[col]).skew()))  # noqa: E731
    rain_skew, wind_skew = skew("Rainfall_Total"), skew("WindSpeed_Mean")

    L = []
    L.append("# HeatCast NCR - data preparation and preprocessing (project Section 4)\n")
    L.append("Generated by `python ncr_preprocessing.py`. **No model has been trained.** Everything below is computed from the "
             "committed dataset `data/ncr/output/ph_heat_index_next_day.csv`; statistics that guide a decision use the training "
             "rows only. The test rows are written to disk but nothing here looks at them except to count rows and classes.\n")
    L.append("## Decisions at a glance (workspace table 4.2)\n")
    L.append(_md(pd.DataFrame([
        ("Missing values", f"None in the source ({q['missing_cells']} cells). Each city's first {MAX_LAG} days have no lag history, "
                           f"so {c['dropped_for_lag_history']} rows are dropped. A median imputer fitted on training rows is in the pipeline as a safety net.",
         f"Nothing to repair; dropping {c['dropped_for_lag_history']} of {c['rows_in']:,} rows keeps all feature sets on identical rows."),
        ("Duplicates / invalid values", f"{q['duplicate_rows']} duplicate rows, {q['duplicate_city_date']} duplicate City/Date pairs, "
                                        f"{sum(q['out_of_range'].values())} values outside physical limits, {sum(q['order_violations'].values())} min/mean/max "
                                        f"order violations, {q['label_mismatch_today'] + q['label_mismatch_tomorrow'] + q['tomorrow_shift_mismatch']} label or next-day mismatches. "
                                        "No action needed.",
         "Checked by `data_quality_checks`; the pipeline stops if any check fails."),
        ("Outliers", "Kept. No value is an error: the extremes come in storm spells, not as isolated spikes (evidence below). Rainfall and mean wind speed are log-transformed.",
         "Dropping extremes would remove real weather the models must handle."),
        ("Encoding", "City one-hot (16 columns) by default; DayOfYear as sin/cos (Month and raw DayOfYear are not inputs, as in EDA 3.5); Date and the targets are not inputs. "
                     "Class labels get fixed integer codes.", "One-hot city equals latitude + longitude + elevation. Cyclic season avoids a jump from Dec 31 to Jan 1."),
        ("Scaling", "log1p of Rainfall_Total and WindSpeed_Mean (and their lags), then standard scaling of all numeric columns, fitted on training rows only. "
                    "Same transformer for all three models.",
         "Needed by Logistic Regression; a monotone transform and scaling do not change a tree model. The course asks for identical preprocessing."),
        ("Feature selection / engineering", f"Three named feature sets: `today` ({n_pre} columns), `hi_lags` ({n_hi}), `all_lags` ({n_all}). "
                                            f"Two options from the EDA: `weather='lr_pruned'` (the 7 weather columns of EDA 3.6, {n_pruned} columns with `today`) and "
                                            f"`level_today=True` (adds today's heat level as a 0-4 code, {n_level} columns with `today`). "
                                            "No correlated feature is removed from the full set. The set is chosen by cross-validation on the training rows.",
         "Lags are past-only. Heat-index lags carry signal beyond today's value (screening below). Collinearity matters for Logistic Regression only."),
        ("Class imbalance", f"Balanced class weights computed from training labels. No resampling. No class merged or dropped (group decision, see below).",
         "Resampling would copy near-identical rows (16 cities share weather) and cannot invent the missing class."),
        ("Leakage prevention", "Split by date, gap between train and test and between folds, transformer fitted inside each training fold, "
                               "forbidden-column guard, past-only lags, test untouched.", "Listed in full below."),
    ], columns=["Decision", "Choice", "Why"]), index=False))
    L.append("\n## Shared experiment settings (workspace table 4.1)\n")
    L.append(_md(pd.DataFrame([
        ("Problem type", "Classification"),
        ("Target", f"`{U.TARGET}` ({len(labels)} classes under the `{prep.target_policy}` policy: {', '.join(labels)})"),
        ("Train / test split", f"80 % / 20 % of the calendar dates. Test: {d(prep.test_start)} to {d(te['Date'].max())} "
                               f"({te['Date'].nunique()} days, {len(te):,} rows). Train: {d(tr['Date'].min())} to {d(tr['Date'].max())} "
                               f"({tr['Date'].nunique()} days, {len(tr):,} rows)."),
        ("Split strategy", "Chronological, by date. All 16 cities of one date are always on the same side. Not stratified: stratifying would shuffle time."),
        ("Gap", f"{prep.gap_days} days ({MAX_LAG} days of lag + {HORIZON_DAYS} day of horizon) between train and test and before every validation block. "
                f"{c['gap_rows']:,} rows fall in the gap and are used by neither side."),
        ("Cross-validation", f"{len(prep.cv_splits)} expanding-window folds inside the training period (table below), the same folds for all three models."),
        ("Random state", f"{RANDOM_STATE}, for every model with a random component."),
        ("Primary metric", "Not decided here (group decision). Proposed: macro F1 over the classes that occur, plus Danger recall."),
        ("Final test rule", "Untouched until model selection; the selected model is evaluated once."),
    ], columns=["Setting", "Value"]), index=False))
    L.append("\n## 1. Data checks\n")
    L.append(f"- Rows x columns: {q['rows']:,} x {q['columns']}; {q['cities']} cities, {q['days_per_city'][0]:,} to {q['days_per_city'][1]:,} days each; "
             f"{q['date_gaps']} gaps in the daily series.")
    L.append(f"- Missing cells: {q['missing_cells']}. Duplicate rows: {q['duplicate_rows']}. Duplicate City/Date pairs: {q['duplicate_city_date']}.")
    L.append(f"- Values outside physical limits (e.g. humidity outside 0-100 %, rain below 0, hours outside 0-24): {sum(q['out_of_range'].values())}.")
    L.append(f"- Order rules (Temp_Min <= Temp_Mean <= Temp_Max, humidity, wind <= gust, dew point <= Temp_Max): {sum(q['order_violations'].values())} violations.")
    L.append(f"- Heat levels recomputed from the heat index disagree with the stored label on {q['label_mismatch_today']} (today) and "
             f"{q['label_mismatch_tomorrow']} (tomorrow) rows; tomorrow's heat index differs from the next day's value on {q['tomorrow_shift_mismatch']} rows.")
    L.append("\n## 2. Split and cross-validation folds\n")
    L.append(_md(fold_table(prep)))
    L.append("\nRows per class (training set, each validation block, test set):\n")
    L.append(_md(counts, fmt="{:,.0f}"))
    L.append("")
    L.append(f"- Classes with no training rows: {', '.join(absent) if absent else 'none'}. Classes with no test rows: {', '.join(test_absent) if test_absent else 'none'}.")
    seen = [k for k in range(len(labels)) if (y_tr == k).sum() > 0]
    gaps = []
    for i in range(1, len(prep.cv_splits) + 1):
        miss = [labels[k] for k in seen if counts.loc[labels[k], f"fold {i} valid."] == 0]
        if miss:
            gaps.append(f"fold {i} ({', '.join(miss)})")
    L.append(f"- Validation blocks that lack a class the training set has: {'; '.join(gaps) if gaps else 'none'}. "
             "Macro F1 must be computed over the classes present in the block being scored, or an absent class would count as a score of 0.")
    L.append("\n## 3. Outliers\n")
    L.append(_md(out_t, fmt="{:,.2f}"))
    st = storm_evidence(tr)
    L.append(f"\nEvery maximum is inside the physical limits, so none is treated as a data error and none is removed. The extremes look like storms, "
             f"not sensor glitches: {st['rain_spell']} of the {st['n']} rainiest training days have at least 10 mm of rain on a neighbouring day "
             f"(the typical day's highest city value is {st['rain_typical']:.1f} mm), and every one of the {st['n']} windiest days has gusts of "
             f"at least {st['gust_min_neighbour']:.0f} km/h on a neighbouring day (typical: {st['gust_typical']:.0f} km/h). "
             f"After log1p the skewness of Rainfall_Total falls from {rain_skew[0]:.2f} to {rain_skew[1]:.2f} and that of WindSpeed_Mean from "
             f"{wind_skew[0]:.2f} to {wind_skew[1]:.2f} (training rows); the tree models are not affected by the transform.\n")
    L.append("## 4. Feature engineering and selection\n")
    L.append(f"- Candidate lag features: {len(LAG_COLUMNS)} (1-3 day lags and a 3-day mean of ten weather variables, plus 1-day changes of "
             f"{', '.join(DELTA_COLUMNS)}). Computed within each city, strictly from earlier days.")
    L.append(f"- Linear screening on the training rows: {len(useful)} of {len(lag)} lag features have |partial correlation| >= 0.05 with tomorrow's heat index "
             "after removing today's heat index and the season. The strongest:\n")
    L.append(_md(lag.head(10), fmt="{:,.3f}"))
    L.append("\n- `today` uses no history, so the app only needs today's weather. `hi_lags` adds the heat index of the previous 3 days "
             f"(three extra numbers for the app). `all_lags` is for research: an app cannot ask a user for {len(LAG_COLUMNS)} extra numbers.")
    if len(corr):
        L.append("\nPairs of weather features with |r| >= 0.9 on the training rows (kept; Logistic Regression is regularised and trees do not mind):\n")
        L.append(_md(corr, fmt="{:,.3f}"))
    L.append("\n## 5. Class imbalance and the two rare classes (needs a group decision)\n")
    w_rows = pd.DataFrame({"training rows": [int((y_tr == k).sum()) for k in range(len(labels))],
                           "balanced weight": [weights.get(k, float('nan')) for k in range(len(labels))]}, index=pd.Index(labels, name="class"))
    L.append(_md(w_rows, fmt="{:,.2f}"))
    L.append("")
    L.append("Nothing is merged or dropped. The two options for the rare classes, for the group to choose:\n")
    L.append("1. **Keep five classes** (default). The models can never predict a class with no training rows, and the very small class gets "
             "a large weight; scores must be computed over the classes that occur.")
    L.append("2. **Three classes** (`target_policy='three_class'`): Caution or below / Extreme Caution / Danger or above. This merges the two "
             "rare classes into their neighbours. It is implemented but **not applied** until the group agrees.\n")
    L.append("## 6. How this follows the EDA decisions (workspace Section 3)\n")
    nh_days, nh_last = prep.facts["not_hazardous_days"], prep.facts["last_not_hazardous_day"]
    L.append(_md(pd.DataFrame([
        ("Drop `HeatIndex_Max_Tomorrow` (3.1)", "Done. Never an input; kept in the data only for diagnostics."),
        ("Cyclic season, drop raw `DayOfYear` (3.5)", "Done with `DayOfYear` as sin/cos. `Month` and raw `DayOfYear` are not inputs (they carry the same information)."),
        ("log1p of `Rainfall_Total` and `WindSpeed_Mean` (3.7)", "Done for all three models, because the course asks for one shared preprocessing and trees ignore a monotone transform."),
        ("Pruned weather set for Logistic Regression, full set for the trees (3.6)", "Available as `weather='lr_pruned'` (7 columns); the default is the full set."),
        ("Keep `HeatLevelToday`, ordinal (3.4)", f"Available as `level_today=True` (`{LEVEL_CODE}`, 0-4). Off by default: it is a binning of `HeatIndex_Max_Today` and is not in the "
                                                  "specified feature list. **Open: the group decides.**"),
        ("Class weights, no heavy resampling (3.2)", "Done: `class_weights` from training labels, no resampling."),
        ("Also test merging Not Hazardous with Caution (3.2)", "Available as `target_policy='three_class'` (Caution or below / Extreme Caution / Danger or above). Not applied."),
        ("Day-to-day differences (3.4)", f"Added as candidate features: 1-day changes of {', '.join(DELTA_COLUMNS)} (part of `all_lags`)."),
        ("Distance to the 33 and 42 degree cut-offs (3.3)", "Not added: each is `HeatIndex_Max_Today` minus a constant, so it carries no new information for a linear or a tree model. "
                                                          "An absolute distance could be tested later."),
        ("Chronological split, e.g. 2015-2022 / 2023-2024 / 2025 (3.8)", "Section 4 uses the course default instead: 80 / 20 by date, with 5 time-aware folds inside the training period as the validation. "
                                                                          "That example split is not used."),
    ], columns=["EDA decision", "In the preprocessing"]), index=False))
    L.append("")
    L.append(f"- Section 3.2 says chronological splits keep every class in each part. That cannot hold: Not Hazardous occurs on {nh_days} days in total and "
             f"the last such day is {nh_last}, so no test period that ends in 2025 contains it, and Extreme Danger never occurs.")
    L.append("\n## 7. Leakage prevention\n")
    L.extend([
        "1. The split is by calendar date and never by row (the CSV is sorted by city, so a row split would split by city).",
        f"2. A {prep.gap_days}-day gap separates training from the test period and each validation block from the days before it.",
        "3. The transformer learns its means, scales and city list only in `fit`, on training rows. In a `Pipeline` every cross-validation fold re-fits it on that fold's training rows.",
        "4. Lag features look backwards only, and are computed within each city on exact calendar days.",
        "5. `HeatIndex_Max_Tomorrow`, `HeatLevelTomorrow` and `HeatLevelToday` are never inputs; `assert_no_future_information` enforces it.",
        "6. Class weights come from training labels only.",
        "7. All 16 cities of a date stay together, because nearby cities share the same ERA5 weather.",
        "8. The test partition is not used to choose anything in this module.\n",
    ])
    L.append("## Files\n")
    L.append("- `ncr_model_ready.csv.gz` - the checked data with the lag features, a `Split` column (`train` / `test`), sorted by Date then City "
             "(read it with `pd.read_csv(path)`; gzip keeps it at about 2 MB). Rows in the gap and rows without lag history are not included.")
    L.append("- `split_definition.json` - the split date, gap, fold dates, row counts, feature sets and class weights.")
    L.append("- `ncr_preprocessing.py` - the code (`prepare_dataset`, `make_preprocessor`, `class_weights`).")
    return "\n".join(L) + "\n"


def write_outputs(prep: Prepared, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([prep.train.assign(Split="train"), prep.test.assign(Split="test")], ignore_index=True)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    paths = [out_dir / "ncr_model_ready.csv.gz", out_dir / "split_definition.json", out_dir / "preprocessing_report.md"]
    frame.round(4).to_csv(paths[0], index=False)          # gzip is chosen from the file name; pandas and Excel-less tools read it directly
    paths[1].write_text(json.dumps(split_definition(prep), indent=2) + "\n", encoding="utf-8")
    paths[2].write_text(render_report(prep), encoding="utf-8")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_CSV), help="ph_heat_index_next_day.csv")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    ap.add_argument("--target-policy", default="five_class", choices=sorted(TARGET_POLICIES))
    a = ap.parse_args(argv)
    prep = prepare_dataset(a.data, gap_days=a.gap_days, target_policy=a.target_policy)
    for p in write_outputs(prep, Path(a.out_dir)):
        print("wrote", p)
    print(f"train {len(prep.train):,} rows, test {len(prep.test):,} rows, test starts {prep.test_start.date()}, gap {prep.gap_days} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
