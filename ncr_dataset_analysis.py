#!/usr/bin/env python3
"""Pre-modelling analysis of the HeatCast NCR dataset.  No model is trained.

Everything here is model-free (counting, correlations, rule-based reference baselines) and answers,
for the *actual* data produced by build_ph_heat_index_dataset.py:

  1. how many distinct ERA5 grid series the NCR cities really resolve to;
  2. whether Open-Meteo's elevation adjustment separates cities in any meaningful way;
  3. class counts / percentages, with special attention to Danger and Extreme Danger;
  4. which classes have too few observations for reliable modelling (flagged, never merged or dropped);
  5. how shared grid series inflate the sample and could leak across train/test splits;
  6. which 1-3 day lag features (past information only) carry incremental information;
  7. reference baselines (persistence, climatology, majority) that any model must beat;
  8. methodological recommendations that keep NCR as the geographic scope.

Usage:
    python ncr_dataset_analysis.py --dataset out/ph_heat_index_next_day.csv \
        --city-list out/ph_heat_index_city_list.csv --out-dir out
(The build script runs this automatically after validating the dataset.)
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import ncr_modeling_utils as U

NONTHERMAL = ["Pressure_Mean", "CloudCover_Mean", "WindSpeed_Mean", "WindSpeed_Max", "WindGust_Max",
              "Rainfall_Total", "RainHours", "SolarRadiation_Total"]
THERMAL = ["Temp_Min", "Temp_Mean", "Temp_Max", "Humidity_Min", "Humidity_Mean", "Humidity_Max",
           "DewPoint_Mean", "HeatIndex_Max_Today"]
WEATHER = NONTHERMAL + THERMAL
HI = "HeatIndex_Max_Today"

# Heuristic thresholds (documented in the report; tune freely, nothing is changed automatically).
MIN_DAYS_OVERALL = 100      # distinct event days a class needs overall
MIN_EPISODES = 10           # distinct heat episodes (runs of consecutive days)
VERY_FEW_DAYS, VERY_FEW_EPISODES = 30, 5
MIN_DAYS_TEST = 30          # distinct event days in the chronological test partition
MATERIAL_HI_DIFF_C, MODEST_HI_DIFF_C = 0.5, 0.2
MATERIAL_CLASS_DISAGREE, MODEST_CLASS_DISAGREE = 0.02, 0.005
STANDARD_LAPSE_C_PER_100M = -0.65
USEFUL_PARTIAL_R = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}", index: bool = True) -> str:
    d = df.reset_index() if index else df
    if index:  # index values are labels (year, month, fold ...), not quantities: no thousands separators
        d = d.copy()
        d[d.columns[0]] = d[d.columns[0]].astype(str)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n"
    rows = []
    for rec in d.itertuples(index=False):
        cells = []
        for v in rec:
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def cell_ids(cities: pd.DataFrame) -> dict[str, str]:
    need = {"City", "grid_latitude", "grid_longitude", "elevation_m"}
    if not need <= set(cities.columns) or cities[["grid_latitude", "grid_longitude"]].isna().any().any():
        raise ValueError(f"city list needs API-derived columns {sorted(need)} (run the build to obtain them)")
    return {r.City: f"{r.grid_latitude:.2f}N/{r.grid_longitude:.2f}E" for r in cities.itertuples()}


def _components(names: list[str], linked: list[tuple[str, str]]) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in linked:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def classification_metrics(y_true, y_pred, labels=U.CLASS_ORDER) -> dict:
    """Per-class precision/recall/F1/support, macro-F1 (over labels present in truth or prediction),
    accuracy and the confusion matrix (rows = true, columns = predicted)."""
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    k = len(labels)
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        cm[pos[t], pos[p]] += 1
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(axis=0), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / pred_n, 0.0)
        rec = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = (true_n > 0) | (pred_n > 0)
    per_class = pd.DataFrame({"precision": prec, "recall": rec, "f1": f1, "support": true_n}, index=labels)
    return {"per_class": per_class, "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
            "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else float("nan"),
            "confusion": pd.DataFrame(cm, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])}


def _runs(dates: np.ndarray) -> list[int]:
    """Lengths of maximal runs of consecutive calendar days in a sorted unique date array."""
    if len(dates) == 0:
        return []
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    cuts = np.flatnonzero(gaps > 1)
    edges = np.concatenate([[0], cuts + 1, [len(dates)]])
    return np.diff(edges).tolist()


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    zz = np.column_stack([np.ones(len(z)), z])
    rx = x - zz @ np.linalg.lstsq(zz, x, rcond=None)[0]
    ry = y - zz @ np.linalg.lstsq(zz, y, rcond=None)[0]
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 1 + 2: shared grid series and elevation adjustment
# --------------------------------------------------------------------------- #
def analyze_grid_series(df: pd.DataFrame, cities: pd.DataFrame) -> dict:
    cell_of = cell_ids(cities)
    elev_of = cities.set_index("City")["elevation_m"].to_dict()
    names = sorted(df["City"].unique())
    piv = {c: df.pivot(index="Date", columns="City", values=c)[names].to_numpy(float) for c in WEATHER}
    code = {n: i for i, n in enumerate(U.CLASS_ORDER)}
    for lvl in ("HeatLevelToday", "HeatLevelTomorrow"):
        piv[lvl] = df.assign(_c=df[lvl].map(code)).pivot(index="Date", columns="City", values="_c")[names].to_numpy(float)

    rows, ident_cols = [], []
    for i, j in combinations(range(len(names)), 2):
        a, b = names[i], names[j]
        same = cell_of[a] == cell_of[b]
        valid = ~np.isnan(piv[HI][:, i]) & ~np.isnan(piv[HI][:, j])
        nvalid = max(int(valid.sum()), 1)
        ident = {c: float((np.isclose(piv[c][:, i], piv[c][:, j], rtol=0, atol=1e-9) & valid).sum() / nvalid)
                 for c in NONTHERMAL}
        allsame = np.ones(len(valid), bool)
        for c in NONTHERMAL:
            allsame &= np.isclose(piv[c][:, i], piv[c][:, j], rtol=0, atol=1e-9)
        d = lambda c: (piv[c][:, j] - piv[c][:, i])[valid]  # noqa: E731
        dhi = np.abs(d(HI))
        dis = lambda lvl: float((piv[lvl][:, i] != piv[lvl][:, j])[~np.isnan(piv[lvl][:, i]) & ~np.isnan(piv[lvl][:, j])].mean())  # noqa: E731
        rows.append({
            "city_a": a, "city_b": b, "same_cell": same,
            "elev_diff_m": float(elev_of[b] - elev_of[a]),
            "nonthermal_identical_days": float((allsame & valid).sum() / nvalid),
            "mean_dTemp_Max": float(d("Temp_Max").mean()), "mean_dTemp_Mean": float(d("Temp_Mean").mean()),
            "mean_dTemp_Min": float(d("Temp_Min").mean()), "mean_dHumidity_Mean": float(d("Humidity_Mean").mean()),
            "mean_dDewPoint": float(d("DewPoint_Mean").mean()), "mean_dHI": float(d(HI).mean()),
            "mean_abs_dHI": float(dhi.mean()), "p95_abs_dHI": float(np.percentile(dhi, 95)),
            "max_abs_dHI": float(dhi.max()),
            "frac_abs_dHI_ge_0p5": float((dhi >= 0.5).mean()), "frac_abs_dHI_ge_1": float((dhi >= 1.0).mean()),
            "class_disagree_today": dis("HeatLevelToday"), "class_disagree_tomorrow": dis("HeatLevelTomorrow"),
        })
        if same:
            ident_cols.append(ident)
    pair_cols = ["city_a", "city_b", "same_cell", "elev_diff_m", "nonthermal_identical_days", "mean_dTemp_Max",
                 "mean_dTemp_Mean", "mean_dTemp_Min", "mean_dHumidity_Mean", "mean_dDewPoint", "mean_dHI", "mean_abs_dHI",
                 "p95_abs_dHI", "max_abs_dHI", "frac_abs_dHI_ge_0p5", "frac_abs_dHI_ge_1", "class_disagree_today",
                 "class_disagree_tomorrow"]
    pairs = pd.DataFrame(rows, columns=pair_cols)  # explicit columns: a single-city dataset has no pairs
    pairs["same_cell"] = pairs["same_cell"].astype(bool)

    cells = (cities[cities["City"].isin(names)].assign(cell=lambda x: x["City"].map(cell_of))
             .groupby("cell").agg(n_cities=("City", "size"), cities=("City", lambda s: ", ".join(sorted(s))),
                                  elev_min_m=("elevation_m", "min"), elev_max_m=("elevation_m", "max"))
             .sort_values(["n_cities", "cities"], ascending=[False, True]))
    cells["elev_range_m"] = cells["elev_max_m"] - cells["elev_min_m"]

    linked_nt = [(r.city_a, r.city_b) for r in pairs.itertuples() if r.nonthermal_identical_days >= 0.999]
    nt_groups = _components(names, linked_nt)

    # effective number of independent series: participation ratio of deseasonalised HI anomalies
    wide = df.pivot(index="Date", columns="City", values=HI)[names].dropna()
    anom = wide - wide.groupby(wide.index.month).transform("mean")
    n_eff = n95 = float("nan")
    lam1 = float("nan")
    mean_corr = min_corr = float("nan")
    if wide.shape[1] >= 2 and len(anom) > 30 and (anom.std() > 0).all():
        R = np.corrcoef(anom.to_numpy(), rowvar=False)
        lam = np.sort(np.linalg.eigvalsh(R))[::-1].clip(min=0)
        n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
        n95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
        lam1 = float(lam[0] / lam.sum())
        off = R[np.triu_indices_from(R, k=1)]
        mean_corr, min_corr = float(off.mean()), float(off.min())
    elif wide.shape[1] == 1:
        n_eff, n95, lam1 = 1.0, 1, 1.0

    same_pairs, diff_pairs = pairs[pairs["same_cell"]], pairs[~pairs["same_cell"]]
    summ = lambda p, c: float(p[c].mean()) if len(p) else float("nan")  # noqa: E731
    ident_by_col = (pd.DataFrame(ident_cols).mean() if ident_cols else pd.Series(dtype=float)).rename("identical_days_fraction")

    # elevation adjustment: pooled lapse slope over same-cell pairs with a real elevation difference
    use = same_pairs[same_pairs["elev_diff_m"].abs() >= 1.0]
    slopes = {}
    if len(use):
        de = use["elev_diff_m"].to_numpy()
        for label, col in [("Temp_Mean", "mean_dTemp_Mean"), ("Temp_Max", "mean_dTemp_Max"), ("Temp_Min", "mean_dTemp_Min"),
                           ("HeatIndex_Max_Today", "mean_dHI"), ("Humidity_Mean (%)", "mean_dHumidity_Mean"),
                           ("DewPoint_Mean", "mean_dDewPoint")]:
            slopes[label] = float(100.0 * (use[col].to_numpy() * de).sum() / (de ** 2).sum())
    mean_hi, dis_today = summ(same_pairs, "mean_abs_dHI"), summ(same_pairs, "class_disagree_today")
    if not len(same_pairs):
        verdict = "not testable: no two cities share a grid cell"
    elif not len(use):
        verdict = "not testable: cities in a shared cell report identical elevation"
    elif mean_hi >= MATERIAL_HI_DIFF_C or dis_today >= MATERIAL_CLASS_DISAGREE:
        verdict = "material"
    elif mean_hi >= MODEST_HI_DIFF_C or dis_today >= MODEST_CLASS_DISAGREE:
        verdict = "modest"
    else:
        verdict = "negligible"
    return {
        "n_cities": len(names), "n_grid_cells": int(cells.shape[0]), "cells": cells,
        "nonthermal_groups": nt_groups, "n_nonthermal_series": len(nt_groups),
        "effective_series": n_eff, "components_for_95pct": n95, "first_component_share": lam1,
        "mean_anomaly_corr": mean_corr, "min_anomaly_corr": min_corr,
        "pairs": pairs, "n_pairs_same_cell": len(same_pairs), "n_pairs_diff_cell": len(diff_pairs),
        "same_cell_mean_abs_dHI": mean_hi, "diff_cell_mean_abs_dHI": summ(diff_pairs, "mean_abs_dHI"),
        "same_cell_p95_abs_dHI": summ(same_pairs, "p95_abs_dHI"),
        "same_cell_max_abs_dHI": float(same_pairs["max_abs_dHI"].max()) if len(same_pairs) else float("nan"),
        "same_cell_class_disagree_today": dis_today,
        "same_cell_class_disagree_tomorrow": summ(same_pairs, "class_disagree_tomorrow"),
        "diff_cell_class_disagree_today": summ(diff_pairs, "class_disagree_today"),
        "same_cell_frac_dHI_ge_0p5": summ(same_pairs, "frac_abs_dHI_ge_0p5"),
        "same_cell_frac_dHI_ge_1": summ(same_pairs, "frac_abs_dHI_ge_1"),
        "identical_by_column": ident_by_col,
        "elev_min_m": float(cities["elevation_m"].min()), "elev_max_m": float(cities["elevation_m"].max()),
        "max_within_cell_elev_range_m": float(cells["elev_range_m"].max()),
        "lapse_slopes_per_100m": slopes, "n_pairs_with_elev_diff": int(len(use)),
        "elevation_effect": verdict,
        "expected_max_temp_effect_c": abs(STANDARD_LAPSE_C_PER_100M) * float(cells["elev_range_m"].max()) / 100.0,
    }


# --------------------------------------------------------------------------- #
# 3 + 4: class distribution, rare classes, splits
# --------------------------------------------------------------------------- #
def default_test_start(df: pd.DataFrame) -> pd.Timestamp:
    d0, d1 = pd.to_datetime(df["Date"]).min(), pd.to_datetime(df["Date"]).max()
    t = pd.Timestamp("2023-01-01")
    if d0 + pd.Timedelta(days=730) <= t <= d1 - pd.Timedelta(days=90):
        return t
    uniq = np.sort(pd.to_datetime(df["Date"]).unique())
    return pd.Timestamp(uniq[int(len(uniq) * 0.75)])


def _class_counts(frame: pd.DataFrame, cell_of: dict, level: str = U.TARGET) -> pd.DataFrame:
    ev = frame.assign(event_date=pd.to_datetime(frame["Date"]) + pd.Timedelta(days=1), cell=frame["City"].map(cell_of))
    out = []
    for cls in U.CLASS_ORDER:
        sub = ev[ev[level] == cls]
        dates = np.sort(sub["event_date"].unique())
        runs = _runs(dates)
        out.append({"class": cls, "rows": len(sub), "percent_rows": 100.0 * len(sub) / max(len(ev), 1),
                    "cell_days": int(sub.drop_duplicates(["cell", "event_date"]).shape[0]),
                    "distinct_days": len(dates), "episodes": len(runs), "longest_run_days": max(runs) if runs else 0,
                    "years_present": int(pd.DatetimeIndex(dates).year.nunique()) if len(dates) else 0})
    return pd.DataFrame(out).set_index("class")


def analyze_classes(df: pd.DataFrame, cities: pd.DataFrame, test_start=None, embargo_days: int = 1,
                    cv_splits: int = 5, min_train_days: int | None = None) -> dict:
    cell_of = cell_ids(cities)
    overall = _class_counts(df, cell_of)
    ev_year = pd.to_datetime(df["Date"]).dt.year  # forecast-issue year (event day = +1 day)
    per_year = pd.crosstab(ev_year, df[U.TARGET]).reindex(columns=U.CLASS_ORDER, fill_value=0)
    per_year.index.name = "year"
    per_month = pd.crosstab((pd.to_datetime(df["Date"]) + pd.Timedelta(days=1)).dt.month, df[U.TARGET]).reindex(
        columns=U.CLASS_ORDER, fill_value=0)
    per_month.index.name = "month"
    today = (df["HeatLevelToday"].value_counts().reindex(U.CLASS_ORDER, fill_value=0))

    ts = pd.Timestamp(test_start) if test_start is not None else default_test_start(df)
    tr_idx, te_idx = U.chronological_split(df, ts, embargo_days)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    split_tr, split_te = _class_counts(train, cell_of), _class_counts(test, cell_of)
    shift = (test[U.TARGET].value_counts(normalize=True).reindex(U.CLASS_ORDER, fill_value=0) * 100
             - train[U.TARGET].value_counts(normalize=True).reindex(U.CLASS_ORDER, fill_value=0) * 100)

    n_dates = df["Date"].nunique()
    mtd = min_train_days if min_train_days is not None else min(730, n_dates // 3)
    try:
        folds = U.expanding_window_folds(df, n_splits=cv_splits, gap_days=embargo_days, min_train_days=mtd)
    except ValueError:  # too few distinct dates for the requested folds
        folds = []
    fold_rows = []
    for k, (a, b) in enumerate(folds, 1):
        fa, fb = df.iloc[a], df.iloc[b]
        row = {"fold": k, "train_end": str(fa["Date"].max().date()), "test_start": str(fb["Date"].min().date()),
               "test_end": str(fb["Date"].max().date())}
        for cls in U.CLASS_ORDER:
            row[f"train {cls}"] = int((fa[U.TARGET] == cls).sum())
        for cls in U.CLASS_ORDER:
            row[f"test {cls}"] = int((fb[U.TARGET] == cls).sum())
        fold_rows.append(row)
    cv = pd.DataFrame(fold_rows).set_index("fold") if fold_rows else pd.DataFrame()

    flags = []
    for cls in U.CLASS_ORDER:
        o = overall.loc[cls]
        reasons = []
        if o["rows"] == 0:
            level = "ABSENT"
            reasons.append("no observations")
        else:
            level = "ok"
            if o["distinct_days"] < VERY_FEW_DAYS or o["episodes"] < VERY_FEW_EPISODES:
                level = "VERY FEW"
            elif o["distinct_days"] < MIN_DAYS_OVERALL or o["episodes"] < MIN_EPISODES:
                level = "FEW"
            if o["distinct_days"] < MIN_DAYS_OVERALL:
                reasons.append(f"{int(o['distinct_days'])} distinct days (< {MIN_DAYS_OVERALL})")
            if o["episodes"] < MIN_EPISODES:
                reasons.append(f"{int(o['episodes'])} episodes (< {MIN_EPISODES})")
            te = split_te.loc[cls, "distinct_days"]
            if te < MIN_DAYS_TEST:
                level = "FEW" if level == "ok" else level
                reasons.append(f"{int(te)} test-period days (< {MIN_DAYS_TEST})")
            if split_tr.loc[cls, "rows"] == 0:
                level = "ABSENT IN TRAIN"
                reasons.append("absent from the training partition")
            empty_folds = [int(f) for f in cv.index
                           if cv.loc[f, f"test {cls}"] == 0 or cv.loc[f, f"train {cls}"] == 0]
            if empty_folds:
                if level == "ok":
                    level = "FEW"
                reasons.append(f"CV folds {empty_folds} lack this class in train or test")
        flags.append({"class": cls, "status": level, "reasons": "; ".join(reasons) or "-"})
    return {"overall": overall, "today_counts": today, "per_year": per_year, "per_month": per_month,
            "test_start": ts, "embargo_days": embargo_days, "split_train": split_tr, "split_test": split_te,
            "n_train_rows": len(tr_idx), "n_test_rows": len(te_idx), "prevalence_shift_pp": shift,
            "train_range": (str(train["Date"].min().date()), str(train["Date"].max().date())),
            "test_range": (str(test["Date"].min().date()), str(test["Date"].max().date())),
            "cv": cv, "flags": pd.DataFrame(flags).set_index("class")}


def analyze_danger(df: pd.DataFrame, cities: pd.DataFrame) -> dict:
    """Danger / Extreme Danger detail: temporal spread and how many NCR cities reach the class together."""
    out = {}
    n_cities = df["City"].nunique()
    for cls in ("Danger", "Extreme Danger"):
        sub = df[df[U.TARGET] == cls]
        per_day = sub.groupby("Date")["City"].nunique()
        out[cls] = {
            "rows": len(sub), "event_days": int(per_day.size),
            "mean_cities_in_class_on_event_days": float(per_day.mean()) if len(per_day) else float("nan"),
            "share_event_days_all_cities": float((per_day == n_cities).mean()) if len(per_day) else float("nan"),
            "share_event_days_single_city": float((per_day == 1).mean()) if len(per_day) else float("nan"),
            "by_year": sub.groupby(pd.to_datetime(sub["Date"]).dt.year).size(),
            "by_month": sub.groupby((pd.to_datetime(sub["Date"]) + pd.Timedelta(days=1)).dt.month).size(),
        }
    return out


# --------------------------------------------------------------------------- #
# 5: shared series and leakage
# --------------------------------------------------------------------------- #
def analyze_leakage(df: pd.DataFrame, cities: pd.DataFrame, test_start, embargo_days: int = 1, seed: int = 42) -> dict:
    cell_of = cell_ids(cities)
    d = df.assign(cell=df["City"].map(cell_of), Date=pd.to_datetime(df["Date"])).reset_index(drop=True)
    n_cities_in_cell = d.groupby("cell")["City"].transform("nunique")

    nt_dup = float(d.duplicated(["Date"] + NONTHERMAL, keep=False).mean())
    full_dup = float(d.duplicated(["Date"] + WEATHER, keep=False).mean())
    vec_per_date = d.drop_duplicates(["Date"] + NONTHERMAL).groupby("Date").size()

    y = d[U.NUMERIC_TARGET]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_date = float(((d.groupby("Date")[U.NUMERIC_TARGET].transform("mean") - y.mean()) ** 2).sum())
    ss_city = float(((d.groupby("City")[U.NUMERIC_TARGET].transform("mean") - y.mean()) ** 2).sum())
    ss_cell_date = float(((d.groupby(["cell", "Date"])[U.NUMERIC_TARGET].transform("mean") - y.mean()) ** 2).sum())

    # --- partition simulations (assignment only; no model involved)
    def sibling_share(train_mask: np.ndarray, test_mask: np.ndarray) -> tuple[float, float]:
        """Share of test rows that have a same-date row in train (any city / same grid cell)."""
        if test_mask.sum() == 0:
            return float("nan"), float("nan")
        tr_by_date = d.loc[train_mask].groupby("Date").size()
        tr_by_cell_date = d.loc[train_mask].groupby(["cell", "Date"]).size()
        any_city = d.loc[test_mask, "Date"].map(tr_by_date).fillna(0).gt(0).mean()
        idx = pd.MultiIndex.from_frame(d.loc[test_mask, ["cell", "Date"]])
        same_cell = pd.Series(tr_by_cell_date.reindex(idx).fillna(0).to_numpy() > 0).mean()
        return float(any_city), float(same_cell)

    tr_idx, te_idx = U.chronological_split(d, test_start, embargo_days)
    chron_train = np.zeros(len(d), bool)
    chron_train[tr_idx] = True
    chron_test = np.zeros(len(d), bool)
    chron_test[te_idx] = True
    chron_any, chron_cell = sibling_share(chron_train, chron_test)
    # exact duplicate weather vectors across the chronological partitions
    dup_cross = int(d.loc[chron_test, WEATHER].round(6).merge(
        d.loc[chron_train, WEATHER].round(6).drop_duplicates(), on=WEATHER).shape[0])

    rng = np.random.default_rng(seed)
    rand_train = rng.random(len(d)) < 0.8
    rand_any, rand_cell = sibling_share(rand_train, ~rand_train)

    loco_share = float((n_cities_in_cell > 1).mean())  # every held-out city keeps its cell mates in train

    n80 = int(len(df) * 0.8)  # file order is City, Date -> "first 80% of rows" splits by city
    naive_tr, naive_te = df.iloc[:n80], df.iloc[n80:]
    return {
        "same_day_nonthermal_duplicate_share": nt_dup, "same_day_full_weather_duplicate_share": full_dup,
        "distinct_nonthermal_vectors_per_date_mean": float(vec_per_date.mean()),
        "distinct_nonthermal_vectors_per_date_max": int(vec_per_date.max()),
        "eta2_date": ss_date / ss_tot if ss_tot else float("nan"),
        "eta2_cell_date": ss_cell_date / ss_tot if ss_tot else float("nan"),
        "eta2_city": ss_city / ss_tot if ss_tot else float("nan"),
        "chron_test_rows_with_same_date_train_row": chron_any, "chron_test_rows_with_same_cell_date_train_row": chron_cell,
        "chron_exact_duplicate_weather_vectors_across_partitions": dup_cross,
        "random_split_test_rows_with_same_date_train_row": rand_any,
        "random_split_test_rows_with_same_cell_date_train_row": rand_cell,
        "leave_one_city_out_rows_with_same_cell_mate_in_train": loco_share,
        "naive_row_order_split": {
            "train_dates": (str(pd.to_datetime(naive_tr["Date"]).min().date()), str(pd.to_datetime(naive_tr["Date"]).max().date())),
            "test_dates": (str(pd.to_datetime(naive_te["Date"]).min().date()), str(pd.to_datetime(naive_te["Date"]).max().date())),
            "train_cities": int(naive_tr["City"].nunique()), "test_cities": int(naive_te["City"].nunique()),
            "chronological": bool(pd.to_datetime(naive_tr["Date"]).max() <= pd.to_datetime(naive_te["Date"]).min()),
        },
        "static_identifier_combinations": int(df[["Latitude", "Longitude", "Elevation"]].drop_duplicates().shape[0]),
        "n_cities": int(df["City"].nunique()), "n_rows": len(df),
        "n_cell_days": int(d.drop_duplicates(["cell", "Date"]).shape[0]),
    }


# --------------------------------------------------------------------------- #
# 6: lag features
# --------------------------------------------------------------------------- #
def analyze_lags(df: pd.DataFrame, max_pacf: int = 7) -> dict:
    lagged = U.add_lag_features(df)
    names = U.lag_feature_names()
    doy = 2 * np.pi * lagged["DayOfYear"].to_numpy() / 365.25
    ctrl = np.column_stack([lagged[HI].to_numpy(), np.sin(doy), np.cos(doy)])
    y_all = lagged[U.NUMERIC_TARGET].to_numpy()
    rows = []
    for name in names:
        x = lagged[name].to_numpy()
        ok = ~np.isnan(x) & ~np.isnan(y_all)
        if ok.sum() < 50 or np.nanstd(x[ok]) == 0:
            continue
        r = float(np.corrcoef(x[ok], y_all[ok])[0, 1])
        pr = _partial_corr(x[ok], y_all[ok], ctrl[ok])
        rows.append({"feature": name, "n_rows": int(ok.sum()), "corr_with_target": r, "partial_corr": pr})
    table = pd.DataFrame(rows, columns=["feature", "n_rows", "corr_with_target", "partial_corr"]).set_index("feature")
    table["useful"] = table["partial_corr"].abs() >= USEFUL_PARTIAL_R
    table = table.reindex(table["partial_corr"].abs().sort_values(ascending=False).index)

    wide = df.pivot(index="Date", columns="City", values=HI)
    series = wide.mean(axis=1).dropna()
    anom = (series - series.groupby(series.index.month).transform("mean")).to_numpy()
    pacf = {}
    for p in range(1, max_pacf + 1):
        if len(anom) <= p + 30:
            break
        X = np.column_stack([anom[p - k - 1: len(anom) - k - 1] for k in range(p)])
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), anom[p:], rcond=None)[0]
        pacf[p] = float(coef[-1])
    ac1 = float(np.corrcoef(anom[1:], anom[:-1])[0, 1]) if len(anom) > 31 else float("nan")
    raw1 = float(np.corrcoef(series.to_numpy()[1:], series.to_numpy()[:-1])[0, 1]) if len(series) > 31 else float("nan")
    return {"table": table, "pacf": pacf, "anomaly_autocorr_lag1": ac1, "raw_autocorr_lag1": raw1}


# --------------------------------------------------------------------------- #
# 7: reference baselines (rules, not trained models)
# --------------------------------------------------------------------------- #
def analyze_baselines(df: pd.DataFrame, test_start, embargo_days: int = 1) -> dict:
    tr_idx, te_idx = U.chronological_split(df, test_start, embargo_days)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    out = {}
    out["persistence (tomorrow = today's class)"] = classification_metrics(test[U.TARGET], test["HeatLevelToday"])
    major = train[U.TARGET].value_counts().idxmax()
    out[f"majority class of train ('{major}')"] = classification_metrics(test[U.TARGET], [major] * len(test))
    month_major = train.assign(m=(pd.to_datetime(train["Date"]) + pd.Timedelta(days=1)).dt.month).groupby("m")[U.TARGET].agg(
        lambda s: s.value_counts().idxmax())
    tm = (pd.to_datetime(test["Date"]) + pd.Timedelta(days=1)).dt.month
    out["climatology (train majority class of the target month)"] = classification_metrics(
        test[U.TARGET], tm.map(month_major).fillna(major))
    return out


# --------------------------------------------------------------------------- #
# 8: recommendations + orchestration
# --------------------------------------------------------------------------- #
def build_recommendations(r: dict) -> list[str]:
    g, c, lk = r["grid"], r["classes"], r["leakage"]
    rec = []
    rec.append(
        f"**Treat the data as ~{g['effective_series']:.1f} effective independent weather series, not {g['n_cities']}.** "
        f"The {g['n_cities']} cities resolve to {g['n_grid_cells']} ERA5 grid cell(s); report the sample size honestly "
        f"({lk['n_cell_days']:,} grid-cell days vs {lk['n_rows']:,} city rows) and do not describe the cities as "
        "independent stations.") if not np.isnan(g["effective_series"]) else rec.append(
        "Treat the cities as strongly dependent; the effective number of independent series could not be estimated.")
    rec.append(
        "**Split by `Date`, never by row.** The CSV is sorted City-then-Date, so `train_test_split(shuffle=False)` or "
        "`TimeSeriesSplit` on the raw frame splits by city"
        + (" (confirmed: the first 80 % of rows is not chronological)" if not lk["naive_row_order_split"]["chronological"] else "")
        + ". Use `ncr_modeling_utils.chronological_split` / `expanding_window_folds` (all cities of a date stay together).")
    rec.append(
        "**Keep a 1-3 day embargo between train and test.** For a 1-day horizon the overlap is only between the last "
        "training labels and the first test *inputs* (no test target is exposed), so this is cheap insurance rather than "
        "a hard requirement - but it becomes necessary once rolling features are used.")
    if g["n_pairs_same_cell"]:
        rec.append(
            f"**Do not use random row splits or leave-one-city-out.** A random 80/20 split gives "
            f"{lk['random_split_test_rows_with_same_cell_date_train_row']:.0%} of test rows a same-date, same-grid-cell twin in training; "
            f"leave-one-city-out gives {lk['leave_one_city_out_rows_with_same_cell_mate_in_train']:.0%}. "
            "Both leak almost-identical weather into training. A chronological split gives "
            f"{lk['chron_test_rows_with_same_date_train_row']:.0%} by construction.")
    else:
        rec.append(
            "**Do not use random row splits.** No two cities share a grid cell in this data, so the twin-leakage figures are 0 %, "
            "but a random split still mixes neighbouring days of one autocorrelated series; split by date.")
    rec.append(
        f"**Treat `City`, `Latitude`, `Longitude`, `Elevation` as identifiers.** They take {lk['static_identifier_combinations']} distinct "
        f"combinations; city identity explains {lk['eta2_city']:.1%} of target variance and the date alone explains {lk['eta2_date']:.1%}. "
        "Fit with and without them (or use grid-cell id) and prefer the simpler model if scores are equal; tree models can "
        "otherwise memorise city/cell identity.")
    eff = g["elevation_effect"]
    if eff in ("negligible", "modest", "material"):
        rec.append(
            f"**Elevation adjustment is {eff}** (mean |ΔHI| between cities in the same cell {g['same_cell_mean_abs_dHI']:.2f} °C; "
            f"{g['same_cell_class_disagree_today']:.1%} of same-cell city-days land in different classes). "
            + ("Cities in one cell are near-duplicates; evaluate on one row per grid-cell day, or weight rows by 1/(cities in cell), as a sensitivity check."
               if eff != "material" else "Cities differ enough within a cell that city-level labels carry some distinct information."))
    flagged = c["flags"][c["flags"]["status"] != "ok"]
    if len(flagged):
        rec.append("**Rare classes need a decision from you (nothing was merged or dropped):** "
                   + "; ".join(f"{k}: {v['status']} ({v['reasons']})" for k, v in flagged.iterrows())
                   + ". Options that keep the 5-class task: class-weighted training, and per-class metrics with date-block bootstrap "
                     "confidence intervals reported alongside Macro F1; merging classes would be a separate, explicit decision.")
    rec.append(
        "**Get uncertainty from dates, not rows.** Use a date-block bootstrap (resample days or weeks, keep all cities of a day "
        "together); row-level standard errors are far too small because cities repeat the same weather and consecutive days "
        "are autocorrelated.")
    rec.append(
        "**Expect distribution shift across years.** Compare the per-year class table: heat years (e.g. 2015-16, 2023-24) are "
        "not exchangeable with cool years, so report per-fold and per-year scores, not only one pooled number.")
    useful = r["lags"]["table"][r["lags"]["table"]["useful"]]
    rec.append(
        "**Lag features (past only):** candidates with |partial r| >= "
        f"{USEFUL_PARTIAL_R} beyond today's heat index and season: "
        + (", ".join(f"`{i}`" for i in useful.index[:8]) if len(useful) else "none cleared the threshold")
        + ". Linear partial correlation understates nonlinear value for tree models, so confirm with time-aware CV; keep any "
          "feature set that does not beat today's-values-only on Macro F1 out.")
    rec.append(
        "**Always compare against the reference baselines below** (persistence is a strong baseline for next-day heat). "
        "Add ordinal-aware diagnostics (e.g. the confusion matrix, errors of one class vs several classes) because the five "
        "classes are ordered.")
    rec.append("`HeatIndex_Max_Tomorrow` and `HeatLevelTomorrow` must never enter the feature list: "
               "use `ncr_modeling_utils.assert_no_future_information`. `HeatLevelToday` is a binning of `HeatIndex_Max_Today` (redundant).")
    return rec


def run_analysis(df: pd.DataFrame, cities: pd.DataFrame, test_start=None, embargo_days: int = 1, cv_splits: int = 5,
                 seed: int = 42) -> dict:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["City", "Date"], kind="mergesort").reset_index(drop=True)
    cities = cities[cities["City"].isin(df["City"].unique())].reset_index(drop=True)
    U.assert_no_future_information(U.INPUT_FEATURES)
    res: dict = {"n_rows": len(df), "n_cities": int(df["City"].nunique()),
                 "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date())}
    res["grid"] = analyze_grid_series(df, cities)
    res["classes"] = analyze_classes(df, cities, test_start, embargo_days, cv_splits)
    res["danger"] = analyze_danger(df, cities)
    res["leakage"] = analyze_leakage(df, cities, res["classes"]["test_start"], embargo_days, seed)
    res["lags"] = analyze_lags(df)
    res["baselines"] = analyze_baselines(df, res["classes"]["test_start"], embargo_days)
    res["recommendations"] = build_recommendations(res)
    return res


def key_findings(r: dict) -> list[str]:
    g, c, lk = r["grid"], r["classes"], r["leakage"]
    lines = [
        f"{r['n_cities']} cities -> {g['n_grid_cells']} ERA5 grid cell(s); {g['n_nonthermal_series']} distinct non-thermal series; "
        f"effective independent series ~{g['effective_series']:.2f} (95% of anomaly variance needs {g['components_for_95pct']} component(s)).",
        f"Elevation adjustment: {g['elevation_effect']} (API elevations {g['elev_min_m']:.0f}-{g['elev_max_m']:.0f} m; "
        f"same-cell mean |dHI| {g['same_cell_mean_abs_dHI']:.2f} C; class disagreement {g['same_cell_class_disagree_today']:.1%}).",
        "Class counts (HeatLevelTomorrow rows | distinct days | episodes | status): " + "; ".join(
            f"{k}: {int(v['rows']):,} ({v['percent_rows']:.2f}%) | {int(v['distinct_days']):,} | {int(v['episodes'])} | {c['flags'].loc[k, 'status']}"
            for k, v in c["overall"].iterrows()),
        f"Leakage: random split -> {lk['random_split_test_rows_with_same_cell_date_train_row']:.0%} of test rows have a same-day same-cell twin in train; "
        f"chronological -> {lk['chron_test_rows_with_same_date_train_row']:.0%}.",
    ]
    return lines


def render_markdown(r: dict) -> str:
    g, c, lk, dg = r["grid"], r["classes"], r["leakage"], r["danger"]
    p = []
    p.append("# HeatCast NCR - pre-modelling data analysis\n")
    p.append(f"Dataset: {r['n_rows']:,} rows, {r['n_cities']} NCR cities, {r['date_min']} to {r['date_max']}. "
             "Geographic scope is NCR / Metro Manila only. **No model has been trained**; everything below is counting, "
             "correlation and rule-based baselines computed from the data.\n")
    p.append("## Key findings\n")
    p.extend(f"- {x}" for x in key_findings(r))
    p.append("")
    p.append("## 1. How many distinct grid series are there?\n")
    p.append(f"The data are **gridded ERA5 reanalysis estimates for representative city coordinates** (population-weighted centres), "
             f"not {g['n_cities']} independent weather stations. The {g['n_cities']} coordinates resolve to **{g['n_grid_cells']} Open-Meteo/ERA5 "
             f"grid cell(s)** (grid centre returned by the API):\n")
    p.append(_md_table(g["cells"].reset_index().rename(columns={"cell": "grid cell"}), index=False))
    p.append(f"Measured from the data themselves: {g['n_nonthermal_series']} group(s) of cities have identical non-thermal weather "
             "(pressure, cloud, wind, rain, solar) on at least 99.9 % of days: "
             + "; ".join("{" + ", ".join(grp) + "}" for grp in g["nonthermal_groups"]) + ".\n")
    p.append("Share of days on which two cities **in the same grid cell** report exactly the same value:\n")
    p.append(_md_table(g["identical_by_column"].to_frame().reset_index().rename(columns={"index": "column"}), "{:.4f}", index=False)
             if len(g["identical_by_column"]) else "(no two cities share a cell)\n")
    p.append(f"Effective number of independent series (participation ratio of deseasonalised daily heat-index anomalies): "
             f"**{g['effective_series']:.2f}**; first principal component explains {g['first_component_share']:.1%}; "
             f"{g['components_for_95pct']} component(s) reach 95 %. Mean / minimum pairwise anomaly correlation "
             f"{g['mean_anomaly_corr']:.3f} / {g['min_anomaly_corr']:.3f}.\n")
    p.append("## 2. Does Open-Meteo's elevation adjustment separate cities?\n")
    p.append(f"API elevations span {g['elev_min_m']:.0f}-{g['elev_max_m']:.0f} m; the largest elevation range inside one grid cell is "
             f"{g['max_within_cell_elev_range_m']:.0f} m (a standard lapse rate would predict at most "
             f"{g['expected_max_temp_effect_c']:.2f} °C temperature difference from that).\n")
    if g["lapse_slopes_per_100m"]:
        p.append(f"Pooled within-cell slopes per +100 m elevation ({g['n_pairs_with_elev_diff']} city pair(s); standard lapse "
                 f"{STANDARD_LAPSE_C_PER_100M} °C/100 m):\n")
        p.append(_md_table(pd.Series(g["lapse_slopes_per_100m"], name="slope per +100 m").to_frame().reset_index().rename(
            columns={"index": "variable"}), "{:.3f}", index=False))
    p.append(f"Between cities in the **same** cell: mean |ΔHI| {g['same_cell_mean_abs_dHI']:.3f} °C (95th pct {g['same_cell_p95_abs_dHI']:.2f}, "
             f"max {g['same_cell_max_abs_dHI']:.2f}); days with |ΔHI| >= 0.5 °C: {g['same_cell_frac_dHI_ge_0p5']:.1%}, >= 1 °C: "
             f"{g['same_cell_frac_dHI_ge_1']:.1%}; today's heat level differs on {g['same_cell_class_disagree_today']:.2%} of days "
             f"(tomorrow's: {g['same_cell_class_disagree_tomorrow']:.2%}). Between cities in **different** cells: mean |ΔHI| "
             f"{g['diff_cell_mean_abs_dHI']:.3f} °C, class disagreement {g['diff_cell_class_disagree_today']:.2%}.\n")
    p.append(f"**Verdict (heuristic: material if mean |ΔHI| >= {MATERIAL_HI_DIFF_C} °C or class disagreement >= {MATERIAL_CLASS_DISAGREE:.0%}; "
             f"modest if >= {MODEST_HI_DIFF_C} °C or >= {MODEST_CLASS_DISAGREE:.1%}): elevation adjustment is {g['elevation_effect']}.**\n")
    p.append("## 3. Class counts and percentages (target `HeatLevelTomorrow`)\n")
    p.append("Rows are city-days. *Distinct days* collapses the cities (a day counts once if any city is in the class, dated by the "
             "day the heat occurs); *cell-days* counts each grid cell once per day; *episodes* are runs of consecutive such days "
             "- the closest thing to an independent event count.\n")
    ov = c["overall"].reset_index().rename(columns={"class": "class"})
    p.append(_md_table(ov, index=False))
    p.append("Reference - `HeatLevelToday` rows: " + ", ".join(f"{k} {int(v):,}" for k, v in c["today_counts"].items()) + ".\n")
    p.append("Rows per year (forecast-issue year):\n")
    p.append(_md_table(c["per_year"]))
    p.append("## 4. Danger and Extreme Danger\n")
    for cls, d in dg.items():
        if d["rows"] == 0:
            p.append(f"- **{cls}: no observations.**")
            continue
        p.append(f"- **{cls}**: {d['rows']:,} city-day rows on {d['event_days']:,} distinct days; on those days on average "
                 f"{d['mean_cities_in_class_on_event_days']:.1f} of {r['n_cities']} cities are in the class; all cities together on "
                 f"{d['share_event_days_all_cities']:.0%} of the days, a single city on {d['share_event_days_single_city']:.0%}. "
                 f"By year: " + ", ".join(f"{int(y)}: {int(n):,}" for y, n in d["by_year"].items())
                 + ". By month of the heat day: " + ", ".join(f"{int(m)}: {int(n):,}" for m, n in d["by_month"].items()) + ".")
    p.append("")
    p.append(f"### Reliability check (flags only - no class is merged or dropped)\n")
    p.append(f"Heuristic thresholds: fewer than {MIN_DAYS_OVERALL} distinct days or {MIN_EPISODES} episodes = FEW; fewer than "
             f"{VERY_FEW_DAYS} days or {VERY_FEW_EPISODES} episodes = VERY FEW; fewer than {MIN_DAYS_TEST} days in the test partition, "
             "or a class missing from train/test in any CV fold, also raises FEW.\n")
    p.append(_md_table(c["flags"].reset_index(), index=False))
    p.append(f"### Proposed chronological split (embargo {c['embargo_days']} day(s))\n")
    p.append(f"Train {c['train_range'][0]} to {c['train_range'][1]} ({c['n_train_rows']:,} rows); test {c['test_range'][0]} to "
             f"{c['test_range'][1]} ({c['n_test_rows']:,} rows), test starts {c['test_start'].date()}.\n")
    both = pd.concat({"train rows": c["split_train"]["rows"], "train days": c["split_train"]["distinct_days"],
                      "test rows": c["split_test"]["rows"], "test days": c["split_test"]["distinct_days"],
                      "test minus train (pp)": c["prevalence_shift_pp"]}, axis=1)
    p.append(_md_table(both.reset_index().rename(columns={"index": "class"}), index=False))
    p.append(f"### Expanding-window CV folds (gap {c['embargo_days']} day(s))\n")
    p.append(_md_table(c["cv"]) if len(c["cv"]) else "(too few distinct dates to form CV folds)\n")
    p.append("## 5. Shared grid series and leakage\n")
    p.append(f"- {lk['n_rows']:,} city rows correspond to only {lk['n_cell_days']:,} (grid cell, day) combinations.")
    p.append(f"- On {lk['same_day_nonthermal_duplicate_share']:.1%} of city-days another city has identical non-thermal weather the same day "
             f"(identical full weather incl. temperature/humidity: {lk['same_day_full_weather_duplicate_share']:.1%}); distinct non-thermal "
             f"vectors per day: mean {lk['distinct_nonthermal_vectors_per_date_mean']:.2f}, max {lk['distinct_nonthermal_vectors_per_date_max']}.")
    p.append(f"- Variance of `HeatIndex_Max_Tomorrow` explained by date alone: {lk['eta2_date']:.1%}; by (grid cell, date): "
             f"{lk['eta2_cell_date']:.1%}; by city identity alone: {lk['eta2_city']:.2%}. "
             f"The {lk['static_identifier_combinations']} (Latitude, Longitude, Elevation) combinations are identifiers, not independent information.")
    p.append("- Partition simulations (assignment only, no model): share of **test rows that have a same-day twin in training**:")
    p.append(f"  - chronological split: any city {lk['chron_test_rows_with_same_date_train_row']:.1%}, same cell "
             f"{lk['chron_test_rows_with_same_cell_date_train_row']:.1%}; exact duplicate weather vectors across partitions: "
             f"{lk['chron_exact_duplicate_weather_vectors_across_partitions']:,}.")
    p.append(f"  - random 80/20 row split: any city {lk['random_split_test_rows_with_same_date_train_row']:.1%}, same cell "
             f"{lk['random_split_test_rows_with_same_cell_date_train_row']:.1%}  -> **invalid**.")
    p.append(f"  - leave-one-city-out: {lk['leave_one_city_out_rows_with_same_cell_mate_in_train']:.1%} of held-out rows keep a same-cell "
             "mate in training -> **invalid** as a generalisation test.")
    nv = lk["naive_row_order_split"]
    p.append(f"  - naive 'first 80 % of rows' of the saved file (sorted by City, Date): train {nv['train_dates'][0]}..{nv['train_dates'][1]} "
             f"({nv['train_cities']} cities), test {nv['test_dates'][0]}..{nv['test_dates'][1]} ({nv['test_cities']} cities) -> "
             f"{'chronological' if nv['chronological'] else '**NOT chronological: it splits by city**'}.")
    p.append("")
    p.append("## 6. Lag features (1-3 days, past information only)\n")
    p.append(f"Autocorrelation of the daily mean heat index, lag 1: raw {r['lags']['raw_autocorr_lag1']:.3f}, seasonally adjusted "
             f"{r['lags']['anomaly_autocorr_lag1']:.3f}. Partial autocorrelation of the adjusted series: "
             + ", ".join(f"lag {k}: {v:.3f}" for k, v in r["lags"]["pacf"].items()) + ".\n")
    p.append(f"Correlation with `{U.NUMERIC_TARGET}` and partial correlation after controlling for `{HI}` and season "
             f"(useful = |partial r| >= {USEFUL_PARTIAL_R}, a linear heuristic):\n")
    p.append(_md_table(r["lags"]["table"].reset_index(), "{:.3f}", index=False))
    p.append("Each city's first three days have no lags (NaN); build them with `ncr_modeling_utils.add_lag_features`, which is "
             "calendar-aware and strictly past-only.\n")
    p.append("## 7. Reference baselines (rules, not trained models) on the proposed test period\n")
    for name, m in r["baselines"].items():
        p.append(f"### {name}\n")
        p.append(f"Macro F1 **{m['macro_f1']:.3f}**, accuracy {m['accuracy']:.3f}.\n")
        p.append(_md_table(m["per_class"].reset_index().rename(columns={"index": "class"}), "{:.3f}", index=False))
        if name.startswith("persistence"):
            p.append(_md_table(m["confusion"].reset_index().rename(columns={"index": ""}), index=False))
    p.append("## 8. Recommendations (NCR scope retained)\n")
    p.extend(f"{i}. {t}" for i, t in enumerate(r["recommendations"], 1))
    p.append("")
    return "\n".join(p) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--city-list", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--test-start", default=None, help="first test date (default 2023-01-01 when the data allow)")
    ap.add_argument("--embargo-days", type=int, default=1)
    ap.add_argument("--cv-splits", type=int, default=5)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.dataset, parse_dates=["Date"])
    cities = pd.read_csv(a.city_list, dtype={"city_id": str})
    res = run_analysis(df, cities, a.test_start, a.embargo_days, a.cv_splits)
    out = Path(a.out_dir) / "ncr_dataset_analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(res), encoding="utf-8")
    print("\n".join(key_findings(res)))
    print(f"\nFull report: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
