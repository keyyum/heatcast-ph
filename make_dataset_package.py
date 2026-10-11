"""Dataset package for the HeatCast NCR submission (deliverable 4: dataset package).

Two jobs:

* ``processed_dictionary()`` documents every column of ``data/ncr/processed/ncr_model_ready.csv.gz`` (the 26 columns
  of the original dataset plus the 46 added by ``ncr_preprocessing.py``). It is built from the same constants the
  preprocessing uses, so it cannot drift from the code. ``python make_dataset_package.py`` rewrites
  ``data/ncr/processed/ncr_model_ready_data_dictionary.csv``.
* ``build_package(out_dir)`` assembles the folder layout the course asks for (``data/original`` and
  ``data/processed``) with plain-CSV copies, the dictionaries, the source and licence note, the processing notes and
  SHA-256 checksums. The output goes to ``out/dataset_package/`` (ignored by git) and is meant for the submission ZIP.

The repository keeps its own layout (``data/ncr/...``) because the code and the tests read from there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

import ncr_modeling_utils as U
import ncr_preprocessing as P

ROOT = Path(__file__).resolve().parent
NCR = ROOT / "data" / "ncr"
ORIGINAL_DICT = NCR / "output" / "ph_heat_index_data_dictionary.csv"
PROCESSED_GZ = NCR / "processed" / "ncr_model_ready.csv.gz"
PROCESSED_DICT = NCR / "processed" / "ncr_model_ready_data_dictionary.csv"
SPLIT_JSON = NCR / "processed" / "split_definition.json"
SOURCES_NOTE = NCR / "SOURCES_AND_LICENSES.md"
DEFAULT_OUT = ROOT / "out" / "dataset_package"

_LAG = re.compile(r"^(?P<src>.+)_(?:lag(?P<k>\d+)|mean(?P<w>\d+)d|(?P<chg>change1d))$")


def _dtype_name(dtype) -> str:
    s = str(dtype)
    return s if s in ("float64", "int64") else "string"


def processed_dictionary() -> pd.DataFrame:
    """One row per column of ncr_model_ready.csv.gz, in file order, in the format of the original dictionary."""
    original = pd.read_csv(ORIGINAL_DICT, keep_default_na=False).set_index("column")
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    sample = pd.read_csv(PROCESSED_GZ, nrows=200)
    rows = []
    for col in sample.columns:
        dtype = _dtype_name(sample[col].dtype)
        if col in original.index:
            r = original.loc[col].to_dict()
            if col == "Date":
                dtype = "date"
            rows.append({"column": col, "dtype": dtype, **{k: r[k] for k in
                         ("role", "unit", "description", "derivation", "allowed_values")}})
            continue
        m = _LAG.match(col)
        if m:
            src = m["src"]
            unit = original.loc[src, "unit"]
            if m["k"]:
                k = int(m["k"])
                role, desc = "feature (past day, derived)", f"{src} {k} day{'s' if k > 1 else ''} before Date, same city."
                deriv = f"{src} of the row with Date - {k} day(s) in the same City (exact calendar day)"
            elif m["w"]:
                w = int(m["w"])
                role, desc = "feature (today and past days, derived)", f"Mean of {src} over Date and the {w - 1} days before it, same city."
                deriv = f"mean of {src} over Date - {w - 1} day(s) .. Date in the same City"
            else:
                role, desc = "feature (day-to-day change, derived)", f"Change in {src} from the day before Date to Date, same city."
                deriv = f"{src} on Date minus {src} on Date - 1 day in the same City"
            rows.append({"column": col, "dtype": dtype, "role": role, "unit": unit, "description": desc,
                         "derivation": deriv + "; ncr_preprocessing.py / ncr_modeling_utils.add_lag_features", "allowed_values": ""})
            continue
        if col == P.LEVEL_CODE:
            rows.append({"column": col, "dtype": dtype, "role": "feature (same day, derived, optional)", "unit": "code 0-4",
                         "description": "HeatLevelToday as an ordered integer code. Off by default in the models (a binning of HeatIndex_Max_Today).",
                         "derivation": "HeatLevelToday mapped with ncr_preprocessing.LEVEL_CODES",
                         "allowed_values": " | ".join(f"{v} = {k}" for k, v in P.LEVEL_CODES.items())})
            continue
        if col == "Split":
            tr, te = split["train_dates"], split["test_dates"]
            rows.append({"column": col, "dtype": dtype, "role": "partition", "unit": "category",
                         "description": f"Chronological 80/20 partition by date. train: {tr[0]} to {tr[1]}; test: {te[0]} to {te[1]}. "
                                        f"The {split['gap_days']}-day gap between them is not in the file. The test rows must stay unused until the final evaluation.",
                         "derivation": "ncr_preprocessing.prepare_dataset (see split_definition.json)", "allowed_values": "train | test"})
            continue
        raise ValueError(f"column {col!r} of the processed file has no dictionary entry; add it to processed_dictionary()")
    out = pd.DataFrame(rows, columns=list(original.reset_index().columns))
    U.assert_no_future_information([c for c in out["column"] if out.set_index("column").loc[c, "role"].startswith("feature")])
    return out


def write_processed_dictionary(path: Path = PROCESSED_DICT) -> Path:
    processed_dictionary().to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_package(out_dir: Path = DEFAULT_OUT, with_raw: bool = False) -> Path:
    """Assemble the dataset package in the course's folder layout. Returns the package folder."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    orig, proc = out_dir / "data" / "original", out_dir / "data" / "processed"
    orig.mkdir(parents=True)
    proc.mkdir(parents=True)

    for name in ("ph_heat_index_next_day.csv", "ph_heat_index_data_dictionary.csv", "ph_heat_index_city_list.csv",
                 "dataset_documentation.md", "ncr_dataset_analysis.md"):
        shutil.copy2(NCR / "output" / name, orig / name)
    shutil.copy2(SOURCES_NOTE, orig / SOURCES_NOTE.name)
    if with_raw:
        raw = NCR / "cache" / "raw"
        if not any(raw.rglob("*.json.gz")):
            raise FileNotFoundError(f"{raw} has no raw API responses in this checkout")
        shutil.copytree(raw, orig / "raw_api_responses")

    frame = pd.read_csv(PROCESSED_GZ, keep_default_na=False, dtype=str)       # as text: no re-formatting of numbers
    frame.to_csv(proc / "ncr_model_ready.csv", index=False, lineterminator="\n")
    back = pd.read_csv(proc / "ncr_model_ready.csv", keep_default_na=False, dtype=str)
    if not back.equals(frame):
        raise RuntimeError("plain CSV copy does not match ncr_model_ready.csv.gz")
    processed_dictionary().to_csv(proc / PROCESSED_DICT.name, index=False, lineterminator="\n", encoding="utf-8")
    for path in (SPLIT_JSON, NCR / "processed" / "preprocessing_report.md"):
        shutil.copy2(path, proc / path.name)

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    lines = [
        "# HeatCast NCR - dataset package",
        "",
        "Deliverable 4 of the course project. Built by `python make_dataset_package.py` from the project repository.",
        "",
        "- `data/original/`: the dataset as built from the Open-Meteo API (`ph_heat_index_next_day.csv`, 64,240 rows x 26 columns), "
        "its data dictionary, the city list, the build documentation, the pre-modelling analysis and the source and licence note "
        "(`SOURCES_AND_LICENSES.md`)." + (" `raw_api_responses/` holds the 176 hourly API responses (gzip JSON) the CSV is built from." if with_raw else
                                           " The 176 raw API responses are in the repository under `data/ncr/cache/raw/`."),
        f"- `data/processed/`: the model-ready data (`ncr_model_ready.csv`, {len(frame):,} rows x {frame.shape[1]} columns), its data dictionary, "
        "the train/test split definition and the preprocessing report (processing notes).",
        "",
        "In the repository these files live under `data/ncr/output/` and `data/ncr/processed/`; the processed file is stored there "
        "gzipped (`ncr_model_ready.csv.gz`). This package holds a plain CSV copy with identical content.",
        "",
        "## Files and SHA-256 checksums",
        "",
        "| File | Size | SHA-256 |",
        "|---|---|---|",
    ]
    lines += [f"| `{p.relative_to(out_dir).as_posix()}` | {p.stat().st_size:,} B | `{_sha256(p)}` |" for p in files]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--with-raw", action="store_true", help="also copy the 176 raw API responses into data/original/")
    ap.add_argument("--dictionary-only", action="store_true", help="only rewrite the processed data dictionary")
    a = ap.parse_args(argv)
    print("wrote", write_processed_dictionary())
    if not a.dictionary_only:
        print("package ->", build_package(Path(a.out_dir), a.with_raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
