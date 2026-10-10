# HeatCast NCR

**Next-Day Maximum Heat Index Risk Classification in Metro Manila Using Traditional Machine Learning**

This repository builds and checks the dataset for that project. **Scope is NCR / Metro Manila only** (the 16 NCR cities
of the PSA PSGC list) and is not meant to be extended automatically. The ML task is **classification** of
`HeatLevelTomorrow`; Logistic Regression, Random Forest and Gradient Boosting are planned, with Macro F1 as the
likely primary metric. **No model is trained here.**

> **Status:** the real NCR dataset has been built (2026-10-07, Google Colab, Open-Meteo / ERA5), validated, and committed
> with its raw API responses under [`data/ncr/`](data/ncr/README.md): 64,240 rows = 16 cities × 4,015 days, no missing values,
> duplicates or gaps, rebuildable byte-for-byte offline. No model has been trained. **Two classes are unusable as they stand:
> Extreme Danger never occurs (the highest heat index is 48.16 °C) and Not Hazardous occurs on only 16 days.** Nothing has been merged
> or dropped; see `data/ncr/output/ncr_dataset_analysis.md` and the decisions listed in `data/ncr/README.md`. The test suite
> (`tests/`) runs against a synthetic mock of the API plus checks on the committed real data.

## What the data are (and are not)

Gridded **ERA5 reanalysis estimates** from the Open-Meteo Historical Weather API, taken at one representative coordinate per
city (the population-weighted centre of its barangays). They are **not station observations and not 16 independent weather
stations**: Metro Manila's 16 coordinates fall into only about two ~25 km ERA5 grid cells, so nearby cities share the same
underlying meteorology. That is accepted by design. The build **measures** it instead of ignoring it (see
`ncr_dataset_analysis.md`).

The 16 cities: Caloocan, Las Piñas, Makati, Malabon, Mandaluyong, Manila, Marikina, Muntinlupa, Navotas, Parañaque, Pasay,
Pasig, Quezon City, San Juan, Taguig, Valenzuela (Pateros is a municipality, not a city, and is not in the PSGC city list).

## Run it

```bash
pip install -r requirements.txt
python build_ph_heat_index_dataset.py --work-dir ph_heat_index_cache --out-dir out     # NCR is the default scope
```

or open `ph_heat_index_colab.ipynb` in Colab (cache + outputs on Google Drive). Re-running resumes from the cache. In an
environment with an outbound allow-list, allow `archive-api.open-meteo.com`, `pypi.org` and `files.pythonhosted.org`.

**"PAUSE ... budget" in the output is not a hang.** The script caps its own API usage just under Open-Meteo's free limits
(570/min, 4,900/h, 9,800/day weighted calls; server limits 600 / 5,000 / 10,000). The whole NCR job (~4,590 calls) fits inside one
hourly window; if you ever hit a cap, the message says how long it waits and how to skip it (`--hour-budget 4900`). Progress
is cached, so interrupting and re-running is always safe. (`api_usage_log.json` in the cache folder records recent calls;
delete it only if you know the server quota has reset.)

Useful flags: `--max-cities 3` (smoke test, labelled SUBSET), `--validate-only FILE.csv`, `--no-keep-raw`, `--model`
(default `era5`), `--region` (default `NCR`; `all` exists but is outside this project's scope). `--help` lists the rest.

### Outputs (written to `--out-dir`)

| File | Content |
|---|---|
| `ph_heat_index_next_day.csv` | the dataset (26 columns, sorted by City, Date; ~64,000 rows = 16 cities × 4,015 days) |
| `ph_heat_index_data_dictionary.csv` | column, dtype, role, unit, description, derivation |
| `ph_heat_index_city_list.csv` | the 16 cities, coordinates, API elevation, ERA5 grid cell, provenance |
| `dataset_documentation.md` | source, endpoint, method, formula, categories, limitations, validation of *that* build |
| `ncr_dataset_analysis.md` | **pre-modelling analysis** (below), computed from the real data |

## The pre-modelling analysis (`ncr_dataset_analysis.md`)

Generated automatically after the dataset passes validation; also runnable alone
(`python ncr_dataset_analysis.py --dataset ... --city-list ...`). Model-free: counting, correlation and rule-based baselines.

1. **Grid series:** how many distinct ERA5 cells / non-thermal series / effective independent series (PCA participation ratio).
2. **Elevation adjustment:** within-cell lapse-rate slopes, heat-index differences between cities in one cell, how often
   they land in different classes, and a heuristic *negligible / modest / material* verdict.
3. **Class counts and percentages** for `HeatLevelTomorrow`, per year, plus distinct days and heat *episodes*.
4. **Danger and Extreme Danger** detail, and a **sufficiency flag per class** (FEW / VERY FEW / ABSENT / ABSENT IN TRAIN) on the
   whole data, the proposed chronological split and every CV fold. **Nothing is merged or dropped automatically.**
5. **Shared-series leakage:** how many test rows would have a same-day, same-cell twin in training under a random split,
   leave-one-city-out, or a chronological split; city/date variance shares; why static features are identifiers.
6. **1–3 day lag features** (past only): correlation and partial correlation with tomorrow's heat index beyond today's value
   and season, plus the autocorrelation structure.
7. **Reference baselines** (persistence, majority class, monthly climatology) on the proposed test period, with Macro F1.
8. **Recommendations** that keep NCR as the scope.

## Using the data safely (`ncr_modeling_utils.py`)

* **Split by `Date`, never by row.** The CSV is sorted City-then-Date, so `train_test_split(shuffle=False)` or
  `TimeSeriesSplit` on the raw frame splits by *city*. Use `chronological_split(df, test_start, embargo_days)` and
  `expanding_window_folds(df, n_splits, gap_days)`; every date, with all its cities, stays on one side.
* `add_lag_features(df)` builds 1–3 day lags, 3-day means and a pressure change, calendar-aware and strictly past-only
  (tested by scrambling the future and checking nothing in the past changes).
* The committed dataset is in `data/ncr/output/`; to rebuild or verify it without any API call see `data/ncr/README.md`.
* `INPUT_FEATURES` is the specified 22-column feature list; `assert_no_future_information(cols)` rejects
  `HeatIndex_Max_Tomorrow`, `HeatLevelTomorrow` and anything that looks like tomorrow.

## Preprocessing (`ncr_preprocessing.py`, project Section 4)

One shared, leak-safe preprocessing for all three models. No model is trained here. `python ncr_preprocessing.py` checks the data and
writes `data/ncr/processed/`: `preprocessing_report.md` (every decision with its evidence), `split_definition.json` and
`ncr_model_ready.csv.gz` (the checked data plus the lag features, with a `Split` column; read it with `pd.read_csv`).

```python
import ncr_preprocessing as P
from sklearn.pipeline import Pipeline

prep = P.prepare_dataset("data/ncr/output/ph_heat_index_next_day.csv")     # checks, lags, date split, 5 time-aware folds
X, y = prep.X(prep.train, "hi_lags"), prep.y(prep.train)                   # raw feature frame and integer class codes
model = Pipeline([("prep", P.make_preprocessor("hi_lags")), ("clf", YourModel(random_state=P.RANDOM_STATE))])
# cross_validate(model, X, y, cv=prep.cv_splits, scoring=...)  # the transformer is re-fitted inside every training fold
# prep.test is the final test set: leave it alone until the selected model is evaluated once.
```

* **Split:** 80 / 20 of the calendar dates. Test 2023-10-18 to 2025-12-28; train 2015-01-04 to 2023-10-13; a 4-day gap
  (3 days of lag + 1 day of horizon) between them and before every validation block; 5 expanding-window folds; random state 42.
* **Checks:** no missing values, duplicates, impossible values or label mismatches in the committed data. Each city's first 3 days
  (no lag history) are dropped. Outliers are real storm days and are kept; rainfall and mean wind speed are log-transformed.
* **Transformer:** standard scaling, `DayOfYear` as sin/cos, one-hot `City` (or `static="coords"` / `"none"`), fitted on training rows only.
* **Feature sets:** `today` (today's weather), `hi_lags` (+ heat index of the previous 3 days), `all_lags` (+ 44 candidate lags and
  day-to-day changes). The set is chosen by cross-validation on the training rows. Two options follow the EDA (workspace Section 3):
  `weather="lr_pruned"` (the 7 weather columns kept for Logistic Regression) and `level_today=True` (adds today's heat level as a 0-4 code;
  off by default, the group decides).
* **Classes:** nothing is merged or dropped. Extreme Danger has no rows and Not Hazardous has 158 training rows (none in the test
  period); `class_weights` gives balanced weights from training labels, and `target_policy="three_class"` exists as an option that
  is **not applied** until the group decides.

## Key decisions

- **Class rule:** `HeatIndex_Max_Tomorrow` is the next day's maximum *hourly* heat index (NWS/Rothfusz on hourly temperature and
  humidity). The thresholds leave gaps for fractional values, so each class's lower bound is inclusive:
  `<27` Not Hazardous, `27–<33` Caution, `33–<42` Extreme Caution, `42–<52` Danger, `>=52` Extreme Danger.
- **Model pinned to `era5`.** Open-Meteo's default `best_match` blends in ECMWF IFS from 2017, a model change inside the period.
  A one-request preflight verifies all 10 variables come back (the docs are ambiguous about ERA5 wind gusts).
- **City list:** 149 PSGC cities from a pinned, SHA-256-verified `psgc==2026.4.13.0` PyPI wheel, filtered to NCR. Coordinates are
  population-weighted barangay centres (the package's area centroids can sit far from the urban core).
- **Raw responses are kept** (gzip, `raw/` in the work dir, ~25 MB for NCR and committed under `data/ncr/cache/`) so an aggregation fix never needs the API again.
- **Resumable and rate-limit aware:** per city-year cache, persisted call budget, graceful stop on the daily limit.

## Tests

```bash
python -m unittest discover -s tests -v      # ~1 min; two tests need PyPI (skipped if unreachable)
pip install metpy                             # optional: heat-index cross-check against MetPy
```

`tests/test_committed_data.py` checks the committed real build: it validates, the quoted numbers and checksum are true, and the
current code rebuilds it byte-for-byte from the committed cache and from the raw responses alone (offline).

`tests/mock_open_meteo.py` serves **synthetic** data with fault injection (500/503, 429 minutely/hourly/daily, truncated JSON,
short arrays, wrong timezone, quota) and a lapse-rate elevation effect. The analysis is tested against panels with
**planted** structure (known grid cells, a −0.65 °C/100 m lapse effect, AR(2) dynamics, known class counts).

## Layout

```
build_ph_heat_index_dataset.py   dataset pipeline (fetch, cache, aggregate, targets, validate, documentation)
ncr_dataset_analysis.py          pre-modelling analysis (no models trained)
ncr_modeling_utils.py            date-based splits, past-only lag features, future-information guard
ncr_preprocessing.py             shared preprocessing: data checks, split, folds, transformer, class weights (no models)
ph_heat_index_colab.ipynb        Colab runner (Drive-backed cache)
data/ncr/                        the real NCR build: output/ (5 files), cache/ (raw responses + chunks), processed/ (Section 4), README.md
reference/ph_city_list_base.csv  the 149-city PSGC list before the NCR filter and API enrichment
tests/                           unit + end-to-end tests and the mock API
```

Data attribution: weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0, non-commercial on the free tier),
containing modified Copernicus Climate Change Service (ERA5) information; barangay boundaries OCHA/HDX (CC BY-IGO) via `psgc` (MIT).
