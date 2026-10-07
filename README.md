# HeatCast NCR

**Next-Day Maximum Heat Index Risk Classification in Metro Manila Using Traditional Machine Learning**

This repository builds and checks the dataset for that project. **Scope is NCR / Metro Manila only** (the 16 NCR cities
of the PSA PSGC list) and is not meant to be extended automatically. The ML task is **classification** of
`HeatLevelTomorrow`; Logistic Regression, Random Forest and Gradient Boosting are planned, with Macro F1 as the
likely primary metric. **No model is trained here.**

> **Status:** the pipeline and the pre-modelling analysis are implemented and tested (94 tests) against a local mock of
> the Open-Meteo API. **The real NCR dataset has not been built yet**: the environment this was developed in blocks
> `open-meteo.com`, so no real class counts or grid-series measurements exist yet. Run it where the API is reachable
> (Google Colab: `ph_heat_index_colab.ipynb`); one run costs ~4,600 weighted API calls = one day of the free tier.
> Nothing in this repository is real weather data.

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
* `INPUT_FEATURES` is the specified 22-column feature list; `assert_no_future_information(cols)` rejects
  `HeatIndex_Max_Tomorrow`, `HeatLevelTomorrow` and anything that looks like tomorrow.

## Key decisions

- **Class rule:** `HeatIndex_Max_Tomorrow` is the next day's maximum *hourly* heat index (NWS/Rothfusz on hourly temperature and
  humidity). The thresholds leave gaps for fractional values, so each class's lower bound is inclusive:
  `<27` Not Hazardous, `27–<33` Caution, `33–<42` Extreme Caution, `42–<52` Danger, `>=52` Extreme Danger.
- **Model pinned to `era5`.** Open-Meteo's default `best_match` blends in ECMWF IFS from 2017, a model change inside the period.
  A one-request preflight verifies all 10 variables come back (the docs are ambiguous about ERA5 wind gusts).
- **City list:** 149 PSGC cities from a pinned, SHA-256-verified `psgc==2026.4.13.0` PyPI wheel, filtered to NCR. Coordinates are
  population-weighted barangay centres (the package's area centroids can sit far from the urban core).
- **Raw responses are kept** (gzip, `raw/` in the work dir, a few tens of MB for NCR) so an aggregation fix never needs the API again.
- **Resumable and rate-limit aware:** per city-year cache, persisted call budget, graceful stop on the daily limit.

## Tests

```bash
python -m unittest discover -s tests -v      # ~35 s; two tests need PyPI (skipped if unreachable)
pip install metpy                             # optional: heat-index cross-check against MetPy
```

`tests/mock_open_meteo.py` serves **synthetic** data with fault injection (500/503, 429 minutely/hourly/daily, truncated JSON,
short arrays, wrong timezone, quota) and a lapse-rate elevation effect. The analysis is tested against panels with
**planted** structure (known grid cells, a −0.65 °C/100 m lapse effect, AR(2) dynamics, known class counts).

## Layout

```
build_ph_heat_index_dataset.py   dataset pipeline (fetch, cache, aggregate, targets, validate, documentation)
ncr_dataset_analysis.py          pre-modelling analysis (no models trained)
ncr_modeling_utils.py            date-based splits, past-only lag features, future-information guard
ph_heat_index_colab.ipynb        Colab runner (Drive-backed cache)
reference/ph_city_list_base.csv  the 149-city PSGC list before the NCR filter and API enrichment
tests/                           unit + end-to-end tests and the mock API
```

Data attribution: weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0, non-commercial on the free tier),
containing modified Copernicus Climate Change Service (ERA5) information; barangay boundaries OCHA/HDX (CC BY-IGO) via `psgc` (MIT).
