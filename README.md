# heatcast-ph — Philippine next-day heat-index dataset

Builds `ph_heat_index_next_day.csv`: one row per **city-day** for the **149 Philippine cities**,
2015-01-01 → 2025-12-29 (Asia/Manila), from the **Open-Meteo Historical Weather API (ERA5 reanalysis)**.
Inputs are same-day weather aggregates; targets are the **next day's maximum heat index** (°C, exact) and its
**heat level** (Not Hazardous / Caution / Extreme Caution / Danger / Extreme Danger). No model is trained here.

> **Status:** the pipeline is implemented and tested against a local mock of the Open-Meteo API
> (52 tests, including full-scale 149-city runs on synthetic data). **The real dataset has not been
> collected yet** — the environment this was written in blocks `open-meteo.com`. Run it where the API is
> reachable (Google Colab: `ph_heat_index_colab.ipynb`). Nothing in this repo is real weather data.

## Run it

```bash
pip install -r requirements.txt
python build_ph_heat_index_dataset.py --work-dir ph_heat_index_cache --out-dir out
```

or open `ph_heat_index_colab.ipynb` in Colab (cache + outputs on Google Drive).

**Expect several runs.** A whole-year request for 10 hourly variables counts as ~26 weighted calls on Open-Meteo's
free tier (10,000/day; formula from their pricing page). The full job is ≈ **42,700 calls ≈ 5 daily runs**. Each
finished city-year is cached; re-run the same command and it resumes, stopping cleanly (exit code 2) when the daily
budget is used up. A client-side budget (500/min, 4,500/h, 9,500/day, persisted in `api_usage_log.json`) keeps you
under the server limits; delete that file if you know the server quota has reset. With a commercial key
(`OPEN_METEO_API_KEY` or `--api-key`) the customer endpoint is used and no client budget applies.

**NCR only?** Add `--region NCR` (also matches `Metro Manila`; any PSGC region name or fragment works, comma-separated
or repeated). That is 16 cities ≈ 4,600 weighted calls, i.e. **one day** on the free tier instead of five, and the
generated documentation states the regional scope. Caveat: Metro Manila's 16 cities fall into only about **2 ERA5
grid cells**, so you get roughly two distinct weather series, not sixteen (they differ mainly by elevation
adjustment). The cache is keyed per city, so a regional run reuses chunks from a full run and vice versa.

Useful flags: `--max-cities 3` (smoke test, clearly labelled SUBSET output), `--allow-partial`,
`--validate-only FILE.csv`, `--model` (default `era5`), `--no-keep-raw`. `--help` lists the rest.

**Raw responses are kept** (gzip, `raw/` inside `--work-dir`, roughly 150–250 MB for the full run) so that if the
aggregation or heat-index step ever has to change, the dataset is rebuilt offline in seconds instead of re-spending
~5 days of API quota. Bump `AGG_VERSION` after such a change; chunks are then re-aggregated from raw automatically.

## Outputs (written to `--out-dir`)

| File | Content |
|---|---|
| `ph_heat_index_next_day.csv` | the dataset (~105 MB; git-ignored: over GitHub's 100 MB limit) |
| `ph_heat_index_data_dictionary.csv` | column, dtype, role, unit, description, derivation |
| `ph_heat_index_city_list.csv` | cities used, coordinates, API elevation, ERA5 grid cell, provenance |
| `dataset_documentation.md` | source, endpoint, method, formula, categories, limitations, validation of *that* build |

Before burning API quota you can inspect the city list that will be used: `reference/ph_city_list_base.csv`
(real; rebuilt from the pinned source by the script, checked by a test). Elevation is added from the API at run time.

## Key decisions (details in the generated documentation)

- **City list:** 149 PSGC cities (33 HUC, 111 component, 5 independent component) from the pinned, SHA-256-verified
  `psgc==2026.4.13.0` PyPI wheel (community package of the PSA PSGC Q1 2026; not an official PSA product).
  Coordinates are the **population-weighted mean of barangay centroids**, not the package's area-weighted centroid,
  which sits up to 28 km from the urban core in big cities. Duplicate names get a province suffix (e.g. `Naga (Cebu)`).
- **Model pinned to `era5`.** Open-Meteo's default `best_match` blends in ECMWF IFS from 2017, a model change inside
  the period. A one-request preflight checks that all 10 variables actually come back.
- **Heat index:** NWS/Rothfusz procedure on hourly T and RH (°F internally, °C stored), not `apparent_temperature`.
- **Classes:** the thresholds leave gaps for fractional values, so each class's lower bound is inclusive
  (`27 ≤ HI < 33` = Caution, …) on the stored 2-decimal value.
- **Targets:** shifted by calendar date within city; each city's last row dropped; values rounded *before* shifting so
  tomorrow's target equals the next day's feature exactly.
- **Not PAGASA observations.** ERA5 is ~25 km gridded reanalysis; labels are bins of the numeric heat index.

## Tests

```bash
python -m unittest discover -s tests -v      # ~15 s, offline except two PyPI-based city-list tests (skipped if unreachable)
pip install metpy                             # optional: enables the heat-index cross-check vs MetPy
```

`tests/mock_open_meteo.py` serves **synthetic** data with fault injection (500/503, 429 minutely/hourly/daily,
truncated JSON, short arrays, wrong timezone, bad request, daily weighted quota).

## Layout

```
build_ph_heat_index_dataset.py   the whole pipeline (single file, runs unchanged in Colab)
ph_heat_index_colab.ipynb        thin Colab runner (Drive-backed cache)
reference/ph_city_list_base.csv  city list before API enrichment
tests/                           unit + end-to-end tests and the mock API
```

Data attribution: weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0, non-commercial on the free
tier), containing modified Copernicus Climate Change Service (ERA5) information; barangay boundaries OCHA/HDX (CC BY-IGO)
via `psgc` (MIT).
