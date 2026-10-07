# HeatCast NCR - Next-Day Maximum Heat-Index Dataset (Metro Manila)

Project: *HeatCast NCR: Next-Day Maximum Heat Index Risk Classification in Metro Manila Using Traditional Machine Learning.*

Generated 2026-10-07 10:11 UTC by `build_ph_heat_index_dataset.py` v1.0.0
(Python 3.13.16, pandas 2.2.3, numpy 2.1.3).

> **Scope: National Capital Region (NCR) only** - 16 of the 149 Philippine PSGC cities. This is not a nationwide dataset and is not meant to be extended automatically.

## What this is

One row per city-day for **16 NCR cities**, 2015-01-01 to 2025-12-28
(64,240 rows x 26 columns). Inputs are same-day weather aggregates; the
targets are the **next day's** maximum hourly heat index (numeric) and its heat level (class).
**The ML task is classification of `HeatLevelTomorrow`** (five ordered classes); `HeatIndex_Max_Tomorrow` is kept as the
exact numeric value behind the label and must never be used as an input.
Files: `ph_heat_index_next_day.csv`, `ph_heat_index_data_dictionary.csv`, `ph_heat_index_city_list.csv`,
`dataset_documentation.md`, `ncr_dataset_analysis.md` (pre-modelling analysis).

## Shared grid series - read before modelling

**These are gridded ERA5 reanalysis estimates for representative city coordinates (population-weighted city centres).
They are not station observations, and the 16 cities are not 16 independent weather stations.**
Nearby coordinates fall into the same ~25 km ERA5 cell and receive the same underlying meteorological data. This is
accepted by design (NCR-only scope); it is measured here, not hidden. Full numbers: `ncr_dataset_analysis.md`.

| Grid cell (API centre) | Cities | Names |
|---|---|---|
| 14.50N/121.00E | 10 | Las Piñas, Makati, Mandaluyong, Manila, Muntinlupa, Parañaque, Pasay, Pasig, San Juan, Taguig |
| 14.75N/121.00E | 6 | Caloocan, Malabon, Marikina, Navotas, Quezon, Valenzuela |

- **Distinct ERA5 grid cells: 2** for 16 cities. Cities in one cell report identical pressure,
  cloud, wind, rain and solar values on 100.0%-100.0% of days depending on the column.
- **Effective number of independent series: ~1.05** (participation ratio of deseasonalised daily
  heat-index anomalies; 1 principal component(s) reach 95 % of the variance).
- **Elevation adjustment: material.** API elevations span 2-51 m. Between cities in the same cell the mean |difference| of the daily maximum heat index is 0.21 °C and their heat level differs on 2.30% of days (heuristic thresholds in the analysis report).
- **Sample size:** 64,240 city-day rows represent only 8,030 (grid cell, day) combinations. Row-level
  counts and standard errors overstate the information content.
- **Leakage:** random row splits or leave-one-city-out would give 100% /
  100% of test rows a same-day same-cell twin in training;
  a chronological split by `Date` gives 0%. This CSV is sorted City-then-Date, so
  "first 80 % of rows" is **not** a chronological split.

## Data source

- **Open-Meteo Historical Weather API** (<https://open-meteo.com/en/docs/historical-weather-api>), reanalysis data.
- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Model pinned to `era5`** (ERA5, 0.25 degree, about 25 km). The API default `best_match` blends ECMWF IFS
  (9 km, 2017 onward) with ERA5; that would change the underlying model inside the 2015-2025 window, so it is not used.
  `cell_selection=land` (land grid cell with similar elevation to the request).
- **Timezone:** `Asia/Manila` (UTC+8, no daylight saving). All timestamps are local; days are local calendar days.
- **Period requested:** 2015-01-01 to 2025-12-29 (inclusive); data retrieved between 2026-10-07 and 2026-10-07 (cache-file timestamps, approximate).
- **Units requested:** temperature °C, wind km/h, precipitation mm (API explicit parameters).
- **Hourly variables requested:** `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `pressure_msl`, `cloud_cover`, `precipitation`, `rain`, `shortwave_radiation`, `wind_speed_10m`, `wind_gusts_10m`.
- **Attribution:** weather data by Open-Meteo.com (CC BY 4.0); contains modified Copernicus Climate Change Service
  (ERA5) information. The free API is for non-commercial use.

> **ERA5 is gridded reanalysis data, not PAGASA station observations.** Each value is a model estimate for a
> ~25 km grid cell, adjusted by Open-Meteo for the elevation difference between the cell and the requested point
> (90 m elevation model; see the Open-Meteo documentation). Heat-index values here are
> **not** PAGASA-reported station heat indices and can differ from them.

## City list

- **Source:** Philippine Standard Geographic Code (PSGC), PSA Q1 2026 release, as packaged by the community
  `psgc` Python package **v2026.4.13.0** from PyPI (MIT licence; not affiliated with the PSA). The wheel is downloaded
  from PyPI and verified against SHA-256 `e9662b69f3313d90089896c567d9b82be2452358d16f3eb561a9f44b5212a778`; it is read as data, never imported.
- **Selection:** PSGC records with geographic level *City* - **16 cities**
  (16 Highly Urbanized City); restricted to the project scope (National Capital Region (NCR): 16 of the 149 cities in that release).
  Municipalities are not included.
- **Coordinates:** population-weighted mean of the city's barangay centroids (2024 census population; barangay
  centroids from OCHA/HDX 2023 administrative boundaries via `psgc`, CC BY-IGO). Barangays flagged
  `fallback_unverified` are excluded. This approximates the populated area; the package's own area-weighted
  city centroid can lie far from the urban core (up to 2 km here, see `shift_vs_area_centroid_km`).
- **Elevation:** taken from the Open-Meteo response for the requested coordinates (the PSGC data has no elevation).
- **Grid cells:** the 16 cities map to **2 distinct ERA5 grid cell(s)** (`grid_latitude`/`grid_longitude`
  in the city list; see "Shared grid series" above). Cities sharing a cell receive the same underlying data.
- **City keys:** PSGC name without "City of"/"City" (a name that occurs twice nationwide would be written
  `Name (Province)`; none do within NCR). Official names and PSGC codes are in the city list.

## Collection method

1. Build the city list (above).
2. For each city and calendar year, request the 10 hourly variables (one request per city-year; whole-year
   requests cost ~26 weighted API calls each; ~4,590 weighted calls for the cities in this build).
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
~27 °C (e.g. heavy-rain or typhoon days), which are rare in Metro Manila.

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
| RainHours | rain | count of hours with rain >= 0.1 mm |
| SolarRadiation_Total | shortwave_radiation | sum of hourly W/m² (= Wh/m² per day) |
| HeatIndex_Max_Today | hourly heat index | max |

`precipitation` is collected and used only as a consistency check against `rain` (max daily absolute difference
0.0 mm); it is not an output column. Values are rounded to 2 decimals
(solar total 1) **before** targets are created, so the numeric targets are exactly equal to the next day's feature.

## Target creation

Within each city (sorted by Date): `HeatIndex_Max_Tomorrow` = `HeatIndex_Max_Today` shifted by -1 day, attached
only when the next row is exactly the next calendar day. The **last row of every city is dropped** because tomorrow
is unavailable (16 rows). Only today's data appear in the input columns; no tomorrow
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
| Not Hazardous | 158 | 0.25% |
| Caution | 8,733 | 13.59% |
| Extreme Caution | 50,018 | 77.86% |
| Danger | 5,331 | 8.30% |
| Extreme Danger | 0 | 0.00% |

## Validation (this build)

- Rows: 64,240; cities: 16; expected rows per city: 4,015.
- Missing values: 0; duplicate (City, Date): 0; city/date gaps: none.
- Impossible values: none.
- `HeatIndex_Max_Tomorrow` equals the following day's `HeatIndex_Max_Today` (same city, exact equality) in all
  64,224 checkable rows (0 mismatches).
- Warnings:
- none

## Known limitations

- **Reanalysis, not observations.** ~25 km grid cells; no station data. Reanalysis humidity and temperature can be
  biased versus PAGASA stations, so absolute heat-index levels may be off, particularly for the coastal cities.
- **Hourly instantaneous values.** The daily heat-index maximum is the maximum of 24 hourly values; the true
  sub-hourly peak and PAGASA's station-based reporting practice may differ.
- **Heat-index formula range.** The NWS regression is fitted for shade, light wind, T >= 80 °F and RH >= 40 %; it is
  an approximation outside that range and does not account for direct sun or wind. It is not a physiological model.
- **Few independent locations (by design).** NCR's 16 cities resolve to 2 ERA5 grid cell(s), roughly 1.0 effective independent series. Do not describe them as independent stations; split by date, treat City/Latitude/Longitude/Elevation as identifiers, and take uncertainty from date blocks (see `ncr_dataset_analysis.md`).
- **Temporal dependence.** Consecutive days are strongly autocorrelated; use chronological validation.
- **City coordinates are model inputs, not city-hall points.** Population-weighted centers can fall outside
  built-up areas for dispersed or multi-centre cities, and `cell_selection=land` may move to a neighbouring cell.
- **City definition.** The list is the PSGC Q1 2026 city set, including cities converted from municipalities after
  2015; the weather series is physical and unaffected, but "city" status was not constant over 2015-2025.
- **Class balance is determined by climate**, not by design; see the distribution above and the class-sufficiency
  flags in `ncr_dataset_analysis.md` (rare classes are flagged there; none are merged or dropped automatically).
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

This build needs about 1 free-tier day(s) of API budget (4,590 weighted calls; client budget
9,500/day); cached chunks are reused. In Google Colab use `ph_heat_index_colab.ipynb` with Google Drive as the
cache location. The pre-modelling analysis is regenerated with `python ncr_dataset_analysis.py --dataset ... --city-list ...`.
