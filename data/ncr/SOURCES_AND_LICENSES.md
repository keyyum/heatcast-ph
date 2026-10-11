# HeatCast NCR - data sources, licences and permissions

Source note for the dataset package (`data/ncr/output/ph_heat_index_next_day.csv` and everything derived from it).
All data are public; no personal data are collected. Retrieved on **2026-10-07** with `build_ph_heat_index_dataset.py`.

| Source | What it provides | Link | Licence / terms | Attribution required |
|---|---|---|---|---|
| Open-Meteo Historical Weather API, model `era5` | Hourly 2 m temperature, relative humidity, dew point, mean-sea-level pressure, cloud cover, precipitation, rain, shortwave radiation, 10 m wind speed and gusts for the 16 city coordinates, 2015-01-01 to 2025-12-29, time zone Asia/Manila | <https://open-meteo.com/en/docs/historical-weather-api> (endpoint `https://archive-api.open-meteo.com/v1/archive`) | CC BY 4.0. The free API is for non-commercial use. | "Weather data by Open-Meteo.com" |
| Copernicus Climate Change Service, ERA5 hourly data on single levels | The reanalysis behind the Open-Meteo values (about 25 km grid) | <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels> | Copernicus licence (data served through Open-Meteo) | "Contains modified Copernicus Climate Change Service information" |
| Philippine Standard Geographic Code (PSA, Q1 2026 release) via the `psgc` package v2026.4.13.0 | The 16 NCR cities, PSGC codes, 2024 census barangay populations | <https://pypi.org/project/psgc/>, <https://github.com/fish-and-bear/psgc> | PSGC: public information of the Philippine Statistics Authority. Package: MIT (community package, not affiliated with the PSA). Wheel verified against SHA-256 `e9662b69f3313d90089896c567d9b82be2452358d16f3eb561a9f44b5212a778`. | Cite the PSA PSGC and the `psgc` package |
| OCHA / HDX Philippines Subnational Administrative Boundaries (2023), bundled in `psgc` | Barangay centroids used for each city's population-weighted coordinates (barangays flagged `fallback_unverified` excluded) | <https://data.humdata.org/dataset/cod-ab-phl> | CC BY-IGO | Cite OCHA / HDX |

## Methods used (not datasets)

| Method | Use | Reference |
|---|---|---|
| NOAA/NWS heat index equation (Rothfusz regression with NWS adjustments) | Hourly heat index from temperature and relative humidity | <https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml>; L. P. Rothfusz, NWS Southern Region Technical Attachment SR/SSD 90-23, 1990 |
| PAGASA heat index categories | Class thresholds 27 / 33 / 42 / 52 °C for the five heat levels | Cite PAGASA's own heat index advisory in the paper |

## The derived dataset

`ph_heat_index_next_day.csv`, the processed `ncr_model_ready.csv.gz` and their documentation were produced for this course
project from the sources above. They carry the attributions above, and the source terms still apply to them (in particular,
Open-Meteo's free API terms are non-commercial). Ownership and authorship of the project output are set out in the signed
Ownership and Authorship Declaration; it does not claim ownership of the third-party data listed here.

## Limits that follow from the sources

- ERA5 values are gridded reanalysis estimates (about 25 km), not PAGASA station observations. The heat levels here are
  computed from those estimates and are not official PAGASA heat index values.
- The 16 city coordinates fall into 2 ERA5 grid cells, so cities in a cell share the same underlying weather.
- ERA5 can be revised upstream, so a new download may differ slightly. The raw API responses used for this build are kept in
  `data/ncr/cache/raw/`, and the CSV can be rebuilt from them byte for byte (see `data/ncr/README.md`).
