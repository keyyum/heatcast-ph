# HeatCast NCR - pre-modelling data analysis

Dataset: 64,240 rows, 16 NCR cities, 2015-01-01 to 2025-12-28. Geographic scope is NCR / Metro Manila only. **No model has been trained**; everything below is counting, correlation and rule-based baselines computed from the data.

## Key findings

- 16 cities -> 2 ERA5 grid cell(s); 2 distinct non-thermal series; effective independent series ~1.05 (95% of anomaly variance needs 1 component(s)).
- Elevation adjustment: material (API elevations 2-51 m; same-cell mean |dHI| 0.21 C; class disagreement 2.3%).
- Class counts (HeatLevelTomorrow rows | distinct days | episodes | status): Not Hazardous: 158 (0.25%) | 16 | 13 | VERY FEW; Caution: 8,733 (13.59%) | 685 | 221 | ok; Extreme Caution: 50,018 (77.86%) | 3,510 | 204 | ok; Danger: 5,331 (8.30%) | 723 | 238 | ok; Extreme Danger: 0 (0.00%) | 0 | 0 | ABSENT
- Leakage: random split -> 100% of test rows have a same-day same-cell twin in train; chronological -> 0%.

## 1. How many distinct grid series are there?

The data are **gridded ERA5 reanalysis estimates for representative city coordinates** (population-weighted centres), not 16 independent weather stations. The 16 coordinates resolve to **2 Open-Meteo/ERA5 grid cell(s)** (grid centre returned by the API):

| grid cell | n_cities | cities | elev_min_m | elev_max_m | elev_range_m |
|---|---|---|---|---|---|
| 14.50N/121.00E | 10 | Las Piñas, Makati, Mandaluyong, Manila, Muntinlupa, Parañaque, Pasay, Pasig, San Juan, Taguig | 7.00 | 32.00 | 25.00 |
| 14.75N/121.00E | 6 | Caloocan, Malabon, Marikina, Navotas, Quezon, Valenzuela | 2.00 | 51.00 | 49.00 |

Measured from the data themselves: 2 group(s) of cities have identical non-thermal weather (pressure, cloud, wind, rain, solar) on at least 99.9 % of days: {Las Piñas, Makati, Mandaluyong, Manila, Muntinlupa, Parañaque, Pasay, Pasig, San Juan, Taguig}; {Caloocan, Malabon, Marikina, Navotas, Quezon, Valenzuela}.

Share of days on which two cities **in the same grid cell** report exactly the same value:

| column | identical_days_fraction |
|---|---|
| Pressure_Mean | 1.0000 |
| CloudCover_Mean | 1.0000 |
| WindSpeed_Mean | 1.0000 |
| WindSpeed_Max | 1.0000 |
| WindGust_Max | 1.0000 |
| Rainfall_Total | 1.0000 |
| RainHours | 1.0000 |
| SolarRadiation_Total | 1.0000 |

Effective number of independent series (participation ratio of deseasonalised daily heat-index anomalies): **1.05**; first principal component explains 97.6%; 1 component(s) reach 95 %. Mean / minimum pairwise anomaly correlation 0.975 / 0.950.

## 2. Does Open-Meteo's elevation adjustment separate cities?

API elevations span 2-51 m; the largest elevation range inside one grid cell is 49 m (a standard lapse rate would predict at most 0.32 °C temperature difference from that).

Pooled within-cell slopes per +100 m elevation (58 city pair(s); standard lapse -0.65 °C/100 m):

| variable | slope per +100 m |
|---|---|
| Temp_Mean | -0.644 |
| Temp_Max | -0.642 |
| Temp_Min | -0.644 |
| HeatIndex_Max_Today | -1.727 |
| Humidity_Mean (%) | -0.090 |
| DewPoint_Mean | -0.643 |

Between cities in the **same** cell: mean |ΔHI| 0.207 °C (95th pct 0.32, max 1.74); days with |ΔHI| >= 0.5 °C: 11.5%, >= 1 °C: 0.5%; today's heat level differs on 2.30% of days (tomorrow's: 2.30%). Between cities in **different** cells: mean |ΔHI| 1.339 °C, class disagreement 17.34%.

**Verdict (heuristic: material if mean |ΔHI| >= 0.5 °C or class disagreement >= 2%; modest if >= 0.2 °C or >= 0.5%): elevation adjustment is material.**

## 3. Class counts and percentages (target `HeatLevelTomorrow`)

Rows are city-days. *Distinct days* collapses the cities (a day counts once if any city is in the class, dated by the day the heat occurs); *cell-days* counts each grid cell once per day; *episodes* are runs of consecutive such days - the closest thing to an independent event count.

| class | rows | percent_rows | cell_days | distinct_days | episodes | longest_run_days | years_present |
|---|---|---|---|---|---|---|---|
| Not Hazardous | 158 | 0.25 | 23 | 16 | 13 | 3 | 6 |
| Caution | 8,733 | 13.59 | 1,203 | 685 | 221 | 27 | 11 |
| Extreme Caution | 50,018 | 77.86 | 6,476 | 3,510 | 204 | 179 | 11 |
| Danger | 5,331 | 8.30 | 894 | 723 | 238 | 22 | 11 |
| Extreme Danger | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 |

Reference - `HeatLevelToday` rows: Not Hazardous 174, Caution 8,733, Extreme Caution 50,002, Danger 5,331, Extreme Danger 0.

Rows per year (forecast-issue year):

| year | Not Hazardous | Caution | Extreme Caution | Danger | Extreme Danger |
|---|---|---|---|---|---|
| 2015 | 51 | 1,063 | 4,496 | 230 | 0 |
| 2016 | 0 | 493 | 4,777 | 586 | 0 |
| 2017 | 32 | 941 | 4,303 | 564 | 0 |
| 2018 | 22 | 830 | 4,722 | 266 | 0 |
| 2019 | 0 | 887 | 4,437 | 516 | 0 |
| 2020 | 13 | 565 | 4,289 | 989 | 0 |
| 2021 | 0 | 977 | 4,562 | 301 | 0 |
| 2022 | 19 | 646 | 4,916 | 259 | 0 |
| 2023 | 21 | 884 | 4,573 | 362 | 0 |
| 2024 | 0 | 641 | 4,304 | 911 | 0 |
| 2025 | 0 | 806 | 4,639 | 347 | 0 |

## 4. Danger and Extreme Danger

- **Danger**: 5,331 city-day rows on 723 distinct days; on those days on average 7.4 of 16 cities are in the class; all cities together on 14% of the days, a single city on 0%. By year: 2015: 230, 2016: 586, 2017: 564, 2018: 266, 2019: 516, 2020: 989, 2021: 301, 2022: 259, 2023: 362, 2024: 911, 2025: 347. By month of the heat day: 3: 65, 4: 695, 5: 1,926, 6: 1,329, 7: 514, 8: 316, 9: 356, 10: 82, 11: 43, 12: 5.
- **Extreme Danger: no observations.**

### Reliability check (flags only - no class is merged or dropped)

Heuristic thresholds: fewer than 100 distinct days or 10 episodes = FEW; fewer than 30 days or 5 episodes = VERY FEW; fewer than 30 days in the test partition, or a class missing from train/test in any CV fold, also raises FEW.

| class | status | reasons |
|---|---|---|
| Not Hazardous | VERY FEW | 16 distinct days (< 100); 2 test-period days (< 30); CV folds [5] lack this class in train or test |
| Caution | ok | - |
| Extreme Caution | ok | - |
| Danger | ok | - |
| Extreme Danger | ABSENT | no observations |

### Proposed chronological split (embargo 1 day(s))

Train 2015-01-01 to 2022-12-30 (46,736 rows); test 2023-01-01 to 2025-12-28 (17,488 rows), test starts 2023-01-01.

| class | train rows | train days | test rows | test days | test minus train (pp) |
|---|---|---|---|---|---|
| Not Hazardous | 137 | 14 | 21 | 2 | -0.17 |
| Caution | 6,402 | 510 | 2,331 | 175 | -0.37 |
| Extreme Caution | 36,486 | 2,553 | 13,516 | 956 | -0.78 |
| Danger | 3,711 | 505 | 1,620 | 218 | 1.32 |
| Extreme Danger | 0 | 0 | 0 | 0 | 0.00 |

### Expanding-window CV folds (gap 1 day(s))

| fold | train_end | test_start | test_end | train Not Hazardous | train Caution | train Extreme Caution | train Danger | train Extreme Danger | test Not Hazardous | test Caution | test Extreme Caution | test Danger | test Extreme Danger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2017-01-03 | 2017-01-05 | 2018-10-22 | 51 | 1,574 | 9,303 | 816 | 0 | 32 | 1,589 | 8,045 | 830 | 0 |
| 2 | 2018-10-21 | 2018-10-23 | 2020-08-08 | 83 | 3,174 | 17,337 | 1,646 | 0 | 22 | 1,450 | 7,695 | 1,329 | 0 |
| 3 | 2020-08-07 | 2020-08-09 | 2022-05-26 | 105 | 4,618 | 25,038 | 2,975 | 0 | 13 | 1,513 | 8,395 | 575 | 0 |
| 4 | 2022-05-25 | 2022-05-27 | 2024-03-12 | 118 | 6,137 | 33,433 | 3,544 | 0 | 40 | 1,495 | 8,438 | 523 | 0 |
| 5 | 2024-03-11 | 2024-03-13 | 2025-12-28 | 158 | 7,632 | 41,865 | 4,073 | 0 | 0 | 1,101 | 8,137 | 1,258 | 0 |

## 5. Shared grid series and leakage

- 64,240 city rows correspond to only 8,030 (grid cell, day) combinations.
- On 100.0% of city-days another city has identical non-thermal weather the same day (identical full weather incl. temperature/humidity: 75.9%); distinct non-thermal vectors per day: mean 2.00, max 2.
- Variance of `HeatIndex_Max_Tomorrow` explained by date alone: 95.2%; by (grid cell, date): 99.6%; by city identity alone: 3.16%. The 16 (Latitude, Longitude, Elevation) combinations are identifiers, not independent information.
- Partition simulations (assignment only, no model): share of **test rows that have a same-day twin in training**:
  - chronological split: any city 0.0%, same cell 0.0%; exact duplicate weather vectors across partitions: 0.
  - random 80/20 row split: any city 100.0%, same cell 100.0%  -> **invalid**.
  - leave-one-city-out: 100.0% of held-out rows keep a same-cell mate in training -> **invalid** as a generalisation test.
  - naive 'first 80 % of rows' of the saved file (sorted by City, Date): train 2015-01-01..2025-12-28 (13 cities), test 2015-01-01..2025-12-28 (4 cities) -> **NOT chronological: it splits by city**.

## 6. Lag features (1-3 days, past information only)

Autocorrelation of the daily mean heat index, lag 1: raw 0.831, seasonally adjusted 0.642. Partial autocorrelation of the adjusted series: lag 1: 0.641, lag 2: -0.004, lag 3: 0.052, lag 4: -0.002, lag 5: 0.007, lag 6: 0.024, lag 7: 0.014.

Correlation with `HeatIndex_Max_Tomorrow` and partial correlation after controlling for `HeatIndex_Max_Today` and season (useful = |partial r| >= 0.05, a linear heuristic):

| feature | n_rows | corr_with_target | partial_corr | useful |
|---|---|---|---|---|
| HeatIndex_Max_Today_lag2 | 64,208 | 0.674 | 0.129 | yes |
| HeatIndex_Max_Today_lag3 | 64,192 | 0.636 | 0.122 | yes |
| Temp_Max_lag3 | 64,192 | 0.504 | 0.118 | yes |
| HeatIndex_Max_Today_mean3d | 64,208 | 0.800 | 0.117 | yes |
| Temp_Max_lag2 | 64,208 | 0.532 | 0.116 | yes |
| DewPoint_Mean_mean3d | 64,208 | 0.614 | 0.112 | yes |
| Temp_Max_mean3d | 64,208 | 0.638 | 0.089 | yes |
| DewPoint_Mean_lag1 | 64,224 | 0.590 | 0.083 | yes |
| DewPoint_Mean_lag2 | 64,208 | 0.547 | 0.077 | yes |
| Humidity_Mean_lag3 | 64,192 | -0.011 | -0.074 | yes |
| HeatIndex_Max_Today_lag1 | 64,224 | 0.726 | 0.073 | yes |
| DewPoint_Mean_lag3 | 64,192 | 0.511 | 0.068 | yes |
| Temp_Max_lag1 | 64,224 | 0.575 | 0.066 | yes |
| Humidity_Mean_lag2 | 64,208 | -0.003 | -0.065 | yes |
| Rainfall_Total_mean3d | 64,208 | 0.030 | 0.060 | yes |
| Rainfall_Total_lag1 | 64,224 | 0.037 | 0.045 | no |
| RainHours_lag3 | 64,192 | 0.153 | -0.045 | no |
| SolarRadiation_Total_lag3 | 64,192 | 0.292 | 0.036 | no |
| SolarRadiation_Total_lag2 | 64,208 | 0.308 | 0.032 | no |
| Humidity_Mean_lag1 | 64,224 | 0.002 | -0.030 | no |
| SolarRadiation_Total_mean3d | 64,208 | 0.406 | -0.030 | no |
| Humidity_Mean_mean3d | 64,208 | -0.005 | -0.029 | no |
| SolarRadiation_Total_lag1 | 64,224 | 0.334 | -0.029 | no |
| RainHours_lag2 | 64,208 | 0.144 | -0.026 | no |
| WindSpeed_Mean_lag1 | 64,224 | -0.333 | 0.023 | no |
| CloudCover_Mean_lag3 | 64,192 | 0.090 | -0.021 | no |
| Pressure_Mean_lag1 | 64,224 | -0.310 | -0.017 | no |
| WindSpeed_Mean_mean3d | 64,208 | -0.393 | -0.014 | no |
| WindSpeed_Mean_lag2 | 64,208 | -0.282 | -0.011 | no |
| Rainfall_Total_lag2 | 64,208 | 0.054 | -0.011 | no |
| RainHours_mean3d | 64,208 | 0.124 | -0.011 | no |
| CloudCover_Mean_mean3d | 64,208 | 0.051 | -0.010 | no |
| Pressure_Mean_mean3d | 64,208 | -0.321 | -0.009 | no |
| Rainfall_Total_lag3 | 64,192 | 0.069 | -0.006 | no |
| Pressure_Mean_change1d | 64,224 | 0.013 | 0.006 | no |
| Pressure_Mean_lag2 | 64,208 | -0.304 | 0.005 | no |
| CloudCover_Mean_lag2 | 64,208 | 0.079 | -0.004 | no |
| CloudCover_Mean_lag1 | 64,224 | 0.051 | 0.003 | no |
| WindSpeed_Mean_lag3 | 64,192 | -0.237 | 0.002 | no |
| RainHours_lag1 | 64,224 | 0.120 | -0.001 | no |
| Pressure_Mean_lag3 | 64,192 | -0.300 | -0.001 | no |

Each city's first three days have no lags (NaN); build them with `ncr_modeling_utils.add_lag_features`, which is calendar-aware and strictly past-only.

## 7. Reference baselines (rules, not trained models) on the proposed test period

### persistence (tomorrow = today's class)

Macro F1 **0.532**, accuracy 0.829.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Not Hazardous | 0.000 | 0.000 | 0.000 | 21 |
| Caution | 0.638 | 0.638 | 0.638 | 2,331 |
| Extreme Caution | 0.891 | 0.891 | 0.891 | 13,516 |
| Danger | 0.599 | 0.599 | 0.599 | 1,620 |
| Extreme Danger | 0.000 | 0.000 | 0.000 | 0 |

|  | pred Not Hazardous | pred Caution | pred Extreme Caution | pred Danger | pred Extreme Danger |
|---|---|---|---|---|---|
| true Not Hazardous | 0 | 21 | 0 | 0 | 0 |
| true Caution | 21 | 1,487 | 823 | 0 | 0 |
| true Extreme Caution | 0 | 823 | 12,044 | 649 | 0 |
| true Danger | 0 | 0 | 649 | 971 | 0 |
| true Extreme Danger | 0 | 0 | 0 | 0 | 0 |

### majority class of train ('Extreme Caution')

Macro F1 **0.218**, accuracy 0.773.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Not Hazardous | 0.000 | 0.000 | 0.000 | 21 |
| Caution | 0.000 | 0.000 | 0.000 | 2,331 |
| Extreme Caution | 0.773 | 1.000 | 0.872 | 13,516 |
| Danger | 0.000 | 0.000 | 0.000 | 1,620 |
| Extreme Danger | 0.000 | 0.000 | 0.000 | 0 |

### climatology (train majority class of the target month)

Macro F1 **0.330**, accuracy 0.786.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Not Hazardous | 0.000 | 0.000 | 0.000 | 21 |
| Caution | 0.575 | 0.363 | 0.445 | 2,331 |
| Extreme Caution | 0.805 | 0.954 | 0.874 | 13,516 |
| Danger | 0.000 | 0.000 | 0.000 | 1,620 |
| Extreme Danger | 0.000 | 0.000 | 0.000 | 0 |

## 8. Recommendations (NCR scope retained)

1. **Treat the data as ~1.0 effective independent weather series, not 16.** The 16 cities resolve to 2 ERA5 grid cell(s); report the sample size honestly (8,030 grid-cell days vs 64,240 city rows) and do not describe the cities as independent stations.
2. **Split by `Date`, never by row.** The CSV is sorted City-then-Date, so `train_test_split(shuffle=False)` or `TimeSeriesSplit` on the raw frame splits by city (confirmed: the first 80 % of rows is not chronological). Use `ncr_modeling_utils.chronological_split` / `expanding_window_folds` (all cities of a date stay together).
3. **Keep a 1-3 day embargo between train and test.** For a 1-day horizon the overlap is only between the last training labels and the first test *inputs* (no test target is exposed), so this is cheap insurance rather than a hard requirement - but it becomes necessary once rolling features are used.
4. **Do not use random row splits or leave-one-city-out.** A random 80/20 split gives 100% of test rows a same-date, same-grid-cell twin in training; leave-one-city-out gives 100%. Both leak almost-identical weather into training. A chronological split gives 0% by construction.
5. **Treat `City`, `Latitude`, `Longitude`, `Elevation` as identifiers.** They take 16 distinct combinations; city identity explains 3.2% of target variance and the date alone explains 95.2%. Fit with and without them (or use grid-cell id) and prefer the simpler model if scores are equal; tree models can otherwise memorise city/cell identity.
6. **Elevation adjustment is material** (mean |ΔHI| between cities in the same cell 0.21 °C; 2.3% of same-cell city-days land in different classes). Cities differ enough within a cell that city-level labels carry some distinct information.
7. **Rare classes need a decision from you (nothing was merged or dropped):** Not Hazardous: VERY FEW (16 distinct days (< 100); 2 test-period days (< 30); CV folds [5] lack this class in train or test); Extreme Danger: ABSENT (no observations). Options that keep the 5-class task: class-weighted training, and per-class metrics with date-block bootstrap confidence intervals reported alongside Macro F1; merging classes would be a separate, explicit decision.
8. **Get uncertainty from dates, not rows.** Use a date-block bootstrap (resample days or weeks, keep all cities of a day together); row-level standard errors are far too small because cities repeat the same weather and consecutive days are autocorrelated.
9. **Expect distribution shift across years.** Compare the per-year class table: heat years (e.g. 2015-16, 2023-24) are not exchangeable with cool years, so report per-fold and per-year scores, not only one pooled number.
10. **Lag features (past only):** candidates with |partial r| >= 0.05 beyond today's heat index and season: `HeatIndex_Max_Today_lag2`, `HeatIndex_Max_Today_lag3`, `Temp_Max_lag3`, `HeatIndex_Max_Today_mean3d`, `Temp_Max_lag2`, `DewPoint_Mean_mean3d`, `Temp_Max_mean3d`, `DewPoint_Mean_lag1`. Linear partial correlation understates nonlinear value for tree models, so confirm with time-aware CV; keep any feature set that does not beat today's-values-only on Macro F1 out.
11. **Always compare against the reference baselines below** (persistence is a strong baseline for next-day heat). Add ordinal-aware diagnostics (e.g. the confusion matrix, errors of one class vs several classes) because the five classes are ordered.
12. `HeatIndex_Max_Tomorrow` and `HeatLevelTomorrow` must never enter the feature list: use `ncr_modeling_utils.assert_no_future_information`. `HeatLevelToday` is a binning of `HeatIndex_Max_Today` (redundant).

