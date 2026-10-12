# Workspace Section 5.2 — Random Forest (draft text, Tala, Edrian)

Numbers come from `notebooks/05_2_random_forest.ipynb` (this repo, 16-city NCR build, five-class target). Validation only; the test days were not used.

*I fitted a Random Forest behind the shared transformer, so the forest sees the same inputs as the other two models. I varied the feature set (today's weather, plus the heat index of the last 3 days, or plus all 44 candidate lags), the weather set (all 16 daily columns or the 7 pruned ones), the tree depth (4, 8, 12, unlimited), the minimum leaf size (5, 20, 50), the number of inputs tried at each split (square root of the inputs or half), the strength of the class weights (power 0, 0.25, 0.5, 0.75, 1, where 1 is "balanced"), the number of trees (100, 300, 600) and whether to keep the city input. That is 36 settings, each scored on the same five time-ordered folds inside the training days. I did not use out-of-bag scores, because out-of-bag rows are random rows from the same period and so are not a time-aware check. The test days were not used.*

*The best setting uses today's weather plus the heat index of the last 3 days (all 16 weather columns), 100 trees of depth at most 8, at least 5 days per leaf, the square root of the inputs at each split, class weights at half strength (power 0.5) and no city input. Dropping the city changed macro F1 by 0.001, so it was left out (fewer inputs for the app). Using 300 or 600 trees gave the same macro F1 (0.598 to 0.600), so 100 was kept. Class-weight strength made the largest difference: with no weights Danger recall is 0.41 and precision 0.74; with fully balanced weights recall is 0.78 but precision falls to 0.43 and macro F1 to 0.572. Half strength gave the best macro F1 of the five while keeping Danger recall (0.62) above the copy-today rule's 0.58 and precision above 0.60; stronger weights catch more Danger days but at a clearly higher false-alarm cost. Adding the last 3 days of heat index helped slightly (0.593 against 0.592 for today only); adding all 44 lags did not (0.578), and the 7-column pruned weather set was lower than the full set (0.583 against 0.592).*

| Setting | Macro F1 | Danger recall | Danger precision |
|---|---|---|---|
| Weight power 0 (none) | 0.568 | 0.408 | 0.739 |
| 0.25 | 0.586 | 0.511 | 0.689 |
| **0.5 (selected)** | **0.600** | **0.616** | **0.607** |
| 0.75 | 0.582 | 0.705 | 0.509 |
| 1 (balanced) | 0.572 | 0.783 | 0.425 |

*(rows above are the stage-3 runs with 300 trees; the selected row with 100 trees and no city input scores 0.599 / 0.619 / 0.599.)*

Over the five folds the chosen setting scores (mean ± SD), against the copy-today rule on the same folds:

| Metric | Random Forest | Copy today |
|---|---|---|
| Macro F1 (classes present) | 0.599 ± 0.084 | 0.572 ± 0.084 |
| Accuracy | 0.849 ± 0.025 | 0.835 ± 0.026 |
| Danger recall | 0.619 ± 0.054 | 0.581 ± 0.073 |
| Danger precision | 0.599 ± 0.069 | 0.582 ± 0.073 |
| Danger false-alarm rate | 0.035 ± 0.006 | 0.036 ± 0.012 |
| ROC-AUC (one-vs-rest) | 0.913 ± 0.029 | n/a |

*The Random Forest beats the copy-today rule on every row. Fold by fold, it is ahead on macro F1 in 5 of 5 folds and on accuracy in 5 of 5; ahead on Danger recall in 4 of 5 (fold 1 is 0.011 behind) and on Danger precision in 4 of 5 (fold 2 is 0.024 behind). The average gain in macro F1 is only 0.027, smaller than the fold-to-fold SD of 0.084, because the folds differ a lot from each other (Danger is seasonal and varies from year to year). So the result is a small but consistent improvement and not a large one.*

Confusion matrix, summed over the five validation blocks (rows = true level, columns = predicted level):

| True / predicted | Not Hazardous | Caution | Extreme Caution | Danger |
|---|---|---|---|---|
| Not Hazardous | 6 | 88 | 13 | 0 |
| Caution | 0 | 3,645 | 1,952 | 2 |
| Extreme Caution | 0 | 1,426 | 27,875 | 1,275 |
| Danger | 0 | 6 | 1,212 | 2,020 |

*Of the 3,238 true Danger days, 2,020 are predicted as Danger (pooled recall about 62%) and 1,212 as Extreme Caution; only 6 are predicted as Caution. Of the 3,297 days called Danger, 2,020 really were (pooled precision about 61%); the other 1,275 were Extreme Caution days. Not Hazardous (107 validation rows) is almost never found (6 correct), as expected for a class with so few days. The inputs the forest relies on most are today's heat index (impurity importance 0.22; shuffling it drops macro F1 by 0.098 on the last fold), then mean temperature, dew point, the heat index of the previous days and the day of the year. The cities add little, which fits the fact that the 16 cities share only 2 weather grid cells. Fold scores vary (macro F1 from 0.537 to 0.736), so the group should compare models on the mean and on how often they win, not on one fold.*

Caveats to state in the paper: the selected setting uses the heat index of the previous 3 days, so the app must ask for it (feature set `hi_lags`). Random Forest settings were compared on the same folds that were used to choose them, so the scores are slightly optimistic; only the single test evaluation in Section 6.4 is unbiased.
