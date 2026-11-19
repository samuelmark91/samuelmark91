# Validation record

## Completed R run

The owner supplied scripts, all generated figures/tables and run metadata. `outputs/run_status.txt` records successful completion on **November 13, 2026 at 13:54 America/Chicago** with 4,001 training rows, 999 test rows, seed 20261113 and RandomForest selected. `outputs/session_info.txt` identifies **R 4.4.1 on Windows 11**. Exact package versions are in `outputs/package_versions.csv`.

The initial strict `identical` check on the class-label vector stopped the first attempt. The saved helper uses a set comparison while scoring explicitly named yes probabilities. The subsequent run completed. The submitted tests are the original known-answer tests; the later proposed additional class-order regression test was not present and is not claimed to have run.

The reported ggplot2/caret build-version warnings did not stop the run. They are compatibility warnings, not evidence of failed model fitting.

## Independent review of the saved outputs

The reviewer used the submitted test predictions and Python/scikit-learn to recompute, for all three models plus baseline:

- Confusion-matrix counts, accuracy, precision, recall, specificity and F1.
- ROC AUC, non-interpolated average precision and Brier score.
- Top-20% selection size, captured churners, precision, recall and lift, using stable tie ordering.

All recomputed values match the supplied CSVs within numerical tolerance. Precision remains undefined when no positives are predicted. The manifest contains 5,000 unique row indices, with 4,001 training and 999 test rows; prediction row indices exactly match test membership. The highest training CV ROC AUC and saved selection protocol both identify RandomForest.

All five PNG charts were visually inspected and their report image references resolve. The owner previously opened the browser report successfully. The reviewer checked report references and chart files, but did not independently rerun R or retrain the models in this environment.

## Publication edits

The source scripts, submitted metric tests, CSV values, HTML report and PNG charts are preserved. README, output index, validation and walkthrough text were updated to reflect execution and actual findings. RStudio session files and large generated RDS objects are excluded from the publication ZIP. This packaging step does not constitute a new R execution.

## Remaining limitations

No temporal validation, repeated holdout, calibration assessment or uncertainty intervals are included. Neither predictive importance nor retrospective risk ranking measures a causal retention benefit. Package versions and seed are recorded, but a dependency lockfile is not supplied.
