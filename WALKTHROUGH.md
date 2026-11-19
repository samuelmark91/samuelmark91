# Walkthrough and interview notes

## Explain the project

"I developed a coursework churn analysis into a reproducible model-comparison project. I corrected target leakage, compared three models on identical training folds, evaluated a held-out test set, and translated risk scores into a hypothetical contact-capacity analysis. The data are artificial, so I present it as a methodological demonstration."

The completed run selected random forest by training CV. Its top-200 test-record list captured 118 of 141 churners: 83.7% recall, 59.0% precision and 4.18 times the test churn prevalence. At the separate 0.5 cutoff, recall was 73.0% and precision 92.8%.

## Understand each stage

1. **Audit:** check labels, missing values and required columns. Stop on unexpected source changes rather than silently repairing them.
2. **Split:** stratify on churn to preserve class representation. The source row number tracks records but is excluded from predictors.
3. **Explore training data:** describe segment rates with sample sizes. Avoid claiming that a plan causes churn.
4. **Fit:** GLM provides a simple benchmark; a tree captures simple nonlinear splits; a random forest combines trees. Do not assume the most complex model wins.
5. **Validate:** shared CV folds support a fair comparison. Tune and select on training data; assess on the held-out test data.
6. **Interpret:** precision and recall address different errors. A 20% list expresses limited outreach capacity without inventing revenue or treatment effects.
7. **Reproduce:** record seeds, indices, package versions, source fingerprint and outputs.

## Questions to be ready for

- What was wrong with putting churn in the original lasso input matrix?
- Why does a high accuracy score not guarantee useful churn detection?
- Why access probabilities by the name `yes` instead of column position?
- Why remove charges while keeping minutes, and what alternative feature sets could be tested within training CV?
- What is the difference between ranking likely churners and proving that a retention offer works?
- Why is a random split insufficient to establish future performance?
- What would you change if the dataset had missing values or repeated records per customer?

## Results to explain

Record the CV-selected model, its held-out ROC AUC and average precision, its precision/recall at 0.5, and the share of observed churners captured in the top 20% list. Compare with the baseline and explain limitations. Never fill these values from the assignments' old results: the predictor set and validation procedure changed.
