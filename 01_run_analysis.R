# Open CustomerChurn.Rproj, run 00_setup.R once, then source this entire file.
# Synthetic educational data; no causal retention or financial-return claims.
required <- c("modeldata", "caret", "rpart", "randomForest", "pROC", "ggplot2", "e1071")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Run source('00_setup.R') first. Missing: ",
                          paste(missing, collapse = ", "))
if (!file.exists("CustomerChurn.Rproj")) {
  stop("Open CustomerChurn.Rproj before running this script.")
}
source("R/helpers.R")
source("tests/test_metrics.R")
dir.create("outputs", showWarnings = FALSE)
library(ggplot2)
seed <- 20261113L
threshold <- 0.5
capacity_fraction <- 0.2

# 1. Load and audit. Never install packages or authenticate in the analysis.
data_env <- new.env()
utils::data("mlc_churn", package = "modeldata", envir = data_env)
raw <- as.data.frame(data_env$mlc_churn)
stopifnot(nrow(raw) > 0L, "churn" %in% names(raw),
          !anyNA(raw$churn), all(as.character(raw$churn) %in% c("yes", "no")))
raw$churn <- factor(as.character(raw$churn), levels = c("yes", "no"))
raw$customer_row <- seq_len(nrow(raw))
# These are source row numbers, not real customer IDs.
write_table(data.frame(Column = names(raw), Missing = colSums(is.na(raw))),
            "missing_values.csv")
# Fail visibly if a later package release changes missingness. Do not silently
# impute all observations or use test data to estimate replacement values.
if (anyNA(raw)) stop("Missing data found. Review outputs/missing_values.csv before modeling.")

# Charges largely duplicate usage measures. Exclude them by a declared design
# choice, not by inspecting outcome correlations. Geographic codes are omitted
# for a compact behavior-based demonstration. No field selection uses test labels.
predictors <- c("account_length", "international_plan", "voice_mail_plan",
                "number_vmail_messages", "total_day_minutes", "total_day_calls",
                "total_eve_minutes", "total_eve_calls", "total_night_minutes",
                "total_night_calls", "total_intl_minutes", "total_intl_calls",
                "number_customer_service_calls")
stopifnot(all(predictors %in% names(raw)), !"churn" %in% predictors,
          !"customer_row" %in% predictors)
model_data <- raw[, c("churn", predictors), drop = FALSE]

# 2. Stratified split before exploratory outcome analysis or modeling.
set.seed(seed)
train_index <- as.integer(caret::createDataPartition(model_data$churn, p = 0.8,
                                                    list = FALSE))
test_index <- setdiff(seq_len(nrow(model_data)), train_index)
stopifnot(!length(intersect(train_index, test_index)),
          length(union(train_index, test_index)) == nrow(model_data))
training <- model_data[train_index, , drop = FALSE]
testing <- model_data[test_index, , drop = FALSE]
write_table(data.frame(CustomerRow = seq_len(nrow(raw)),
                       Split = ifelse(seq_len(nrow(raw)) %in% train_index,
                                      "training", "test")), "split_manifest.csv")
# Export a fingerprint and package versions to document the exact dataset used.
saveRDS(raw, file.path("outputs", "source_snapshot.rds"))
writeLines(paste("Source snapshot MD5:", tools::md5sum(file.path("outputs", "source_snapshot.rds"))),
           file.path("outputs", "source_fingerprint.txt"))

# 3. Training-only descriptive analysis: plan segments with denominators.
segments <- do.call(rbind, lapply(levels(training$international_plan), function(plan) {
  outcome <- training$churn[training$international_plan == plan]
  data.frame(InternationalPlan = plan, Records = length(outcome),
             Churners = sum(outcome == "yes"), ChurnRate = mean(outcome == "yes"))
}))
write_table(segments, "training_plan_segments.csv")
plot_segments <- ggplot(segments, aes(InternationalPlan, ChurnRate)) +
  geom_col(fill = "#287D8E", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%\nn = %s", 100 * ChurnRate, Records)),
            vjust = -0.3, size = 3.6) +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%"),
                     expand = expansion(mult = c(0, 0.22))) +
  labs(title = "Churn by international plan", subtitle = "Training records only; observed association",
       x = "International plan", y = "Observed churn rate") + theme_minimal(base_size = 12)
ggsave(file.path("outputs", "01_plan_segments.png"), plot_segments, width = 8, height = 5, dpi = 160)

# 4. Identical five-fold training resamples for all models.
# This is a declared 13-predictor design, with no learned preprocessing.
# Tree models need no scaling; GLM accepts these original units. If adding KNN
# or lasso later, estimate scaling inside caret's resampling, not on all data.
set.seed(seed + 1L)
folds <- caret::createFolds(training$churn, k = 5, returnTrain = TRUE)
control <- caret::trainControl(method = "cv", number = 5, index = folds,
                               classProbs = TRUE, summaryFunction = cv_summary,
                               savePredictions = "final", allowParallel = FALSE)
models <- list()
message("Training logistic regression...")
set.seed(seed + 2L)
models$Logistic <- caret::train(churn ~ ., data = training, method = "glm",
                               family = stats::binomial(), metric = "ROC", trControl = control)
message("Training decision tree...")
set.seed(seed + 3L)
models$DecisionTree <- caret::train(churn ~ ., data = training, method = "rpart",
                                   tuneGrid = data.frame(cp = c(0.001, 0.01, 0.05)),
                                   metric = "ROC", trControl = control)
message("Training random forest; this step may take several minutes...")
set.seed(seed + 4L)
models$RandomForest <- caret::train(churn ~ ., data = training, method = "rf",
                                   tuneGrid = data.frame(mtry = c(2, 4, 6)),
                                   ntree = 300, importance = TRUE,
                                   metric = "ROC", trControl = control)

# 5. Select model using CV ROC only, BEFORE making any test predictions.
cv_scores <- do.call(rbind, lapply(names(models), function(name) {
  best <- merge(models[[name]]$results, models[[name]]$bestTune)
  stopifnot(nrow(best) == 1L, is.finite(best$ROC))
  data.frame(Model = name, CV_ROC_AUC = best$ROC,
             CV_ROC_SD = best$ROCSD, CV_AveragePrecision = best$AveragePrecision)
}))
chosen <- cv_scores$Model[which.max(cv_scores$CV_ROC_AUC)]
write_table(cv_scores, "cross_validation_comparison.csv")
writeLines(c(paste("Selected model:", chosen), "Selection criterion: highest training CV ROC AUC",
             paste("Classification threshold fixed in advance:", threshold),
             paste("Contact fraction fixed in advance:", capacity_fraction)),
           file.path("outputs", "selection_protocol.txt"))
for (name in names(models)) {
  write_table(models[[name]]$results, paste0("cv_grid_", name, ".csv"))
}
saveRDS(list(models = models, chosen = chosen, predictors = predictors,
             threshold = threshold, capacity_fraction = capacity_fraction,
             folds = folds, seed = seed), file.path("outputs", "fitted_models.rds"))

# 6. Final held-out evaluation. No model/threshold changes based on this block.
prevalence <- mean(training$churn == "yes")
probabilities <- lapply(models, function(fit) {
  result <- predict(fit, newdata = testing, type = "prob")
  stopifnot("yes" %in% names(result))
  as.numeric(result$yes)
})
probabilities$Baseline <- rep(prevalence, nrow(testing))
metrics <- do.call(rbind, lapply(names(probabilities), function(name) {
  values <- classification_metrics(testing$churn, probabilities[[name]], threshold)
  data.frame(Model = name, SelectedByCV = name == chosen,
             Threshold = threshold, as.list(values), check.names = FALSE)
}))
write_table(metrics, "test_model_comparison.csv")
contact <- do.call(rbind, lapply(names(probabilities), function(name) {
  values <- capacity_metrics(testing$churn, probabilities[[name]], capacity_fraction)
  data.frame(Model = name, SelectedByCV = name == chosen,
             ContactFraction = capacity_fraction, as.list(values), check.names = FALSE)
}))
write_table(contact, "test_contact_capacity.csv")
predictions <- data.frame(CustomerRow = raw$customer_row[test_index], Actual = testing$churn,
                           as.data.frame(probabilities))
write_table(predictions, "test_predictions.csv")
# Outcome is included only in evaluation/export, never passed into a predictor matrix.

# 7. Charts show the declared comparisons, without selecting a new test winner.
roc_lines <- do.call(rbind, lapply(names(models), function(name) {
  obj <- roc_object(testing$churn, probabilities[[name]])
  data.frame(FalsePositiveRate = 1 - obj$specificities,
             TruePositiveRate = obj$sensitivities, Model = name)
}))
plot_roc <- ggplot(roc_lines, aes(FalsePositiveRate, TruePositiveRate, color = Model)) +
  geom_path(linewidth = 1) + geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey60") +
  coord_equal() + labs(title = "Held-out ROC comparison",
                       subtitle = paste("Model selected using training CV:", chosen),
                       x = "False-positive rate", y = "Recall (true-positive rate)") +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(file.path("outputs", "02_test_roc.png"), plot_roc, width = 8, height = 6, dpi = 160)

capacity_long <- rbind(data.frame(Model = contact$Model, Metric = "Precision", Value = contact$PrecisionAtCapacity),
                       data.frame(Model = contact$Model, Metric = "Recall", Value = contact$RecallAtCapacity))
plot_capacity <- ggplot(capacity_long, aes(Model, Value, fill = Metric)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_y_continuous(limits = c(0, 1), labels = function(x) paste0(round(100 * x), "%")) +
  scale_fill_manual(values = c(Precision = "#287D8E", Recall = "#D98324")) +
  labs(title = "What does a 20% contact list capture?",
       subtitle = "Ranked by predicted churn risk; held-out records",
       x = NULL, y = "Share") + theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(file.path("outputs", "03_contact_capacity.png"), plot_capacity, width = 8, height = 5, dpi = 160)

chosen_probability <- probabilities[[chosen]]
rank_index <- order(-chosen_probability, seq_along(chosen_probability))
gains <- data.frame(ContactShare = c(0, seq_len(nrow(testing)) / nrow(testing)),
                     ChurnerShare = c(0, cumsum(testing$churn[rank_index] == "yes") /
                                        sum(testing$churn == "yes")))
plot_gains <- ggplot(gains, aes(ContactShare, ChurnerShare)) +
  geom_line(color = "#287D8E", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
  geom_vline(xintercept = capacity_fraction, linetype = 3) +
  scale_x_continuous(labels = function(x) paste0(round(100 * x), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%")) +
  labs(title = "Cumulative capture of observed churners", subtitle = paste(chosen, "| dashed line: random selection"),
       x = "Share of test records contacted", y = "Share of observed churners captured") +
  theme_minimal(base_size = 12)
ggsave(file.path("outputs", "04_cumulative_gains.png"), plot_gains, width = 8, height = 5, dpi = 160)

importance <- as.data.frame(caret::varImp(models[[chosen]], scale = TRUE)$importance)
importance_score <- if ("Overall" %in% names(importance)) importance$Overall else
  if ("yes" %in% names(importance)) importance$yes else rowMeans(importance)
importance_df <- data.frame(Feature = rownames(importance), Importance = importance_score)
importance_df <- importance_df[order(-importance_df$Importance), , drop = FALSE]
write_table(importance_df, "selected_model_importance.csv")
top <- head(importance_df, 10)
plot_importance <- ggplot(top, aes(reorder(Feature, Importance), Importance)) +
  geom_col(fill = "#287D8E", width = 0.7) + coord_flip() +
  labs(title = "Predictive importance in the selected model", subtitle = chosen,
       x = NULL, y = "Relative model importance (maximum = 100)") +
  theme_minimal(base_size = 11)
ggsave(file.path("outputs", "05_feature_importance.png"), plot_importance, width = 9, height = 6, dpi = 160)

# 8. Generate a browser-readable report from actual computed values.
summary <- data.frame(Split = c("Training", "Test"),
                       Records = c(nrow(training), nrow(testing)),
                       Churners = c(sum(training$churn == "yes"), sum(testing$churn == "yes")),
                       ChurnRate = c(mean(training$churn == "yes"), mean(testing$churn == "yes")))
write_table(summary, "split_summary.csv")
selected_metrics <- metrics[metrics$Model == chosen, , drop = FALSE]
selected_contact <- contact[contact$Model == chosen, , drop = FALSE]
report <- paste0(
  "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
  "<meta name='viewport' content='width=device-width,initial-scale=1'>",
  "<title>Customer Churn | Mark Samuel</title><style>",
  "body{font:16px/1.6 system-ui,sans-serif;color:#243647;background:#f4f7f9;margin:0}",
  "main{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:36px;line-height:1.15}",
  "h2{margin-top:36px}img{display:block;width:100%;max-width:900px;background:white;margin:20px auto}",
  "table{border-collapse:collapse;background:white;width:100%;font-size:14px}",
  "th,td{padding:10px 12px;border-bottom:1px solid #dde4e8;text-align:left;white-space:nowrap}",
  "th{background:#e5f0f2}.table-wrap{overflow-x:auto}p{max-width:85ch}",
  ".tag{color:#287D8E;font-weight:700;letter-spacing:.08em;text-transform:uppercase}",
  "</style></head><body><main><p class='tag'>R portfolio / predictive analytics</p>",
  "<h1>Customer Churn Prediction<br>&amp; Retention Prioritization</h1>",
  "<p>Mark Samuel | Generated ", html_escape(Sys.Date()),
  " | Artificial educational telecom data from modeldata::mlc_churn.</p>",
  "<p><strong>Question:</strong> How well can usage and account information rank churn risk, ",
  "and what would a contact list covering 20% of records capture?</p>",
  "<h2>Evaluation design</h2><p>Stratified 80/20 split; identical five-fold training resamples; ",
  "hyperparameters and model chosen by training CV ROC AUC. Churn=yes is positive. ",
  "The classification threshold of 0.5 and contact fraction of 20% were fixed in advance. ",
  "The test set is used for final reporting, not selection.</p>", html_table(summary),
  "<h2>Model selected before test evaluation: ", html_escape(chosen), "</h2>",
  html_table(cv_scores), "<h2>Held-out performance</h2>", html_table(metrics),
  "<p>Accuracy can hide missed churners. Recall measures how many observed churners are identified; ",
  "precision measures how often flags are correct. ROC AUC measures ranking across thresholds. ",
  "Average precision summarizes precision-recall performance using tied-score groups. ",
  "Brier score measures squared probability error (lower is better). An undefined precision is ",
  "shown as NA when a model predicts no positives.</p>",
  "<img src='02_test_roc.png' alt='ROC curves for three models on the test set'>",
  "<h2>Contact capacity: a hypothetical 20% list</h2>", html_table(contact),
  "<p>For the selected model, the list captures ", sprintf("%.1f%%", 100 * selected_contact$RecallAtCapacity),
  " of observed churners at ", sprintf("%.1f%%", 100 * selected_contact$PrecisionAtCapacity),
  " precision. Lift compares that precision with the test churn rate. ",
  "This describes historical ranking, not customers saved or financial return. ",
  "Baseline probabilities equal training prevalence; tied ranks use source row order, ",
  "so its capacity result is an arbitrary tied ordering, not a targeted strategy. ",
  "The expected random-selection lift is 1.</p>",
  "<img src='03_contact_capacity.png' alt='Precision and recall at twenty percent contact capacity'>",
  "<img src='04_cumulative_gains.png' alt='Cumulative capture curve for selected model'>",
  "<h2>Patterns for investigation</h2>",
  "<img src='01_plan_segments.png' alt='Training churn rates by international plan with sample sizes'>",
  "<img src='05_feature_importance.png' alt='Ten highest model importance values'>",
  "<p>Plan differences and model importance describe associations, not reasons customers leave. ",
  "Importance definitions differ by model and can be affected by correlated predictors. ",
  "A retention experiment would be needed to measure whether an offer changes behavior.</p>",
  "<h2>Limits and next steps</h2><p>This is an artificial benchmark with no observation dates, ",
  "churn horizon, retention interventions, contact costs or response outcomes. ",
  "A random holdout does not establish future performance. Before operational use, validate ",
  "feature availability at prediction time, use a temporal holdout, assess calibration and uncertainty, ",
  "and run a randomized retention pilot. CV results used for selection can be optimistic; ",
  "the held-out result is the assessment for this workflow.</p>",
  "<p>Source: <a href='https://modeldata.tidymodels.org/reference/mlc_churn.html'>modeldata documentation</a>. ",
  "Methods, package versions, split manifest and outputs are recorded alongside this report.</p>",
  "</main></body></html>")
writeLines(report, file.path("outputs", "report.html"), useBytes = TRUE)
writeLines(capture.output(sessionInfo()), file.path("outputs", "session_info.txt"))
write_table(data.frame(Package = required,
                       Version = vapply(required, function(x) as.character(utils::packageVersion(x)), character(1))),
            "package_versions.csv")
writeLines(c("R execution completed without an assertion failure.",
             paste("Completed:", Sys.time()), paste("Selected model:", chosen),
             paste("Seed:", seed), paste("Training rows:", nrow(training)),
             paste("Test rows:", nrow(testing)),
             "Metric checks, split disjointness and finite CV scores passed."),
           file.path("outputs", "run_status.txt"))
print(cv_scores)
print(metrics)
message("Finished. Open outputs/report.html in a browser.")
