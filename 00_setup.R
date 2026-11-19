# Run once in the RStudio Console after opening CustomerChurn.Rproj.
required <- c("modeldata", "caret", "rpart", "randomForest", "pROC", "ggplot2", "e1071")
missing <- setdiff(required, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
still_missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing)) {
  stop("Packages still missing: ", paste(still_missing, collapse = ", "))
}
message("Setup complete. Next run: source('01_run_analysis.R')")
