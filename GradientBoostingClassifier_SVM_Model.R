# Load required libraries
library(e1071)
library(gbm)
library(dplyr)
library(caret)
library(pROC)

# Read the data from the CSV file
data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/feature_genes.csv", stringsAsFactors = TRUE)
test_data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/test_data.csv", stringsAsFactors = TRUE)

# Split the data into features and labels
features <- data %>% select(-ID, -group)
labels <- data$group
features_test <- test_data %>% select(-ID, -group)
labels_test <- test_data$group

# Convert labels to numeric (0 for normal, 1 for tumor)
labels <- ifelse(labels == "Normal", 0, 1)
labels_test <- ifelse(labels_test == "Normal", 0, 1)

# Calculate the median value for each gene
median_values <- apply(features, 2, median)
median_values_test <- apply(features_test, 2, median)

# Binarize the gene expression values based on the median value
binarized_features <- t(apply(features, 1, function(row) {
  binarized_row <- ifelse(row < median(row), 0, 1)
  return(binarized_row)
}))
binarized_features_test <- t(apply(features_test, 1, function(row) {
  binarized_row <- ifelse(row < median(row), 0, 1)
  return(binarized_row)
}))

# Split the data into training and test sets
set.seed(42)
train_index <- createDataPartition(labels, p = 0.8, list = FALSE)
X_train <- binarized_features
X_test <- binarized_features_test
y_train <- labels
y_test <- labels_test

# SVM Classifier
svm_model <- svm(x = X_train, y = y_train, kernel = "linear")
svm_pred <- predict(svm_model, X_test)

# Convert SVM predictions to class labels
svm_pred_labels <- ifelse(svm_pred > 0, 1, 0)

# Evaluate SVM model
y_test_factor <- factor(y_test, levels = c(1, 0))
svm_pred_labels_factor <- factor(svm_pred_labels, levels = c(1, 0))
auc_svm <- auc(roc(y_test, svm_pred))
svm_accuracy <- mean(svm_pred_labels == y_test)
precision_svm <- posPredValue(y_test_factor, svm_pred_labels_factor)
recall_svm <- sensitivity(y_test_factor, svm_pred_labels_factor)
specificity_svm <- specificity(y_test_factor, svm_pred_labels_factor)
f1_svm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)
print("SVM Metrics:\n")
cat("AUC:", auc_svm, "\n")
cat("Precision:", precision_svm, "\n")
cat("Recall:", recall_svm, "\n")
cat("Specificity:", specificity_svm, "\n")
cat("F1-score:", f1_svm, "\n")
print(paste("SVM Accuracy:", svm_accuracy))
conf_svm<-confusionMatrix(data=svm_pred_labels_factor,reference = y_test_factor)

# Gradient Boosting Classifier
gbm_model <- gbm(formula = group ~ ., data = train_data[, -which(names(train_data) == "ID")], distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
gbm_pred <- predict(gbm_model, newdata = test_data[, -which(names(test_data) %in% c("ID", "group"))], type = "response", n.trees = 100)
gbm_pred_labels <- ifelse(gbm_pred > 0.5, 1, 0)


# Evaluate Gradient Boosting model
y_test_factor <- factor(y_test, levels = c(1, 0))
gbm_pred_labels_factor <- factor(gbm_pred_labels, levels = c(1, 0))
auc_gbm <- auc(roc(y_test, gbm_pred))
gbm_accuracy <- mean(gbm_pred_labels == y_test)
precision_gbm <- posPredValue(y_test_factor, gbm_pred_labels_factor)
recall_gbm <- sensitivity(y_test_factor, gbm_pred_labels_factor)
specificity_gbm <- specificity(y_test_factor, gbm_pred_labels_factor)
f1_gbm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)
print("GBM Metrics:\n")
cat("AUC:", auc_gbm, "\n")
cat("Precision:", precision_gbm, "\n")
cat("Recall:", recall_gbm, "\n")
cat("Specificity:", specificity_gbm, "\n")
cat("F1-score:", f1_gbm, "\n")
print(paste("GBM Accuracy:", gbm_accuracy))
conf_gbc<-confusionMatrix(data=gbm_pred_labels_factor,reference = y_test_factor)