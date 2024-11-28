# Load required libraries
library(e1071)
library(gbm)
library(dplyr)
library(caret)
library(pROC)

# Read the data from the CSV file
data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/merged_output.csv", stringsAsFactors = TRUE)
# Split the data into features and labels
features <- data %>% select(-ID, -group)
labels <- data$group
# features_test <- test_data %>% select(-ID, -group)
# labels_test <- test_data$group

# Convert labels to numeric (0 for normal, 1 for tumor)
labels <- ifelse(labels == "Normal", 0, 1)

standard_scaler <- function(data) {
  # Centering the data (subtracting the mean)
  data_centered <- scale(data, center = TRUE, scale = TRUE)
  return(data_centered)
}
binarized_features <- standard_scaler(features)
# binarized_features_test <- standard_scaler(features_test)
# Split the data into training and test sets
set.seed(42)
train_index <- createDataPartition(labels, p = 0.8, list = FALSE)
X_train <- binarized_features[train_index, ] # Training data
y_train <- labels[train_index] # Training labels

X_test <- binarized_features[-train_index, ] # Testing data
y_test <- labels[-train_index]

# SVM Classifier
svm_model <- svm(x = X_train, y = y_train, kernel = "radial",type="C-classification")
svm_pred <- predict(svm_model, X_test)
print("SVM Pred: ")
print(svm_pred)
# Convert SVM predictions to class labels
svm_pred_labels <- svm_pred

# Evaluate SVM model
y_test_factor <- factor(y_test, levels = c(1, 0))
svm_pred_labels_factor <- factor(svm_pred_labels, levels = c(1, 0))
auc_svm <- auc(roc(y_test, as.numeric(svm_pred)-1))
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
gbm_model <- gbm(
  formula = y_train ~ .,
  data = as.data.frame(cbind(X_train, y_train)), # Combine X_train and y_train
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3
)

# Predict using the trained GBM model on X_test
gbm_pred <- predict(gbm_model,
  newdata = as.data.frame(X_test), # Provide X_test for prediction
  type = "response",
  n.trees = 100
)
gbm_pred_labels <- ifelse(gbm_pred > 0.5, 1, 0)
print(gbm_pred_labels)

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