# Load required libraries
# install.packages("dplyr")
# install.packages("caret")
# install.packages("pROC")
# Load required packages
library(dplyr)
library(caret)
library(pROC)

# Read the data from the CSV file
data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/new_file1.csv", stringsAsFactors = TRUE)
test_data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/test_data_new.csv", stringsAsFactors = TRUE)

# Split the data into features and labels
features <- data %>% select(-ID, -group)
labels <- data$group
features_test <- test_data %>% select(-ID, -group)
labels_test <- test_data$group

# Convert labels to numeric (0 for normal, 1 for tumor)
labels <- ifelse(labels == "Normal", 0, 1)
labels_test <- ifelse(labels_test == "Normal", 0, 1)

# Calculate the median value for each gene
# median_values <- apply(features, 2, median)
# median_values_test <- apply(features_test, 2, median)

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

# Create the neural network model using nnet package
library(nnet)
model <- nnet(X_train, y_train, size = 11, entropy = TRUE, MaxNWts = 10000, maxit = 2000)

# Evaluate the model using the ROC curve and AUC
y_pred_proba <- predict(model, X_test, type = "raw")[, 1]
# roc_obj <- roc(y_test, y_pred_proba)
# auc_value <- auc(roc_obj)
# print(paste("Area under the ROC curve:", auc_value))

# Get predicted values
y_pred <- ifelse(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to class predictions
print(y_pred)
# Calculate accuracy
y_test_factor <- factor(y_test, levels = c(1, 0))
y_pred_factor <- factor(y_pred, levels = c(1, 0))
precision <- posPredValue(y_test_factor, y_pred_factor)
recall <- sensitivity(y_test_factor, y_pred_factor)
specificity <- specificity(y_test_factor, y_pred_factor)
f1_score <- 2 * (precision * recall) / (precision + recall)
accuracy <- sum(y_pred == y_test) / length(y_test)
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("Specificity:", specificity))
print(paste("F1-score:", f1_score))

# Print test samples with true labels and predicted labels
test_samples <- test_data  # Extract test samples from original data
test_samples$true_label <- ifelse(test_samples$group == "Normal", 0, 1)  # Add true label column
test_samples$predicted_label <- y_pred  # Add predicted label column

print("Test samples with true labels and predicted labels:")
print(test_samples[, c("ID", "group", "true_label", "predicted_label")])
conf_neural<-confusionMatrix(data=y_pred_factor,reference = y_test_factor)
