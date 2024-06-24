# Load required libraries
library(e1071)
library(gbm)
library(dplyr)
library(caret)
library(pROC)
library(nnet)

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
X_train <-binarized_features
X_test <- binarized_features_test
y_train <- labels
y_test <- labels_test

# SVM Classifier
svm_model <- svm(x = X_train, y = y_train, kernel = "linear")
svm_pred <- predict(svm_model, X_test)

# Convert SVM predictions to class labels
svm_pred_labels <- ifelse(svm_pred > 0, 1, 0)


# Gradient Boosting Classifier
gbm_model <- gbm(formula = group ~ ., data = train_data[, -which(names(train_data) == "ID")], distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
gbm_pred <- predict(gbm_model, newdata = test_data[, -which(names(test_data) %in% c("ID", "group"))], type = "response", n.trees = 100)
gbm_pred_labels <- ifelse(gbm_pred > 0.5, 1, 0)


# Create the neural network model using nnet package

model <- nnet(X_train, y_train, size = 5, entropy = TRUE, MaxNWts = 10000, maxit = 1000)
# Get predicted values
y_pred <- ifelse(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to class predictions

# Combine the predictions from all three models
combined_preds <- cbind(svm_pred_labels, gbm_pred_labels, y_pred)

# Function to get the majority vote
majority_vote <- function(x) {
  return(as.numeric(names(sort(-table(x)))[1]))
}

# Get the final prediction by taking the majority vote
final_preds <- apply(combined_preds, 1, majority_vote)
print(final_preds)
print(y_test)
# Calculate the overall accuracy
y_test_factor <- factor(y_test, levels = c(1, 0))
final_preds_factor <- factor(final_preds, levels = c(1, 0))
precision <- posPredValue(y_test_factor, final_preds_factor)
recall <- sensitivity(y_test_factor, final_preds_factor)
specificity <- specificity(y_test_factor, final_preds_factor)
f1_score <- 2 * (precision * recall) / (precision + recall)
overall_accuracy <- mean(final_preds == y_test)
print(paste("Overall Accuracy:", overall_accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("Specificity:", specificity))
print(paste("F1-score:", f1_score))
conf_combined<-confusionMatrix(data=final_preds_factor,reference = y_test_factor)
# print(paste("AUC",auc(roc(y_test,final_preds))))