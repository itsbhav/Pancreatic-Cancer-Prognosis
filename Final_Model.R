# Load required libraries
library(e1071)
library(gbm)
library(dplyr)
library(caret)
library(pROC)
library(nnet)
library(xgboost)
library(lightgbm)

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

# NN model
model <- nnet(X_train, y_train, size = 11, entropy = TRUE, MaxNWts = 10000, maxit = 2000)
y_pred_proba <- predict(model, X_test, type = "raw")[, 1]
y_pred <- ifelse(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to class predictions
print(y_pred)

# SVM Classifier
svm_model <- svm(x = X_train, y = y_train, kernel = "radial",type="C-classification")
svm_pred <- predict(svm_model, X_test)
print(svm_pred)
# Convert SVM predictions to class labels
svm_pred_labels <- as.numeric(svm_pred)-1

# Gradient Boosting Classifier
purified_data<-data[, -which(names(data) %in% c("ID", "group"))]
purified_data$group<-ifelse(data$group == "Normal", 0, 1)
gbm_model <- gbm(formula = group ~ ., data = purified_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
gbm_pred <- predict(gbm_model, newdata = test_data[, -which(names(test_data) %in% c("ID", "group"))], type = "response", n.trees = 100)
gbm_pred_labels <- ifelse(gbm_pred > 0.5, 1, 0)
print(gbm_pred_labels)

# XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)
params <- list(objective = "binary:logistic", eval_metric = "logloss")
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 100)
xgb_pred <- predict(xgb_model, dtest)
xgb_pred_labels <- ifelse(xgb_pred > 0.5, 1, 0)
print(xgb_pred_labels)

# LightGBM
# Convert data to LightGBM dataset format
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
lgb_test <- lgb.Dataset(data = as.matrix(X_test), label = y_test, reference = lgb_train)

# Set parameters and train the LightGBM model
params <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_iterations = 100,
  learning_rate = 0.1
)

lgb_model <- lgb.train(
  params = params,
  data = lgb_train,
  valids = list(test = lgb_test),
  early_stopping_rounds = 10
)

# Predict using the LightGBM model
lgb_pred <- predict(lgb_model, newdata = as.matrix(X_test))
lgb_pred_labels <- ifelse(lgb_pred > 0.5, 1, 0)
print(lgb_pred_labels)

# Combine the predictions from all three models
combined_preds <- cbind(svm_pred_labels, gbm_pred_labels, y_pred,xgb_pred_labels,lgb_pred_labels)

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