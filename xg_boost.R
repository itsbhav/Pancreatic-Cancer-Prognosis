# install.packages("xgboost")
library(dplyr)
library(caret)
library(pROC)
library(xgboost)
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
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)
params <- list(objective = "binary:logistic", eval_metric = "logloss")
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 100)
xgb_pred <- predict(xgb_model, dtest)
xgb_pred_labels <- ifelse(xgb_pred > 0.5, 1, 0)
print(xgb_pred_labels)
# Evaluate the model using the ROC curve and AUC
y_test_factor <- factor(y_test, levels = c(1, 0))
xgb_pred_labels_factor <- factor(xgb_pred_labels, levels = c(1, 0))
conf_xg <- confusionMatrix(data = xgb_pred_labels_factor, reference = y_test_factor)
auc_xg <- auc(roc(y_test, xgb_pred))
print(paste("Area under the ROC curve:", auc_xg))