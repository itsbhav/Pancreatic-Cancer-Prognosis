# install.packages("xgboost")
library(dplyr)
library(caret)
library(pROC)
library(xgboost)
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
print(conf_xg)