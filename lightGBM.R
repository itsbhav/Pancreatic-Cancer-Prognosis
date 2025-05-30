# Install required packages if not already installed
# install.packages("lightgbm")

# Load necessary libraries
library(dplyr)
library(caret)
library(pROC)
library(lightgbm)

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

# Evaluate the model
y_test_factor <- factor(y_test, levels = c(1, 0))
lgb_pred_labels_factor <- factor(lgb_pred_labels, levels = c(1, 0))
conf_lgb <- confusionMatrix(data = lgb_pred_labels_factor, reference = y_test_factor)
auc_lgb <- auc(roc(y_test, lgb_pred))
print(paste("Area under the ROC curve:", auc_lgb))
lgb_accuracy <- mean(lgb_pred_labels == y_test)
print(paste("Accuracy is: ", lgb_accuracy))
precision_lgb <- posPredValue(y_test_factor, lgb_pred_labels_factor)
print(paste("Precision is: ", precision_lgb))
specificity_lgb <- specificity(y_test_factor, lgb_pred_labels_factor)
print(paste("Specificity is: ", specificity_lgb))
recall_lgb <- sensitivity(y_test_factor, lgb_pred_labels_factor)
f1_lgb <- 2 * (precision_lgb * recall_lgb) / (precision_lgb + recall_lgb)
print(paste("F1 score is: ", f1_lgb))