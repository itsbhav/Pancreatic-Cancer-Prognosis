# Load required libraries
library(dplyr)
library(caret)
library(pROC)
library(rpart)

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
X_train <- binarized_features
X_test <- binarized_features_test
y_train <- labels
y_test <- labels_test

# Create the CART model using rpart package
cart_model <- rpart(y_train ~ ., data = as.data.frame(X_train), method = "class")

# Evaluate the model using the ROC curve and AUC
y_pred_proba_cart <- predict(cart_model, newdata = as.data.frame(X_test), type = "prob")[, 2]
roc_obj_cart <- roc(y_test, y_pred_proba_cart)
auc_value_cart <- auc(roc_obj_cart)
print(paste("Area under the ROC curve (CART):", auc_value_cart))

# Get predicted values
y_pred_cart <- predict(cart_model, newdata = as.data.frame(X_test), type = "class")
y_pred_cart <- as.numeric(as.character(y_pred_cart)) # Convert factor to numeric
print(y_pred_cart)

# Calculate accuracy and other metrics
accuracy_cart <- mean(y_pred_cart == y_test)
precision_cart <- posPredValue(factor(y_test), factor(y_pred_cart, levels = c(0, 1)))
recall_cart <- sensitivity(factor(y_test), factor(y_pred_cart, levels = c(0, 1)))
specificity_cart <- specificity(factor(y_test), factor(y_pred_cart, levels = c(0, 1)))
f1_score_cart <- 2 * (precision_cart * recall_cart) / (precision_cart + recall_cart)

print(paste("Accuracy (CART):", accuracy_cart))
print(paste("Precision (CART):", precision_cart))
print(paste("Recall (CART):", recall_cart))
print(paste("Specificity (CART):", specificity_cart))
print(paste("F1-score (CART):", f1_score_cart))

# Print test samples with true labels and predicted labels
test_samples <- test_data # Extract test samples from original data
test_samples$true_label <- ifelse(test_samples$group == "Normal", 0, 1) # Add true label column
test_samples$predicted_label <- y_pred_cart # Add predicted label column

print("Test samples with true labels and predicted labels (CART):")
print(test_samples[, c("ID", "group", "true_label", "predicted_label")])

# Confusion Matrix
conf_cart <- confusionMatrix(data = factor(y_pred_cart, levels = c(0, 1)), reference = factor(y_test, levels = c(0, 1)))
print(conf_cart)
