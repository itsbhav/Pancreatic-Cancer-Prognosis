# Load required libraries
# install.packages("dplyr")
# install.packages("caret")
# install.packages("pROC")
# install.packages('glmnet')
# Load required packages
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
# Split the data into training and test sets

# Create the neural network model using nnet package
library(glmnet)
model <- glm(y_train ~ ., family = binomial(link = "logit"), data = as.data.frame(X_train))

# Evaluate the model using the ROC curve and AUC
y_pred_proba <- predict(model, newdata=as.data.frame(X_test), type = "response")
roc_obj <- roc(y_test, y_pred_proba)
auc_value <- auc(roc_obj)
print(paste("Area under the ROC curve:", auc_value))

# Get predicted values
y_pred <- ifelse(y_pred_proba > 0.5, 1, 0) # Convert probabilities to class predictions
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
test_samples <- data[-train_index, ] # Extract test samples from the original dataset (excluding training data)
test_samples$true_label <- y_test # Add true label column (already in binary 0 and 1)
test_samples$predicted_label <- y_pred

print("Test samples with true labels and predicted labels:")
print(test_samples[, c("ID", "group", "true_label", "predicted_label")])
conf_neural <- confusionMatrix(data = y_pred_factor, reference = y_test_factor)
print(conf_neural)
