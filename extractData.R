library(caret)
library(dplyr)

# Read the data from the CSV file
data <- read.csv("C:/Users/bhave/OneDrive/Desktop/Minor/merged_output.csv", stringsAsFactors = TRUE)

# Split the data into features and labels
features <- data %>% select(-ID, -group)
labels <- data$group # Keep the original labels

# Standardize the features
standard_scaler <- function(data) {
    data_centered <- scale(data, center = TRUE, scale = TRUE)
    return(data_centered)
}
binarized_features <- standard_scaler(features)

# Split the data into training and test sets
set.seed(42)
train_index <- createDataPartition(labels, p = 0.8, list = FALSE)

# Combine IDs with the training and testing features
X_train <- binarized_features[train_index, ] # Training data
y_train <- labels[train_index] # Original labels for training
ID_train <- data$ID[train_index] # IDs for training

X_test <- binarized_features[-train_index, ] # Testing data
y_test <- labels[-train_index] # Original labels for testing
ID_test <- data$ID[-train_index] # IDs for testing

# Create data frames including IDs
X_train_with_ids <- data.frame(ID = ID_train, X_train) # Combine ID with training features
X_test_with_ids <- data.frame(ID = ID_test, X_test) # Combine ID with testing features
y_train <- data.frame(ID = ID_train, y_train)
y_test <- data.frame(ID = ID_test, y_test)
# Save to CSV
write.csv(X_train_with_ids, "X_train.csv", row.names = FALSE)
write.csv(y_train, "y_train.csv", row.names = FALSE) # Save original labels
write.csv(X_test_with_ids, "X_test.csv", row.names = FALSE)
write.csv(y_test, "y_test.csv", row.names = FALSE) # Save original labels
