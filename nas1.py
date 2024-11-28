import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the datasets from CSV files (these were created in R)
X_train = pd.read_csv('X_train.csv', index_col=0)  # Replace with the actual path
y_train = pd.read_csv('y_train.csv', index_col=0)  # Replace with the actual path
X_test = pd.read_csv('X_test.csv', index_col=0)    # Replace with the actual path
y_test = pd.read_csv('y_test.csv', index_col=0)    # Replace with the actual path
y_test = y_test['y_test'].map({'Normal': 0, 'Tumor': 1})
y_train = y_train['y_train'].map({'Normal': 0, 'Tumor': 1})
# Convert data to numpy arrays (TensorFlow requires NumPy arrays)
X_train = X_train.values
y_train = y_train.values.flatten()  # Flatten in case y_train is a column vector
X_test = X_test.values
y_test = y_test.values.flatten()  # Flatten in case y_test is a column vector
print(y_train)
# Define normalization layer
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train)  # Compute the mean and variance of the training data

# Build a simple neural network model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input layer matches the number of features in X
    normalizer,  # Normalization layer (preprocessing)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions
predicted_y = (model.predict(X_test) > 0.5).astype(int)

# Create a DataFrame for results
result_df = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': ['Normal' if pred[0] == 0 else 'Tumor' for pred in predicted_y]
})

# Print the result DataFrame
print(result_df)
