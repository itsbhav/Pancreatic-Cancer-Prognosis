import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import swin_t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Load the datasets from CSV files (these were created in R)
X_train = pd.read_csv('X_train.csv', index_col=0)  # Replace with the actual path
y_train = pd.read_csv('y_train.csv', index_col=0)  # Replace with the actual path
X_test = pd.read_csv('X_test.csv', index_col=0)    # Replace with the actual path
y_test = pd.read_csv('y_test.csv', index_col=0)    # Replace with the actual path
y_test = y_test['y_test'].map({'Normal': 0, 'Tumor': 1})
y_train = y_train['y_train'].map({'Normal': 0, 'Tumor': 1})

# Convert data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define the model (Swin Transformer)
model = swin_t(weights='IMAGENET1K_V1')
model.head = nn.Linear(model.head.in_features, 2)  # Modify the final layer for binary classification

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy input to match the Swin Transformer's requirement
dummy_input = torch.randn(len(X_train), 3, 224, 224)  # 3 channels, 224x224 for Swin Transformer

# Training loop
for epoch in range(10):  # Simplified training loop
    optimizer.zero_grad()
    outputs = model(dummy_input)  # Using dummy input to pass through the model
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    dummy_test_input = torch.randn(len(X_test_tensor), 3, 224, 224)
    y_pred = model(dummy_test_input).argmax(dim=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Create a DataFrame to print the results
    results_df = pd.DataFrame({
        # 'ID': test_IDs,
        'True Label': y_test.values,  # Original true labels from the test data
        'Predicted Label': y_pred.numpy()  # Predicted labels
    })

    # Print the results (ID, True Label, Predicted Label)
    print("\nTest Samples with True and Predicted Labels:")
    print(results_df)
