import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import swin_t
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv("C:/Users/bhave/OneDrive/Desktop/Minor/new_file1.csv")
test_data = pd.read_csv("C:/Users/bhave/OneDrive/Desktop/Minor/test_data_new.csv")

# Prepare data
X = data.drop(columns=['ID', 'group'])
y = data['group'].map({'Normal': 0, 'Tumor': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define the model (Swin Transformer)
model = swin_t(weights='IMAGENET1K_V1')
model.head = nn.Linear(model.head.in_features, 2)  # Modify the final layer for binary classification

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy input to match the Swin Transformer's requirement (we should have images, but here is an adaptation)
dummy_input = torch.randn(len(X_train), 3, 224, 224)  # Swin Transformer requires 3 channels, 224x224

# Train loop
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
    dummy_test_input = torch.randn(len(X_test), 3, 224, 224)
    y_pred = model(dummy_test_input).argmax(dim=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
