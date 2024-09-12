# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from transformers import ViTForImageClassification, SwinForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.optim as optim

# Load the data
data = pd.read_csv("C:/Users/bhave/OneDrive/Desktop/Minor/new_file1.csv")
test_data = pd.read_csv("C:/Users/bhave/OneDrive/Desktop/Minor/test_data_new.csv")

# Data preparation
features = data.drop(columns=['ID', 'group'])
labels = data['group'].map({'Normal': 0, 'Tumor': 1})
features_test = test_data.drop(columns=['ID', 'group'])
labels_test = test_data['group'].map({'Normal': 0, 'Tumor': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define a simple ViT model for tabular data (hypothetical, as ViT is designed for images)
class SimpleViT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleViT, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return self.softmax(x)

# Instantiate and train the model
input_dim = X_train.shape[1]
hidden_dim = 128  # Example dimension, tune accordingly
num_classes = 2  # Binary classification

model = SimpleViT(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
