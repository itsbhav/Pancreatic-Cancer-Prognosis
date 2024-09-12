import autokeras as ak
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("C:/Users/bhave/OneDrive/Desktop/Minor/new_file1.csv")
X = data.drop(columns=['ID', 'group'])
y = data['group'].map({'Normal': 0, 'Tumor': 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

clf = ak.StructuredDataClassifier(max_trials=10, overwrite=True)
clf.fit(X_train, y_train, epochs=10)

accuracy = clf.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

predicted_y = clf.predict(X_test)
