import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# 🔹 Load dataset
df = pd.read_csv("iris.csv")

print("DATA COLUMNS:", df.columns)

# 🔹 Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 🔹 Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nBaseline model trained with accuracy: {acc:.2f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 🔹 Save model and metadata
joblib.dump(model, "model.joblib")

with open("metadata.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("✅ Model and metadata saved.")
