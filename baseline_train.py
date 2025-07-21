import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os # Import the os module

# Define paths for data and model/metadata
DATA_PATH = "data/base.csv"  # Changed from "iris.csv" to "data/base.csv"
MODEL_PATH = "models/model.joblib" # Store in 'models' directory
METADATA_PATH = "models/metadata.json" # Store in 'models' directory

# Ensure the 'models' directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 🔹 Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found. Please ensure it exists.")
    exit() # Exit if the file is not found

print("DATA COLUMNS:", df.columns)

# 🔹 Features & Target
# IMPORTANT: Adjust 'target' column name if your base.csv has a different one.
# For example, if your target is 'label', change to df.drop("label", axis=1) and y = df["label"]
# You might need to inspect base.csv to confirm the target column.
try:
    X = df.drop("target", axis=1)
    y = df["target"]
except KeyError:
    print("Error: 'target' column not found. Please check your dataset and update the target column name if necessary.")
    exit() # Exit if the target column is not found


# 🔹 Train-test split
# Ensure 'y' has at least 2 unique classes for classification
if y.nunique() < 2:
    print("Error: Target variable has less than 2 unique classes. Cannot perform classification.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Added stratify=y for balanced splits

# 🔹 Model Training
model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
model.fit(X_train, y_train)

# 🔹 Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nBaseline model trained with accuracy: {acc:.2f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 🔹 Save model and metadata
joblib.dump(model, MODEL_PATH)

with open(METADATA_PATH, "w") as f:
    json.dump({"accuracy": acc}, f)

print(f"✅ Model saved to: {MODEL_PATH}")
print(f"✅ Metadata saved to: {METADATA_PATH}")