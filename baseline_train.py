import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import numpy as np

# Define paths for data and model/metadata
DATA_PATH = "data/base.csv"
MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"

# Ensure the 'models' directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 🔹 Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found. Please ensure it exists.")
    exit()

print("DATA COLUMNS:", df.columns)

# 🔹 Features & Target
try:
    X = df.drop("target", axis=1)
    y = df["target"]
except KeyError:
    print("Error: 'target' column not found. Please check your dataset and update the target column name if necessary.")
    exit()


# 🔹 Train-test split
if y.nunique() < 2:
    print("Error: Target variable has less than 2 unique classes. Cannot perform classification.")
    exit()

min_samples_per_class = y.value_counts().min()
num_classes = y.nunique()

current_test_size = 0.2
adjusted_test_size = current_test_size

if min_samples_per_class < 1 / (1 - current_test_size) or min_samples_per_class < 1 / current_test_size:
    print(f"Warning: Dataset is small ({len(df)} samples, smallest class has {min_samples_per_class} samples).")
    if len(df) >= 10 and num_classes < len(df) * 0.4:
        adjusted_test_size = 0.3
        if min_samples_per_class < 1 / (1 - adjusted_test_size) or min_samples_per_class < 1 / adjusted_test_size:
             adjusted_test_size = 0.4
             if min_samples_per_class < 1 / (1 - adjusted_test_size) or min_samples_per_class < 1 / adjusted_test_size:
                 print("Cannot guarantee stratification with current dataset size. Increasing test_size to 0.5.")
                 adjusted_test_size = 0.5
    else:
        print("Dataset too small for stratification, or very few samples in smallest class. Proceeding without stratification.")
        # Original training block for extremely small datasets without stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # Fixed test_size here too
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Use output_dict=True to get a dictionary of metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # zero_division=0 to handle warnings

        # Extract relevant metrics
        metrics_to_save = {
            "accuracy": acc,
            "precision_macro": report['macro avg']['precision'],
            "recall_macro": report['macro avg']['recall'],
            "f1_macro": report['macro avg']['f1-score'],
            "precision_weighted": report['weighted avg']['precision'],
            "recall_weighted": report['weighted avg']['recall'],
            "f1_weighted": report['weighted avg']['f1-score'],
            "classification_report": report # Save the full report for completeness
        }

        joblib.dump(model, MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(metrics_to_save, f, indent=4) # indent for readability

        print(f"\nBaseline model trained with accuracy: {acc:.2f}\n")
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred, zero_division=0)) # print full report
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Metadata saved to: {METADATA_PATH}")
        exit() # Exit after non-stratified training

print(f"Using test_size={adjusted_test_size} for train_test_split.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=adjusted_test_size, random_state=42, stratify=y)


# 🔹 Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# Use output_dict=True to get a dictionary of metrics
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # zero_division=0 to handle warnings

# Extract relevant metrics for saving
metrics_to_save = {
    "accuracy": acc,
    "precision_macro": report['macro avg']['precision'],
    "recall_macro": report['macro avg']['recall'],
    "f1_macro": report['macro avg']['f1-score'],
    "precision_weighted": report['weighted avg']['precision'],
    "recall_weighted": report['weighted avg']['recall'],
    "f1_weighted": report['weighted avg']['f1-score'],
    "classification_report": report # Save the full report for completeness
}

print(f"\nBaseline model trained with accuracy: {acc:.2f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0)) # print full report

# 🔹 Save model and metadata
joblib.dump(model, MODEL_PATH)

with open(METADATA_PATH, "w") as f:
    json.dump(metrics_to_save, f, indent=4) # indent=4 for pretty printing JSON

print(f"✅ Model saved to: {MODEL_PATH}")
print(f"✅ Metadata saved to: {METADATA_PATH}")