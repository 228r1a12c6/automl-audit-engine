import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os # Import the os module
import numpy as np # Import numpy for potential floor division

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
# IMPORTANT: Adjust 'target' column name if your base.csv has a different one.
# For example, if your target is 'label', change to df.drop("label", axis=1) and y = df["label"]
# You might need to inspect base.csv to confirm the target column.
try:
    X = df.drop("target", axis=1)
    y = df["target"]
except KeyError:
    print("Error: 'target' column not found. Please check your dataset and update the target column name if necessary.")
    exit()


# 🔹 Train-test split
# Ensure 'y' has at least 2 unique classes for classification
if y.nunique() < 2:
    print("Error: Target variable has less than 2 unique classes. Cannot perform classification.")
    exit()

# --- MODIFICATION START ---
# Calculate the minimum number of samples needed per class for stratification
min_samples_per_class = y.value_counts().min()
num_classes = y.nunique()

# Determine a safe test_size
# If dataset is very small, we might need to adjust test_size or even remove stratification
current_test_size = 0.2 # Your original test_size
adjusted_test_size = current_test_size

# Calculate how many samples from the smallest class would end up in the test set
# If this is less than 1, we have a problem with stratification.
# For train_test_split with stratify, each class must have at least 1 sample in both splits.
# So, the number of samples for the smallest class in the test set must be at least 1.
# This implies that min_samples_per_class * test_size >= 1
# And min_samples_per_class * (1 - test_size) >= 1

# A safer approach is to ensure that min_samples_per_class is large enough
# If min_samples_per_class is too low for the chosen test_size
if min_samples_per_class < 1 / (1 - current_test_size) or min_samples_per_class < 1 / current_test_size:
    print(f"Warning: Dataset is small ({len(df)} samples, smallest class has {min_samples_per_class} samples).")
    # Try increasing test_size to be more forgiving for stratification
    # If 0.2 didn't work, let's try 0.3 or 0.4.
    if len(df) >= 10 and num_classes < len(df) * 0.4: # Heuristic to avoid making test set too large
        adjusted_test_size = 0.3 # Try 30% for test set first
        if min_samples_per_class < 1 / (1 - adjusted_test_size) or min_samples_per_class < 1 / adjusted_test_size:
             adjusted_test_size = 0.4 # Try 40%
             if min_samples_per_class < 1 / (1 - adjusted_test_size) or min_samples_per_class < 1 / adjusted_test_size:
                 # If even 0.4 doesn't work, or dataset is truly tiny, consider no stratification or larger test_size
                 print("Cannot guarantee stratification with current dataset size. Increasing test_size to 0.5.")
                 adjusted_test_size = 0.5 # As a last resort for very small datasets
    else:
        print("Dataset too small for stratification, or very few samples in smallest class. Proceeding without stratification.")
        # If the dataset is truly too small for any reasonable stratification, remove it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        # Skip the stratified split below
        print("Model trained without stratification due to very small dataset.")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        joblib.dump(model, MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump({"accuracy": acc}, f)
        print(f"\nBaseline model trained with accuracy: {acc:.2f}\n")
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred))
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Metadata saved to: {METADATA_PATH}")
        exit() # Exit after non-stratified training for very small dataset


print(f"Using test_size={adjusted_test_size} for train_test_split.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=adjusted_test_size, random_state=42, stratify=y)
# --- MODIFICATION END ---

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