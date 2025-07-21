import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score
from colorama import Fore, Style

def detect_drift(test_data_path='data/test_sample.csv', threshold=0.05):
    model = joblib.load('models/model.joblib')

    with open('models/metadata.json') as f:
        baseline_accuracy = json.load(f)['baseline_accuracy']

    data = pd.read_csv(test_data_path)
    X_new = data.drop('target', axis=1)
    y_new = data['target']

    y_pred = model.predict(X_new)
    current_accuracy = accuracy_score(y_new, y_pred)

    drift_detected = (baseline_accuracy - current_accuracy) > threshold

    print(f"Current Accuracy: {current_accuracy:.2f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")
    print(f"Drift Detected: {'Yes' if drift_detected else 'No'}")

    return current_accuracy, baseline_accuracy, drift_detected

if __name__ == "__main__":
    current_accuracy, baseline_accuracy, drift_detected = detect_drift()

    accuracy_drop = baseline_accuracy - current_accuracy

    print("="*30)
    print("MODEL PERFORMANCE MONITORING")
    print("-"*30)
    print(f"Baseline Accuracy : {baseline_accuracy:.2f}")
    print(f"Current Accuracy  : {current_accuracy:.2f}")
    print(f"Accuracy Drop     : {accuracy_drop:.2f}")

    if drift_detected:
        print(Fore.RED + "DRIFT DETECTED!" + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "No Drift Detected" + Style.RESET_ALL)
    print("="*30)
