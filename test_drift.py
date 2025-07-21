from monitor import detect_drift

accuracy, baseline, drift = detect_drift('data/test_sample.csv')

print(f"Current Accuracy: {accuracy:.2f}")
print(f"Baseline Accuracy: {baseline:.2f}")
print(f"Drift Detected: {'Yes' if drift else 'No'}")
