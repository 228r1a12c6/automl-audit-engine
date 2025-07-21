import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json

def retrain_model(data_path='data/base.csv'):
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, 'models/model.joblib')

    with open('models/metadata.json', 'w') as f:
        json.dump({'baseline_accuracy': accuracy}, f)

    print(f'Model retrained. New accuracy: {accuracy:.2f}')

