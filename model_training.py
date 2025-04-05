from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from config import *

def train_models():
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    models = {
        'logistic_regression': LogisticRegression(random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        if name == 'random_forest':
            joblib.dump({
                'model': model,
                'preprocessor': preprocessor
            }, MODEL_PATH)
    
    return results

if __name__ == '__main__':
    results = train_models()
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(metrics['classification_report'])
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
