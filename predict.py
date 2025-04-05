import joblib
import pandas as pd
from config import MODEL_PATH

def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data['model'], model_data['preprocessor']

def predict_churn(new_data=None):
    model, preprocessor = load_model()
    
    if new_data is None:
        print("\nEnter customer details for churn prediction:")
        new_data = {
            'gender': input("Gender (Male/Female): "),
            'senior_citizen': int(input("Senior Citizen (0/1): ")),
            'partner': input("Partner (Yes/No): "),
            'dependents': input("Dependents (Yes/No): "),
            'tenure': int(input("Tenure (months): ")),
            'phone_service': input("Phone Service (Yes/No): "),
            'multiple_lines': input("Multiple Lines (Yes/No/No phone service): "),
            'internet_service': input("Internet Service (DSL/Fiber optic/No): "),
            'online_security': input("Online Security (Yes/No/No internet service): "),
            'online_backup': input("Online Backup (Yes/No/No internet service): "),
            'device_protection': input("Device Protection (Yes/No/No internet service): "),
            'tech_support': input("Tech Support (Yes/No/No internet service): "),
            'streaming_tv': input("Streaming TV (Yes/No/No internet service): "),
            'streaming_movies': input("Streaming Movies (Yes/No/No internet service): "),
            'contract': input("Contract (Month-to-month/One year/Two year): "),
            'paperless_billing': input("Paperless Billing (Yes/No): "),
            'payment_method': input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "),
            'monthly_charges': float(input("Monthly Charges ($): ")),
            'total_charges': float(input("Total Charges ($): "))
        }
        new_data = pd.DataFrame([new_data])
    elif isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    elif isinstance(new_data, str):
        new_data = pd.read_csv(new_data)
    
    features = preprocessor.transform(new_data)
    
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'customer_id': new_data.iloc[i].get('customer_id', f'customer_{i}'),
            'churn_prediction': 'Yes' if pred == 1 else 'No',
            'churn_probability': float(prob),
            'churn_risk': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
        })
    
    return results

if __name__ == '__main__':
    sample_data = {
        'gender': 'Female',
        'senior_citizen': 0,
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': 12,
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'No',
        'online_backup': 'Yes',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': 29.85,
        'total_charges': 358.2
    }
    
    predictions = predict_churn(sample_data)
    for pred in predictions:
        print(f"Customer: {pred['customer_id']}")
        print(f"Prediction: {pred['churn_prediction']}")
        print(f"Probability: {pred['churn_probability']:.2f}")
        print(f"Risk Level: {pred['churn_risk']}")
        print("-" * 30)
