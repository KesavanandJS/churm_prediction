import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = 'data/raw/churn_data.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'

MODEL_PATH = 'models/churn_model.pkl'

CATEGORICAL_FEATURES = ['gender', 'partner', 'dependents', 'phone_service', 
                       'multiple_lines', 'internet_service', 'online_security',
                       'online_backup', 'device_protection', 'tech_support',
                       'streaming_tv', 'streaming_movies', 'contract',
                       'paperless_billing', 'payment_method']

NUMERICAL_FEATURES = ['tenure', 'monthly_charges', 'total_charges']

TARGET = 'churn'

TEST_SIZE = 0.2
RANDOM_STATE = 42
