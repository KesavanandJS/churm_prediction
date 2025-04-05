import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import *

def load_data():
    return pd.read_csv(RAW_DATA_PATH)

def preprocess_data(df):
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'].fillna(0, inplace=True)
    
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    features = preprocessor.fit_transform(df)
    target = df[TARGET]
    
    return features, target, preprocessor

def save_processed_data(features, target):
    processed_df = pd.DataFrame(features)
    processed_df[TARGET] = target
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
