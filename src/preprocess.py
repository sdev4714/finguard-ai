import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "data", "raw", "loan_data.csv")
    df = pd.read_csv(path)
    return df

def handle_missing(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='str').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def engineer_features(df):
    df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)
    df['total_income'] = df['income'] + df['co_applicant_income']
    df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['total_income'] + 1)
    df['emi'] = df['loan_amount'] / (df['loan_term'] + 1)
    df['emi_to_income'] = df['emi'] / (df['income'] + 1)
    return df

def encode_features(df, encoders=None, fit=True):
    categorical_cols = ['gender', 'education', 'employment_type', 'property_area']

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col])

    return df, encoders

def scale_features(X, scaler=None, fit=True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"After SMOTE — Class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

def get_feature_names():
    return [
        'age', 'gender', 'education', 'employment_type',
        'income', 'co_applicant_income', 'loan_amount',
        'loan_term', 'credit_score', 'existing_loans',
        'property_area', 'dependents',
        'debt_to_income', 'total_income', 'income_per_dependent',
        'loan_to_income_ratio', 'emi', 'emi_to_income'
    ]

def preprocess_pipeline(path=None, save_artifacts=True):
    df = load_data(path)
    print(f"Loaded: {df.shape}")

    df = handle_missing(df)
    df = engineer_features(df)
    df, encoders = encode_features(df, fit=True)

    feature_cols = get_feature_names()
    X = df[feature_cols]
    y = df['approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)

    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, "cleaned_data.csv"), index=False)

    if save_artifacts:
        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(models_dir, "label_encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)
        print("Artifacts saved: scaler.pkl, label_encoders.pkl")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler, encoders

def preprocess_single(input_dict, scaler, encoders):
    df = pd.DataFrame([input_dict])
    df = handle_missing(df)
    df = engineer_features(df)
    df, _ = encode_features(df, encoders=encoders, fit=False)
    feature_cols = get_feature_names()
    X = df[feature_cols]
    X_scaled, _ = scale_features(X, scaler=scaler, fit=False)
    return X_scaled

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_pipeline()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")