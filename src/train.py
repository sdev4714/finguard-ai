import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression as MetaLearner
from preprocess import preprocess_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_base_models():
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=0.1
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    return lr, rf, xgb

def build_stacking_model(lr, rf, xgb):
    estimators = [
        ('logistic_regression', lr),
        ('random_forest', rf),
        ('xgboost', xgb)
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=MetaLearner(max_iter=1000, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )

    return stack

def train_all_models(X_train, y_train):
    lr, rf, xgb = build_base_models()

    print("Training Logistic Regression...")
    lr.fit(X_train, y_train)
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  LR CV F1: {lr_cv:.4f}")

    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  RF CV F1: {rf_cv:.4f}")

    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  XGB CV F1: {xgb_cv:.4f}")

    print("Training Stacking Ensemble...")
    stack = build_stacking_model(lr, rf, xgb)
    stack.fit(X_train, y_train)
    stack_cv = cross_val_score(stack, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  Stack CV F1: {stack_cv:.4f}")

    models = {
        'logistic_regression': (lr, lr_cv),
        'random_forest': (rf, rf_cv),
        'xgboost': (xgb, xgb_cv),
        'stacking_ensemble': (stack, stack_cv)
    }

    return models

def save_models(models):
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    for name, (model, score) in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {name}.pkl (CV F1: {score:.4f})")

    # Save best model separately as loan_model.pkl
    best_name = max(models, key=lambda k: models[k][1])
    best_model = models[best_name][0]
    with open(os.path.join(models_dir, "loan_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nBest model: {best_name} → saved as loan_model.pkl")

    return best_name, best_model

def load_model(model_name="loan_model"):
    path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    print("=== Starting Training Pipeline ===\n")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_pipeline()
    print()
    models = train_all_models(X_train, y_train)
    best_name, best_model = save_models(models)
    print(f"\n=== Training Complete ===")
    print(f"Best Model: {best_name}")