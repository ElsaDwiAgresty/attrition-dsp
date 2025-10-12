import mlflow
import mlflow.sklearn
import joblib
import requests
import os
import random

def get_model():
    uri_artifacts = "https://dagshub.com/ElsaDwiAgresty/dsp_attrition.mlflow"
    mlflow.set_tracking_uri(uri_artifacts)

    model_uri="models:/rf_model/2"
    model = mlflow.sklearn.load_model(model_uri)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_model_v2.pkl")
    return model

def load_model():
    MODEL_PATH = "model/rf_model_v2.pkl"
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file nor found at {MODEL_PATH}")
        return None

def generate_random_features():
    return {
        "MonthlyIncome": random.randint(1000, 20000),
        "Age": random.randint(18, 60),
        "TotalWorkingYears": random.randint(0, 40),
        "OverTime": random.choice([0, 1]),
        "MonthlyRate": random.randint(1000, 20000),
        "DailyRate": random.randint(100, 2000),
        "DistanceFromHome": random.randint(1, 50),
        "HourlyRate": random.randint(10, 200),
        "NumCompaniesWorked": random.randint(0, 10)
    }

def set_features():
    return [
        "MonthlyIncome", "Age", "TotalWorkingYears", "OverTime",
        "MonthlyRate", "DailyRate", "DistanceFromHome",
        "HourlyRate", "NumCompaniesWorked"
    ]