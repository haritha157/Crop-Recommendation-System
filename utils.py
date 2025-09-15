import json
import pandas as pd
import joblib

model = joblib.load("crop_model_randomforest1.pkl")


def predict_crop(data):
    df = pd.DataFrame([data])
    return model.predict(df)[0]


