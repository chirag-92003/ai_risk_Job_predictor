from fastapi import FastAPI
from schema.requestModel import request_model

import pickle

import pandas as pd

app = FastAPI()

COUNTRIES = ["Australia", "Canada", "Germany", "India", "UK", "USA"]
SKILLS = [
    "AWS",
    "Deep Learning",
    "Docker",
    "Excel",
    "Java",
    "Python",
    "SQL",
    "Security",
    "Strategy",
]

RISK_CATEGORY_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


@app.post("/predict")
def predict(req: request_model):
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    dictionary = req.model_dump()

    for c in COUNTRIES:
        dictionary[f"country_{c}"] = [1 if c == dictionary["country"] else 0]

    for s in SKILLS:
        dictionary[f"primary_skill_{s}"] = [
            1 if s == dictionary["primary_skill"] else 0
        ]

    input_df = pd.DataFrame(dictionary)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "predicted_category": RISK_CATEGORY_MAP[prediction],
        "predicted_class": int(prediction),
        "probabilities": {
            "low_risk": float(probabilities[0]),
            "medium_risk": float(probabilities[1]),
            "high_risk": float(probabilities[2]),
        },
    }
