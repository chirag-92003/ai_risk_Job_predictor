import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
import pickle


def load_trained_model():
    """
    Load the trained model from the notebook.
    Run this once after the notebook has trained the model.
    """
    # Option 1: Load from saved file (recommended)
    # with open('trained_model.pkl', 'rb') as f:
    #     return pickle.load(f)

    # Option 2: Import from notebook (if run in same session)
    # from job import model  # Assumes model variable exists in job.ipynb
    # return model

    raise NotImplementedError(
        "Save your model first: pickle.dump(model, open('trained_model.pkl', 'wb'))"
    )


class JobRiskPredictor:
    """
    Custom input handler for AI Job Risk prediction model.
    Encapsulates data preprocessing and model prediction.
    """

    # Encoding mappings (must match training data)
    EXPERIENCE_LEVEL_MAP = {"Entry": 0, "Mid": 1, "Senior": 2}
    EDUCATION_LEVEL_MAP = {"Bachelor": 0, "Master": 1, "PhD": 2}
    SALARY_BUCKET_MAP = {"Low": 0, "Medium": 1, "High": 2}
    RISK_CATEGORY_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

    # All countries and skills from training data (for one-hot encoding)
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

    def __init__(self, model: RandomForestClassifier):
        """Initialize with a trained RandomForestClassifier model."""
        self.model = model

    def create_input(
        self,
        country: str,
        experience_level: str,
        education_level: str,
        year: int,
        salary: float,
        primary_skill: str,
        skill_demand_score: int,
        job_openings: int,
        job_survival_class: int,
        salary_bucket: str,
    ) -> pd.DataFrame:
        """
        Create a properly formatted input DataFrame for prediction.

        Args:
            country: One of ['USA', 'India', 'Canada', 'UK', 'Germany', 'Australia']
            experience_level: One of ['Entry', 'Mid', 'Senior']
            education_level: One of ['Bachelor', 'Master', 'PhD']
            year: Year (2015-2035)
            salary: Annual salary
            primary_skill: One of ['AWS', 'Deep Learning', 'Docker', 'Excel', 'Java',
                                   'Python', 'SQL', 'Security', 'Strategy']
            skill_demand_score: Demand score (60-99)
            job_openings: Number of job openings
            job_survival_class: 0, 1, or 2
            salary_bucket: One of ['Low', 'Medium', 'High']

        Returns:
            DataFrame with properly encoded features ready for prediction
        """
        # Create base DataFrame
        data = {
            "experience_level": [self.EXPERIENCE_LEVEL_MAP[experience_level]],
            "education_level": [self.EDUCATION_LEVEL_MAP[education_level]],
            "year": [year],
            "salary": [salary],
            "skill_demand_score": [skill_demand_score],
            "job_openings": [job_openings],
            "job_survival_class": [job_survival_class],
            "salary_bucket": [self.SALARY_BUCKET_MAP[salary_bucket]],
        }

        # One-hot encode country
        for c in self.COUNTRIES:
            data[f"country_{c}"] = [1 if c == country else 0]

        # One-hot encode primary_skill
        for skill in self.SKILLS:
            data[f"primary_skill_{skill}"] = [1 if skill == primary_skill else 0]

        return pd.DataFrame(data)

    def predict(self, input_df: pd.DataFrame) -> dict:
        """
        Make prediction on prepared input.

        Args:
            input_df: DataFrame from create_input()

        Returns:
            Dictionary with prediction results
        """
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]

        return {
            "predicted_category": self.RISK_CATEGORY_MAP[prediction],
            "predicted_class": int(prediction),
            "probabilities": {
                "low_risk": float(probabilities[0]),
                "medium_risk": float(probabilities[1]),
                "high_risk": float(probabilities[2]),
            },
        }

    def predict_from_dict(self, payload: dict) -> dict:
        """
        Convenience method: create input and predict in one call.

        Args:
            payload: Dictionary with all required fields

        Returns:
            Dictionary with prediction results
        """
        input_df = self.create_input(**payload)
        return self.predict(input_df)


# Example usage
if __name__ == "__main__":
    # Load your trained model here
    from joblib import load

    model = load("trained_model.pkl")

    predictor = JobRiskPredictor(model)

    # Example prediction
    payload = {
        "country": "India",
        "experience_level": "Mid",
        "education_level": "Bachelor",
        "year": 2021,
        "salary": 18000.0,
        "primary_skill": "AWS",
        "skill_demand_score": 75,
        "job_openings": 20000,
        "job_survival_class": 1,
        "salary_bucket": "Medium",
    }

    result = predictor.predict_from_dict(payload)
    print(f"Prediction: {result['predicted_category']}")
    print(f"Probabilities: {result['probabilities']}")
    print(result["predicted_class"])
