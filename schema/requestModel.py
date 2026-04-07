from pydantic import BaseModel


class request_model(BaseModel):
    country: str
    experience_level: str
    education_level: str
    year: int
    salary: float
    primary_skill: str
    skill_demand_score: int
    job_openings: int
    job_survival_class: int
    salary_bucket: str
