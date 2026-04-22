from pydantic import BaseModel


class request_model(BaseModel):
    country: str
    experience_level: str
    education_level: str
    year: list[int]
    salary: list[float]
    primary_skill: str
    skill_demand_score: list[int]
    job_openings: list[int]
    job_survival_class: list[int]
    salary_bucket: str
