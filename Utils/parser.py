from pydantic import BaseModel, Field
from typing import List

class resume_info_capture(BaseModel):
    """Pydantic model for general query chain output."""

    Name: str = Field(description="name of the candidate")
    email_address: str = Field(description="Email of the candidate")
    Education: str = Field(description="Education details of the candidate")
    Companies: str = Field(description="Companies candidate has worked for")
    projects: str = Field(description="projects candidate has handled")
    Skills: str = Field(description="Different Skills candidate has.")

    class Config:
        allow_population_by_field_name = True
        # With `alias="email address"`, Pydantic will accept either "email_address" or "email address"
        fields = {"email_address": {"alias": "email address"}}


class credit_report_info_capture(BaseModel):
    """Pydantic model for general query chain output."""
    credit_score: str = Field(description="Credit score of the candidate")
    Name: str = Field(description="Name of the candidate")
    email: str = Field(description="Email of the candidate")
    DOB: str = Field(description="Date of birth of the candidate")
    Gender: str = Field(description="Gender of the candidate")
    Address: str = Field(description="Address of the candidate")
    number: str = Field(description="Phone number of the candidate")
    information: str = Field(description="Account information of the candidate")



class emirates_id_info_capture(BaseModel):
    """Pydantic model for general query chain output."""
    Name: str = Field(description="Name of the candidate")
    ID: str = Field(description="ID of the candidate")
    Nationality: str = Field(description="Nationality of the candidate")


class EmploymentHistoryFeatures(BaseModel):
    number_of_companies: int = Field(description="Number of companies person has worked in")
    total_experience_months: int = Field(description="Total years person has worked for")
    average_tenure_months: float = Field(description="total_experience_months divided by number_of_companies")
    current_employer_tenure_months: int = Field(description="total months person has worked for last employer")
    earliest_start_year: int = Field(description="year in which person started working")



