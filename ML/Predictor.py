import joblib
import pandas as pd
from Utils.general_utils import get_most_recent_file

def predict_social_support(new_data):

    # Load the best model
    print("Loading the best model")
    best_model = get_most_recent_file('Models/')
    model_path = 'Models/'+best_model


    pipeline = joblib.load(model_path)

    # new_data = pd.DataFrame([{
    #     'declared_income': 4500.0,
    #     'No_of_companies_worked': 2,
    #     'current_employer_tenure_months': 12,
    #     'earliest_start_year': 2021,
    #     'total_experience': 24,
    #     'average_employement_tenure_months': 12.0,
    #     'total_assets': 100000,
    #     'total_liabilities': 20000,
    #     'net_worth': 80000,
    #     'num_assets': 2,
    #     'num_liabilities': 1,
    #     'asset_to_liability_ratio': 5.0,
    #     'property_asset_value': 80000,
    #     'investment_asset_value': 20000,
    #     'family_size': 4,
    #     'dependents': 2,
    #     'Age': 30,
    #     'Gender': 'Female',
    #     'marital_status': 'Single',
    #     'Disability': 'No',
    #     'Housing_type': 'Owned'
    # }])

    prediction = pipeline.predict(new_data)
    print("Predicted Social Support Eligibility:", prediction[0]) 
    return prediction[0],pipeline
