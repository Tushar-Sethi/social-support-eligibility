from thefuzz import fuzz
import pandas as pd


def nameCheck(application_name, resume_name, credit_report_name):
    """
    Check if three names likely refer to the same person using fuzzy matching.
    
    Args:
        application_name (str): Name from application form
        resume_name (str): Name from resume
        credit_report_name (str): Name from credit report
    
    Returns:
        dict: JSON with 'name_check' (bool) and 'reason' (str)
    """
    # Define similarity threshold for fuzzy matching
    SIMILARITY_THRESHOLD = 90

    # Perform fuzzy matching between pairs of names
    app_resume_score = fuzz.token_sort_ratio(application_name, resume_name)
    app_credit_score = fuzz.token_sort_ratio(application_name, credit_report_name)
    resume_credit_score = fuzz.token_sort_ratio(resume_name, credit_report_name)

    # Check if all pairs are above the threshold
    names_match = (
        app_resume_score >= SIMILARITY_THRESHOLD and
        app_credit_score >= SIMILARITY_THRESHOLD and
        resume_credit_score >= SIMILARITY_THRESHOLD
    )

    print("Names match-----------------------------------------------",names_match)

    # Generate reason based on matching results
    if names_match:
        final_resp = {'final_resp':True,'reason':(
            f"All names likely refer to the same person. "
            f"Similarity scores: Application Form - Resume Score: {app_resume_score}, "
            f"Application Form - Credit Report Score: {app_credit_score}, Resume - Credit Report Score: {resume_credit_score}"
        )}
    else:
        final_resp = {'final_resp':False,'reason':(
            f"Names may not refer to the same person. "
            f"Similarity scores: Application Form - Resume Score: {app_resume_score}, "
            f"Application Form - Credit Report Score: {app_credit_score}, Resume - Credit Report Score: {resume_credit_score}"
        )}

    return final_resp

def check_declared_income_vs_bank_all_credits(
    declared_income: float,
    bank_df: pd.DataFrame,
    tolerance: float = 0.10
) -> dict:
    """
    Compare declared income from application form with average monthly credit (all credit entries)
    calculated from bank statement data.

    Args:
        declared_income (float): The income declared on the application form.
        bank_df (pd.DataFrame): Bank statement DataFrame with columns ['Date', 'Description', 'Debit', 'Credit', 'Balance'].
        tolerance (float): Acceptable relative difference (e.g., 0.10 for ±10%). Defaults to 10%.

    Returns:
        dict containing:
            - declared_income (float)
            - avg_monthly_credit (float)
            - absolute_difference (float)
            - percent_difference (float)
            - is_within_tolerance (bool)
            - details (str)
    """
    # Make a copy and ensure 'Date' is datetime
    df = bank_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')

    # Group by year-month and sum the Credit column for all rows
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['Credit'] = df['Credit'].astype(float)
    monthly_credit = df.groupby('YearMonth')['Credit'].sum()

    if monthly_credit.empty:
        return {
            "declared_income": declared_income,
            "avg_monthly_credit": 0.0,
            "absolute_difference": declared_income,
            "percent_difference": 100.0,
            "is_within_tolerance": False,
            "details": "No credit entries found in bank statement."
        }

    # Calculate average monthly credit across all months present
    avg_monthly_credit = monthly_credit.mean()

    # Compute difference metrics
    abs_diff = abs(declared_income - avg_monthly_credit)
    if declared_income == 0:
        percent_diff = float('inf') if avg_monthly_credit != 0 else 0.0
    else:
        percent_diff = abs_diff / declared_income * 100

    is_within = (percent_diff / 100) <= tolerance

    details = (
        f"Found credit entries over {len(monthly_credit)} month(s). "
        f"Average monthly credit: {avg_monthly_credit:.2f}. "
        f"Declared income: {declared_income:.2f}. "
        f"Difference: {abs_diff:.2f} ({percent_diff:.2f}%)."
    )

    return {
        "final_resp":is_within,
        "declared_income": declared_income,
        "avg_monthly_credit": avg_monthly_credit,
        "absolute_difference": abs_diff,
        "percent_difference": percent_diff,
        "is_within_tolerance": is_within,
        "details": details
    }

def validate_details(application_form_details,bank_df,df_resume,df_assets_liabilities,credit_report_raw):
    """
    Validate the details of the application form, bank statement, resume, assets/liabilities, and credit report.
    
    Args:
        application_form_details (dict): Dictionary containing application form details
        bank_df (pd.DataFrame): DataFrame containing bank statement data
        df_resume (pd.DataFrame): DataFrame containing resume data
        df_assets_liabilities (pd.DataFrame): DataFrame containing assets/liabilities data
        credit_report_raw (dict): Dictionary containing credit report data

    Returns:
        bool: True if all details are valid, False otherwise
    """

    final_resp_name_check = nameCheck(application_form_details['full_name'], df_resume['Name'], 'Shlok Singh')
    response_bank_details = check_declared_income_vs_bank_all_credits(application_form_details['declared_income'], bank_df)

    if final_resp_name_check['final_resp'] == True and response_bank_details['final_resp'] == True:
        return [True,"All details are valid"]
    else:
        return [False,str(final_resp_name_check['reason'] + " \n\n " + response_bank_details['details'])]




