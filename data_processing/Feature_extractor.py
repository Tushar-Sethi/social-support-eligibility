
from datetime import datetime
import pandas as pd

class Data_Processor:
    

    def compute_applicant_features(
        form_data: dict,
        bank_df: pd.DataFrame,
        resume_data: dict,
        assets_libiabilities_df: pd.DataFrame,
        credit_data: dict,
        ) -> dict:
        """
        Assemble a feature dict with keys:
        - annual_income
        - employment_status
        - total_experience_years
        - current_tenure_years
        - num_dependents
        - net_worth
        - avg_monthly_balance
        - credit_utilization_ratio
        - age_years
        - gender
        - nationality
        - education_level
        - marital_status
        - urban_resident

        form_data: parsed from application form (dict)
        bank_df: bank statement as DataFrame with columns ['date','amount','running_balance', …]
        resume_data: parsed resume fields (dict)
        eid_data: parsed Emirates ID fields (dict)
        assets_df, liabilities_df: DataFrames of assets/liabilities with column 'Amount'
        credit_data: parsed credit report (dict)

        get_llm_response: optional function for LLM-based inference:
            get_llm_response(prompt, question, **inputs)
        """
        features = {}

        # 1. annual_income
        if "declared_income" in form_data:
            try:
                s = form_data["Income"]
                features["annual_income"] = s * 12
            except:
                features["annual_income"] = None
        else:
            # fallback: average monthly deposit from bank_df
            if bank_df is not None and not bank_df.empty:
                bank_df["Credit"] = bank_df["Credit"].astype(float)
                bank_df["date"] = pd.to_datetime(bank_df["date"], errors="coerce")
                bank_df = bank_df.dropna(subset=["date"])
                bank_df["month"] = bank_df["date"].dt.to_period("M")
                monthly_credits = bank_df[bank_df["Credit"] > 0].groupby("month")["amount"].sum()
                if not monthly_credits.empty:
                    avg = monthly_credits.mean()
                    features["annual_income"] = avg * 12
                else:
                    features["annual_income"] = None
            else:
                features["annual_income"] = None

        # 2. employment_status, current_tenure_years
        features["employment_status"] = form_data.get("Employment Status") or resume_data.get("employment_status") or None
        start_date = None
        if "Date Hired" in form_data:
            try:
                start_date = datetime.strptime(form_data["Date Hired"], "%m/%Y")
            except:
                start_date = None
        if start_date is None and resume_data.get("Experience"):
            # LLM can infer start date of current job from resume text
            if get_llm_response:
                prompt = "Extract the start date of the current (most recent) job in YYYY-MM format"
                question = resume_data.get("Experience")
                inferred = get_llm_response(prompt, question=question)
                try:
                    start_date = datetime.strptime(inferred, "%Y-%m")
                except:
                    start_date = None
        if start_date:
            delta = datetime.now() - start_date
            features["current_tenure_years"] = round(delta.days / 365, 2)
        else:
            features["current_tenure_years"] = None

        # 3. total_experience_years
        exp_list = resume_data.get("experience", [])
        total_days = 0
        for entry in exp_list:
            # Expect strings like "Jun 2018 – Jul 2023" or "06/2018 - 07/2023"
            parts = entry.replace("–", "-").split("-")
            if len(parts) == 2:
                try:
                    start = datetime.strptime(parts[0].strip(), "%m/%Y")
                    end = datetime.strptime(parts[1].strip(), "%m/%Y")
                    total_days += (end - start).days
                except:
                    pass
        features["total_experience_years"] = round(total_days / 365, 2) if total_days > 0 else None

        # 4. num_dependents
        if "Number of Dependents" in form_data:
            try:
                features["num_dependents"] = int(form_data["Number of Dependents"])
            except:
                features["num_dependents"] = None
        else:
            features["num_dependents"] = None

        # 5. net_worth
        if assets_df is not None and liabilities_df is not None:
            try:
                total_assets = assets_df["Amount"].astype(float).sum()
            except:
                total_assets = None
            try:
                total_liab = liabilities_df["Amount"].abs().astype(float).sum()
            except:
                total_liab = None
            if total_assets is not None and total_liab is not None:
                features["net_worth"] = total_assets - total_liab
            else:
                features["net_worth"] = None
        else:
            features["net_worth"] = None

        # 6. avg_monthly_balance
        if bank_df is not None and not bank_df.empty:
            try:
                bank_df["running_balance"] = bank_df["running_balance"].astype(float)
                features["avg_monthly_balance"] = bank_df["running_balance"].mean()
            except:
                features["avg_monthly_balance"] = None
        else:
            features["avg_monthly_balance"] = None

        # 7. credit_utilization_ratio
        util = None
        acct_info = credit_data.get("Account Information", [])
        for acct in acct_info:
            if acct.get("Type").lower() == "revolving":
                try:
                    limit_str = acct.get("Original Amount", "")
                    limit = float("".join(ch for ch in limit_str if ch.isdigit() or ch == "."))
                    bal_str = acct.get("Recent Balance", "")
                    bal = float("".join(ch for ch in bal_str if ch.isdigit() or ch == "."))
                    util = bal / limit if limit else None
                    break
                except:
                    util = None
        features["credit_utilization_ratio"] = round(util, 2) if util is not None else None

        # 8. age_years
        dob = eid_data.get("DOB") or form_data.get("Date of Birth")
        try:
            birth = datetime.strptime(dob, "%B %d, %Y")
            features["age_years"] = int((datetime.now() - birth).days / 365)
        except:
            features["age_years"] = None

        # 9. gender, nationality
        features["gender"] = eid_data.get("Gender") or resume_data.get("Gender") or None
        features["nationality"] = eid_data.get("Nationality") or form_data.get("Nationality") or None

        # 10. education_level
        edu = resume_data.get("Education") or form_data.get("Education")
        if edu:
            if "phd" in edu.lower() or "doctor" in edu.lower():
                features["education_level"] = "PhD"
            elif "master" in edu.lower() or "m.sc" in edu.lower():
                features["education_level"] = "Master"
            elif "bachelor" in edu.lower() or "b.sc" in edu.lower():
                features["education_level"] = "Bachelor"
            else:
                # fallback to LLM inference
                if get_llm_response:
                    prompt = "Given this education string, classify as HighSchool, Bachelor, Master, or PhD"
                    question = edu
                    inferred = get_llm_response(prompt, question=question)
                    features["education_level"] = inferred
                else:
                    features["education_level"] = None
        else:
            features["education_level"] = None

        # 11. marital_status
        if "Marital Status" in form_data:
            features["marital_status"] = form_data["Marital Status"]
        else:
            # fallback: LLM guess from resume text
            if get_llm_response and resume_data.get("Education"):
                prompt = "Infer marital status (Single, Married, Divorced, Widowed) from this resume text"
                question = resume_data.get("Education") + "\n\n" + resume_data.get("experience", "")
                features["marital_status"] = get_llm_response(prompt, question=question)
            else:
                features["marital_status"] = None

        # 12. urban_resident
        addr = form_data.get("Address") or resume_data.get("Address")
        if addr:
            # simple rule: if city in address list of metros
            metro_list = ["Dubai", "Abu Dhabi", "Sharjah", "Al Ain", "Ras al Khaimah"]
            features["urban_resident"] = any(city.lower() in addr.lower() for city in metro_list)
        else:
            features["urban_resident"] = None

        return features

