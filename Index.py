# Index.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Parsing and validation utilities
from data_ingestion.bank_parser import process_bank_statement
from data_ingestion.resume_parser import process_resume
from data_ingestion.assets_liabilities_parser import process_assets_liabilities
from data_ingestion.credit_report_parser import process_credit_report

from data_processing.data_validation import validate_details

from Utils.langchain_general_method import EmployementHistoryParser, AssetsLiabilitiesParser

from ML.Predictor import predict_social_support
from ML.FInal_approach import generate_model_explanation_prompt

# Agent for chat‐style explanation
from Agents.application_insight_agent import ApplicationInsightAgent

# --------------------------------------------------------------------
# Streamlit page configuration and custom CSS
# --------------------------------------------------------------------
st.set_page_config(page_title="Social Support Eligibility", layout="wide")
st.markdown(
    """
    <style>
    /* Remove default Streamlit padding & make content full-width */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
        margin: 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        margin-top: 1rem;
        color: #222222;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background-color: #fafafa;
        border: 1px dashed #d0d0d0;
        border-radius: 5px;
        padding: 0.5rem;
    }

    /* Compact input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div,
    .stDateInput>div>div>input {
        height: 2.5rem;
        font-size: 1rem;
    }

    /* Button styling */
    .stButton>button {
        background-color: #007AFF;
        color: #FFFFFF;
        padding: 0.5rem 1.25rem;
        font-size: 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #005BBB;
        color: #FFFFFF;
    }

    /* Chat section styling */
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background-color: #ffffff;
        max-height: 500px;
        overflow-y: auto;
    }
    .chat-you {
        color: #004488;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .chat-assistant {
        color: #222222;
        font-weight: 400;
        margin-bottom: 0.75rem;
        margin-left: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Social Support Application Form")

# --------------------------------------------------------------------
# Initialize session state for flags, storage, and chat history
# --------------------------------------------------------------------
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

for key in [
    "df_bank",
    "df_resume",
    "df_assets_liabilities",
    "df_credit_report",
    "validation_errors",
    "df_employment_metrics",
    "df_wealth_metrics",
    "final_feature_dict",
    "final_df",
    "prediction_value",
    "trained_pipeline",
    "model_explanation",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if "state" not in st.session_state:
    st.session_state.state = {}

# Chat agent & history
if "app_agent" not in st.session_state:
    st.session_state.app_agent = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------
# STEP 1: APPLICATION FORM (repopulate with session_state if available)
# --------------------------------------------------------------------
with st.container():
    st.markdown('<div class="section-header">1. Personal Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", value=st.session_state.form_data.get("full_name", ""))
    with col2:
        gender_options = ["", "Male", "Female", "Other"]
        gender = st.selectbox(
            "Gender",
            gender_options,
            index=(
                gender_options.index(st.session_state.form_data.get("gender", ""))
                if st.session_state.form_data.get("gender", "") in gender_options
                else 0
            )
        )

    col3, col4 = st.columns(2)
    with col3:
        prev_dob = st.session_state.form_data.get("dob", "")
        dob = st.date_input(
            "Date of Birth",
            value=(
                datetime.datetime.strptime(prev_dob, "%Y-%m-%d").date()
                if prev_dob
                else datetime.date(1990, 1, 1)
            ),
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date.today()
        )
    with col4:
        contact_number = st.text_input("Contact Number", value=st.session_state.form_data.get("contact_number", ""))

    # Demographic Details
    st.markdown('<div class="section-header">2. Demographic Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        marital_options = ["", "Single", "Married", "Divorced", "Widowed"]
        marital_status = st.selectbox(
            "Marital Status",
            marital_options,
            index=(
                marital_options.index(st.session_state.form_data.get("marital_status", ""))
                if st.session_state.form_data.get("marital_status", "") in marital_options
                else 0
            )
        )
    with col2:
        country = st.text_input("Country", value=st.session_state.form_data.get("country", ""))
    with col3:
        nationality = st.text_input("Nationality", value=st.session_state.form_data.get("nationality", ""))

    col4, col5 = st.columns(2)
    with col4:
        disability_options = ["", "Yes", "No"]
        disability_status = st.selectbox(
            "Disability Status",
            disability_options,
            index=(
                disability_options.index(st.session_state.form_data.get("disability_status", ""))
                if st.session_state.form_data.get("disability_status", "") in disability_options
                else 0
            )
        )
    with col5:
        housing_options = ["", "Owned", "Rented", "Other"]
        housing_type = st.selectbox(
            "Housing Type",
            housing_options,
            index=(
                housing_options.index(st.session_state.form_data.get("housing_type", ""))
                if st.session_state.form_data.get("housing_type", "") in housing_options
                else 0
            )
        )

    # Household & Income
    st.markdown('<div class="section-header">3. Household & Income</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        family_size = st.number_input(
            "Family Size",
            min_value=1,
            step=1,
            value=st.session_state.form_data.get("family_size", 1)
        )
    with col2:
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            step=1,
            value=st.session_state.form_data.get("dependents", 0)
        )
    with col3:
        declared_income = st.number_input(
            "Declared Monthly Income (₹)",
            min_value=0,
            step=1000,
            value=st.session_state.form_data.get("declared_income", 0)
        )

    # Document Uploads
    st.markdown('<div class="section-header">4. Upload Documents</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        bank_statement_file = st.file_uploader("Bank Statement (CSV, XLSX, PDF)", type=["csv", "xlsx", "pdf"])
        resume_file = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        assets_liabilities_file = st.file_uploader("Assets & Liabilities (CSV or XLSX)", type=["csv", "xlsx"])
        credit_report_file = st.file_uploader("Credit Report (PDF)", type=["pdf"])

    st.markdown("")  # Spacer
    submit = st.button("Submit Application")

# --------------------------------------------------------------------
# STEP 2: PROCESS SUBMISSION (runs only when "Submit Application" is clicked)
# --------------------------------------------------------------------
if submit:
    # 1. Collect form inputs into a dict, converting dob to ISO string
    st.session_state.form_data = {
        "full_name": name,
        "gender": gender,
        "dob": dob.strftime("%Y-%m-%d"),
        "contact_number": contact_number,
        "marital_status": marital_status,
        "country": country,
        "nationality": nationality,
        "disability_status": disability_status,
        "housing_type": housing_type,
        "family_size": int(family_size),
        "dependents": int(dependents),
        "declared_income": int(declared_income),
    }

    # 2. Check for missing required fields
    missing_fields = []
    for key, value in st.session_state.form_data.items():
        if (isinstance(value, str) and value.strip() == "") or (key == "declared_income" and value == 0):
            missing_fields.append(key)

    # print('Missing fields:',missing_fields)

    # Ensure each document is uploaded
    if bank_statement_file is None:
        missing_fields.append("bank_statement_file")
    if resume_file is None:
        missing_fields.append("resume_file")
    if assets_liabilities_file is None:
        missing_fields.append("assets_liabilities_file")
    if credit_report_file is None:
        missing_fields.append("credit_report_file")

    # 3. If missing any required fields, record validation errors & seed chat
    if missing_fields:
        st.session_state.validation_errors = missing_fields.copy()

        # Clear previous data
        st.session_state.df_bank = None
        st.session_state.df_resume = None
        st.session_state.df_assets_liabilities = None
        st.session_state.df_credit_report = None
        st.session_state.df_employment_metrics = None
        st.session_state.df_wealth_metrics = None
        st.session_state.final_feature_dict = {}
        st.session_state.final_df = None
        st.session_state.prediction_value = None
        st.session_state.trained_pipeline = None
        st.session_state.model_explanation = ""

        st.session_state.submitted = True

        # Build minimal state for the agent
        st.session_state.state = {
            "form_data": st.session_state.form_data,
            "parsed_docs": {
                "bank": None,
                "resume": None,
                "assets_liabilities": None,
                "credit": None,
            },
            "validation_errors": st.session_state.validation_errors,
            "feature_vector": {},
            "prediction": None,
            "model_explanation": "",
        }

        # Instantiate the agent if not already done
        if st.session_state.app_agent is None:
            st.session_state.app_agent = ApplicationInsightAgent()

        # Seed the chat with a validation‐failure explanation
        initial_prompt = "Why did my application fail validation?"
        first_response = st.session_state.app_agent.run(st.session_state.state, initial_prompt)
        st.session_state.chat_history = [
            ("You", initial_prompt),
            ("Assistant", first_response),
        ]

    else:
        # 4. All required fields are present: parse documents and store results
        # ---------------------------------------------------------------------
        # 4a. Bank Statement Parsing
        try:
            df_bank = process_bank_statement(bank_statement_file)
            # st.markdown("**Parsed Bank Statement:**")
            # st.dataframe(df_bank)
            st.session_state.df_bank = df_bank
        except Exception as e:
            # st.error(f"Failed to parse Bank Statement: {e}")
            st.session_state.df_bank = None

        # 4b. Resume Parsing
        try:
            df_resume = process_resume(resume_file)
            # st.markdown("**Parsed Resume:**")
            # st.dataframe(df_resume)
            st.session_state.df_resume = df_resume
        except Exception as e:
            # st.error(f"Failed to parse Resume: {e}")
            st.session_state.df_resume = None

        # 4c. Assets & Liabilities Parsing
        try:
            df_assets_liabilities = process_assets_liabilities(assets_liabilities_file)
            # st.markdown("**Parsed Assets & Liabilities:**")
            # if isinstance(df_assets_liabilities, dict):
            #     for sheet_name, df_sheet in df_assets_liabilities.items():
            #         st.markdown(f"**Sheet: {sheet_name}**")
            #         st.dataframe(df_sheet)
            # else:
            #     st.dataframe(df_assets_liabilities)
            st.session_state.df_assets_liabilities = df_assets_liabilities
        except Exception as e:
            # st.error(f"Failed to parse Assets & Liabilities: {e}")
            st.session_state.df_assets_liabilities = None

        # 4d. Credit Report Parsing
        try:
            df_credit_report = process_credit_report(credit_report_file)
            # st.markdown("**Parsed Credit Report:**")
            # st.dataframe(df_credit_report)
            st.session_state.df_credit_report = df_credit_report
        except Exception as e:
            # st.error(f"Failed to parse Credit Report: {e}")
            st.session_state.df_credit_report = None

        # 5. Validate all details (including parsed DataFrames)
        validation_ok, validation_errs = validate_details(
            st.session_state.form_data,
            st.session_state.df_bank,
            st.session_state.df_resume,
            st.session_state.df_assets_liabilities,
            st.session_state.df_credit_report,
        )

        if not validation_ok:
            # Validation returned errors
            st.session_state.validation_errors = validation_errs if isinstance(validation_errs, list) else [validation_errs]
            # st.error("Validation errors: " + "; ".join(st.session_state.validation_errors))

            # Clear metrics/prediction
            st.session_state.df_employment_metrics = None
            st.session_state.df_wealth_metrics = None
            st.session_state.final_feature_dict = {}
            st.session_state.final_df = None
            st.session_state.prediction_value = None
            st.session_state.trained_pipeline = None
            st.session_state.model_explanation = ""

            st.session_state.submitted = True

            # Build state for the agent
            st.session_state.state = {
                "form_data": st.session_state.form_data,
                "parsed_docs": {
                    "bank": st.session_state.df_bank,
                    "resume": st.session_state.df_resume,
                    "assets_liabilities": st.session_state.df_assets_liabilities,
                    "credit": st.session_state.df_credit_report,
                },
                "validation_errors": st.session_state.validation_errors,
                "feature_vector": {},
                "prediction": None,
                "model_explanation": "",
            }

            if st.session_state.app_agent is None:
                st.session_state.app_agent = ApplicationInsightAgent()

            # Seed chat with validation failure explanation
            initial_prompt = "Why did my application fail validation?"
            first_response = st.session_state.app_agent.run(st.session_state.state, initial_prompt)
            st.session_state.chat_history = [
                ("You", initial_prompt),
                ("Assistant", first_response),
            ]

        else:
            # 6. Validation passed — compute employment & wealth metrics
            st.success("All details validated successfully.")

            # 6a. Employment history metrics via LLM chain
            try:
                employment_parser = EmployementHistoryParser()
                df_employment_metrics = employment_parser.parse_employement_history_llm(
                    st.session_state.df_resume["Companies"], "process_employement_history"
                )
                # st.markdown("**Employment History Metrics:**")
                # st.dataframe(df_employment_metrics)
                st.session_state.df_employment_metrics = df_employment_metrics
            except Exception as e:
                # st.error(f"Failed to parse Employment History: {e}")
                st.session_state.df_employment_metrics = None

            # 6b. Assets & Liabilities metrics via LLM chain
            try:
                assets_parser = AssetsLiabilitiesParser()
                if isinstance(st.session_state.df_assets_liabilities, dict):
                    combined_assets = pd.concat(
                        st.session_state.df_assets_liabilities.values(), ignore_index=True
                    )
                else:
                    combined_assets = st.session_state.df_assets_liabilities
                df_wealth_metrics = assets_parser.get_wealth_assessment(combined_assets)
                # st.markdown("**Wealth Metrics:**")
                # st.dataframe(df_wealth_metrics)
                st.session_state.df_wealth_metrics = df_wealth_metrics
            except Exception as e:
                # st.error(f"Failed to compute Assets/Liabilities metrics: {e}")
                st.session_state.df_wealth_metrics = None

            # 7. Build feature dictionary for classification
            emp = st.session_state.df_employment_metrics
            if emp is not None:
                num_companies = int(emp.get("number_of_companies", 0))
                curr_tenure = int(emp.get("current_employer_tenure_months", 0))
                earliest_start = int(emp.get("earliest_start_year", int(st.session_state.form_data["dob"][:4])))
                total_exp = int(emp.get("total_experience_months", 0))
                avg_tenure = emp.get("average_tenure_months", 0)
            else:
                num_companies = curr_tenure = earliest_start = total_exp = avg_tenure = 0

            wealth = st.session_state.df_wealth_metrics
            if wealth is not None:
                total_assets = int(wealth.get("total_assets", 0))
                total_liabilities = int(wealth.get("total_liabilities", 0))
                net_worth = int(wealth.get("net_worth", 0))
                num_assets = int(wealth.get("num_assets", 0))
                num_liabilities = int(wealth.get("num_liabilities", 0))
                asset_to_liability_ratio = float(wealth.get("asset_to_liability_ratio", 0))
                property_assets_value = int(wealth.get("property_asset_value", 0))
                investment_assets_value = int(wealth.get("investment_asset_value", 0))
            else:
                (
                    total_assets,
                    total_liabilities,
                    net_worth,
                    num_assets,
                    num_liabilities,
                    asset_to_liability_ratio,
                    property_assets_value,
                    investment_assets_value,
                ) = (0, 0, 0, 0, 0, 0.0, 0, 0)

            # Calculate applicant's age
            dob_date = datetime.datetime.strptime(st.session_state.form_data["dob"], "%Y-%m-%d").date()
            today = datetime.date.today()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))

            final_feature_dict = {
                "declared_income": st.session_state.form_data["declared_income"],
                "No_of_companies_worked": int(num_companies),
                "current_employer_tenure_months": int(curr_tenure),
                "earliest_start_year": int(earliest_start),
                "total_experience": int(total_exp),
                "average_employement_tenure_months": avg_tenure,
                "total_assets": int(total_assets),
                "total_liabilities": int(total_liabilities),
                "net_worth": int(net_worth),
                "num_assets": int(num_assets),
                "num_liabilities": int(num_liabilities),
                "asset_to_liability_ratio": float(asset_to_liability_ratio),
                "property_asset_value": int(property_assets_value),
                "investment_asset_value": int(investment_assets_value),
                "family_size": int(st.session_state.form_data["family_size"]),
                "dependents": int(st.session_state.form_data["dependents"]),
                "Gender": st.session_state.form_data["gender"],
                "marital_status": st.session_state.form_data["marital_status"],
                "Disability": st.session_state.form_data["disability_status"],
                "Housing_type": st.session_state.form_data["housing_type"],
                "Age": int(age),
            }

            st.session_state.final_feature_dict = final_feature_dict
            final_df = pd.DataFrame([final_feature_dict])
            st.session_state.final_df = final_df

            # st.markdown("**Final Classification Model Inputs:**")
            # st.dataframe(final_df)

            # 8. Predict social support eligibility
            try:
                prediction_value, trained_pipeline = predict_social_support(final_df)
                st.session_state.prediction_value = int(prediction_value)
                st.session_state.trained_pipeline = trained_pipeline
                # if prediction_value == 1:
                #     st.success("Final Review Result: Eligible")
                # else:
                #     st.error("Final Review Result: Not Eligible")
            except Exception as e:
                # st.error(f"Prediction failed: {e}")
                st.session_state.prediction_value = None
                st.session_state.trained_pipeline = None

            # 9. Generate natural‐language explanation via LLM
            if st.session_state.trained_pipeline is not None and st.session_state.prediction_value is not None:
                explanation_prompt = generate_model_explanation_prompt(
                    st.session_state.trained_pipeline,
                    st.session_state.final_df,
                    st.session_state.prediction_value,
                )
                try:
                    from Utils.llm import Ollama
                    from langchain.prompts import PromptTemplate
                    from langchain.chains import LLMChain

                    llm = Ollama(model="gemma3:1b", temperature=0.9, max_tokens=25000)
                    explanation_chain = LLMChain(
                        llm=llm,
                        prompt=PromptTemplate(input_variables=["text"], template="{text}")
                    )
                    model_explanation = explanation_chain.run(text=explanation_prompt).strip()
                    st.session_state.model_explanation = model_explanation
                    # st.markdown("**Natural‐Language Explanation:**")
                    # st.write(model_explanation)
                except Exception as e:
                    # st.error(f"Failed to generate explanation: {e}")
                    st.session_state.model_explanation = ""

            # 10. Build complete state and initialize chat‐agent
            parsed_docs_dict = {
                "bank": st.session_state.df_bank,
                "resume": st.session_state.df_resume,
                "assets_liabilities": st.session_state.df_assets_liabilities,
                "credit": st.session_state.df_credit_report,
            }
            st.session_state.state = {
                "form_data": st.session_state.form_data,
                "parsed_docs": parsed_docs_dict,
                "validation_errors": [],  # no validation errors
                "feature_vector": st.session_state.final_feature_dict,
                "prediction": st.session_state.prediction_value,
                "model_explanation": st.session_state.model_explanation,
            }
            st.session_state.submitted = True

            if st.session_state.app_agent is None:
                st.session_state.app_agent = ApplicationInsightAgent()

            # Seed the chat with eligibility/ineligibility explanation
            if st.session_state.prediction_value == 0:
                initial_prompt = "Why am I ineligible?"
            else:
                initial_prompt = "Why am I eligible, and what are the next steps?"

            first_response = st.session_state.app_agent.run(st.session_state.state, initial_prompt)
            st.session_state.chat_history = [
                ("You", initial_prompt),
                ("Assistant", first_response),
            ]

# --------------------------------------------------------------------
# STEP 3: ALWAYS RE-DISPLAY STORED CONTENT (parsed results, chat, etc.)
# --------------------------------------------------------------------
if st.session_state.submitted:
    # 3a) Re-display all parsing & model outputs
    st.markdown("---")

    # 3a.1) If there were validation errors, show them again
    if st.session_state.validation_errors:
        st.error("Validation errors:")
        for err in st.session_state.validation_errors:
            st.write(f"- {err}")

    # 3a.2) If df_bank exists, show it
    if st.session_state.df_bank is not None:
        st.markdown("**Parsed Bank Statement:**")
        st.dataframe(st.session_state.df_bank)

    # 3a.3) If df_resume exists, show it
    if st.session_state.df_resume is not None:
        st.markdown("**Parsed Resume:**")
        st.dataframe(st.session_state.df_resume)

    # 3a.4) If df_assets_liabilities exists, show it
    if st.session_state.df_assets_liabilities is not None:
        st.markdown("**Parsed Assets & Liabilities:**")
        if isinstance(st.session_state.df_assets_liabilities, dict):
            for sheet_name, df_sheet in st.session_state.df_assets_liabilities.items():
                st.markdown(f"**Sheet: {sheet_name}**")
                st.dataframe(df_sheet)
        else:
            st.dataframe(st.session_state.df_assets_liabilities)

    # 3a.5) If df_credit_report exists, show it
    if st.session_state.df_credit_report is not None:
        st.markdown("**Parsed Credit Report:**")
        st.dataframe(st.session_state.df_credit_report)

    # 3a.6) Re-display employment metrics
    if st.session_state.df_employment_metrics is not None:
        st.markdown("**Employment History Metrics:**")
        st.dataframe(st.session_state.df_employment_metrics)

    # 3a.7) Re-display wealth metrics
    if st.session_state.df_wealth_metrics is not None:
        st.markdown("**Wealth Metrics:**")
        st.dataframe(st.session_state.df_wealth_metrics)

    # 3a.8) Re-display final feature inputs
    if st.session_state.final_df is not None:
        st.markdown("**Final Classification Model Inputs:**")
        st.dataframe(st.session_state.final_df)

    # 3a.9) Re-display the eligibility result
    if st.session_state.prediction_value is not None:
        if st.session_state.prediction_value == 1:
            st.success("Final Review Result: Eligible")
        else:
            st.error("Final Review Result: Not Eligible")

    # 3a.10) Re-display the natural-language explanation
    if st.session_state.model_explanation:
        st.markdown("**Natural‐Language Explanation:**")
        st.write(st.session_state.model_explanation)

    # 3b) Chat‐style interface using chat components
    st.markdown("---")
    st.markdown('<div class="section-header">Chat with the Application Assistant</div>', unsafe_allow_html=True)

    # Render old messages first
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

    # Use st.chat_input for follow-ups—only this part “appends” new messages
    if user_input := st.chat_input("Ask a question about your application"):
        # 1) Save the user’s message
        st.session_state.chat_history.append(("You", user_input))
        st.chat_message("user").write(user_input)

        # 2) Generate the assistant’s reply
        response = st.session_state.app_agent.run(st.session_state.state, user_input)
        st.session_state.chat_history.append(("Assistant", response))
        st.chat_message("assistant").write(response)
