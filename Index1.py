# app.py

import streamlit as st
from datetime import date

def main():
    # Page configuration for full-screen (wide) layout
    import streamlit as st
from datetime import date

st.set_page_config(
    page_title="🇦🇪 Social Support Application",
    page_icon="📝",
    layout="wide",  # Full-screen coverage
    initial_sidebar_state="auto"
)

# Custom CSS for a cleaner, full-width look
st.markdown(
    """
    <style>
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;            /* Allow full-width content */
        margin: 0;                   /* Remove extra margins */
    }

    /* Container styling */
    .stContainer {
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-bottom: 2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333333;
    }

    /* File uploader area */
    .stFileUploader > div {
        background-color: #F7F7F7;
        border-radius: 5px;
        padding: 0.5rem;
    }

    /* Button hover effect */
    div.stButton > button:hover {
        background-color: #0072C3;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown(
    """
    <div class="stContainer">
        <h1 style="color:#0072C3;">📝 Social Support Application Form</h1>
        <p style="font-size:1rem; color:#555555;">
            Welcome! Please complete the application form below and upload the required documents.
            Fields marked with <span style="color:#E03737;">*</span> are mandatory.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Begin form container
with st.container():
    with st.form(key="application_form", clear_on_submit=False):
        # --- Section: Applicant Information ---
        st.markdown('<div class="section-header">👤 Applicant Information</div>', unsafe_allow_html=True)

        # Two-column layout for name and gender
        col1, col2 = st.columns([3, 1])
        with col1:
            full_name = st.text_input("Full Name *", placeholder="e.g., Ahmed Al Mansoori")
        with col2:
            gender = st.selectbox("Gender *", options=["Male", "Female", "Other"])

        # Two-column layout for DOB and Contact Number
        col3, col4 = st.columns([3, 1])
        with col3:
            dob = st.date_input(
                "Date of Birth *",
                min_value=date(1900, 1, 1),
                max_value=date.today()
            )
        with col4:
            contact_number = st.text_input(
                "Contact Number *",
                placeholder="+971-50-123-4567",
                help="Include country code"
            )

        # Single-column for address (spans full width)
        address = st.text_area(
            "Residential Address *",
            height=80,
            placeholder="House #, Street Name, City, Emirates"
        )

        # --- Section: Demographic Profile ---
        st.markdown('<div class="section-header">📝 Demographic Profile</div>', unsafe_allow_html=True)

        dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
        with dcol1:
            marital_status = st.selectbox(
                "Marital Status *",
                options=["Single", "Married", "Divorced", "Widowed", "Other"]
            )
        with dcol2:
            country = st.text_input(
                "Country *",
                placeholder="e.g., United Arab Emirates"
            )
        with dcol3:
            nationality = st.text_input(
                "Nationality *",
                placeholder="e.g., Emirati, Expat"
            )

        dcol4, dcol5 = st.columns([1, 1])
        with dcol4:
            has_disability = st.selectbox(
                "Disability / Special Needs *",
                options=["No", "Yes"],
                help="If yes, support needs may differ"
            )
        with dcol5:
            housing_type = st.selectbox(
                "Housing Type *",
                options=["Renting", "Owned", "Living with Family", "Other"]
            )

        st.markdown("---")

        # --- Section: Household & Income ---
        st.markdown('<div class="section-header">🏠 Household & Income</div>', unsafe_allow_html=True)
        col5, col6 = st.columns([1, 1])
        with col5:
            family_size = st.number_input(
                "Family Size *",
                min_value=1,
                step=1,
                help="Including yourself"
            )
            dependents = st.number_input(
                "Dependents *",
                min_value=0,
                step=1
            )
        with col6:
            declared_income = st.number_input(
                "Declared Monthly Income (AED) *",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                help="As declared in the form"
            )

        st.markdown("---")

        # --- Section: Upload Documents ---
        st.markdown('<div class="section-header">📂 Upload Required Documents</div>', unsafe_allow_html=True)
        st.write("Please upload one file per category. Supported file types are shown in each uploader.")

        # Three-column layout for file uploaders
        up_col1, up_col2, up_col3 = st.columns([1, 1, 1])

        with up_col1:
            bank_statement = st.file_uploader(
                "• Bank Statement *",
                type=["pdf", "csv", "xlsx"],
                help="Digital export (CSV/XLSX) or PDF statement"
            )

        with up_col2:
            resume = st.file_uploader(
                "• Resume (CV) *",
                type=["pdf", "docx"],
                help="PDF or Word document of your CV"
            )
            assets_liabilities = st.file_uploader(
                "• Assets/Liabilities File *",
                type=["xlsx", "csv"],
                help="Excel or CSV listing assets & liabilities"
            )

        with up_col3:
            credit_report = st.file_uploader(
                "• Credit Report *",
                type=["pdf", "csv", "xlsx"],
                help="Credit bureau report as PDF, CSV, or Excel"
            )

        st.markdown("---")

        # Submit button in a centered column (spans full width)
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submit_button = st.form_submit_button(
                label="✅ Submit Application",
                help="Click to submit your application",
            )

    application_form_details = {
        "full_name": full_name,
        "dob": dob,
        "gender": gender,
        "contact_number": contact_number,
        "address": address,
        "marital_status": marital_status,
        "country": country,
        "nationality": nationality,
        "has_disability": has_disability,
        "housing_type": housing_type,
        "family_size": family_size,
        "dependents": dependents,
        "declared_income": declared_income
    }

    # Process submission
    if submit_button:
        # Validate required fields
        missing_fields = []
        if not full_name.strip():
            missing_fields.append("Full Name")
        if not dob:
            missing_fields.append("Date of Birth")
        if not gender:
            missing_fields.append("Gender")
        if not address.strip():
            missing_fields.append("Residential Address")
        if not marital_status:
            missing_fields.append("Marital Status")
        if not country.strip():
            missing_fields.append("Country")
        if not nationality.strip():
            missing_fields.append("Nationality")
        if has_disability not in ["Yes", "No"]:
            missing_fields.append("Disability / Special Needs")
        if not housing_type:
            missing_fields.append("Housing Type")
        if family_size < 1:
            missing_fields.append("Family Size")
        if dependents < 0:
            missing_fields.append("Dependents")
        if declared_income < 0:
            missing_fields.append("Declared Monthly Income")
        if bank_statement is None:
            missing_fields.append("Bank Statement")
        if resume is None:
            missing_fields.append("Resume")
        if assets_liabilities is None:
            missing_fields.append("Assets/Liabilities File")
        if credit_report is None:
            missing_fields.append("Credit Report")

        if missing_fields:
            st.markdown(
                f"""
                <div style="background-color:#FFE3E3; padding:1rem; border-radius:5px;">
                    <strong style="color:#E03737;">⚠️ Please complete all mandatory fields:</strong>
                    <ul style="color:#333333; margin-top:0.5rem;">
                        {''.join(f'<li>{field}</li>' for field in missing_fields)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Success confirmation
            st.markdown(
                """
                <div style="background-color:#E7F9ED; padding:1rem; border-radius:5px;">
                    <strong style="color:#097969;">🎉 Application submitted successfully!</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Summary of submitted data (full width)
            st.markdown('<div class="section-header">📋 Submitted Information</div>', unsafe_allow_html=True)
            col7, col8 = st.columns([2, 2])
            with col7:
                st.markdown(f"**Full Name:** {full_name}")
                st.markdown(f"**Date of Birth:** {dob.strftime('%Y-%m-%d')}")
                st.markdown(f"**Gender:** {gender}")
                st.markdown(f"**Contact Number:** {contact_number}")
                st.markdown(f"**Marital Status:** {marital_status}")
                st.markdown(f"**Country:** {country}")
            with col8:
                st.markdown(f"**Nationality:** {nationality}")
                st.markdown(f"**Disability / Special Needs:** {has_disability}")
                st.markdown(f"**Housing Type:** {housing_type}")

            st.markdown("---")
            st.markdown('<div class="section-header">🏠 Household & Income</div>', unsafe_allow_html=True)
            col9, col10 = st.columns([2, 2])
            with col9:
                st.markdown(f"**Family Size:** {family_size}")
                st.markdown(f"**Dependents:** {dependents}")
            with col10:
                st.markdown(f"**Declared Monthly Income:** AED {declared_income:,.2f}")

            st.markdown("---")
            st.markdown('<div class="section-header">📂 Uploaded Documents</div>', unsafe_allow_html=True)
            st.write(f"• **Bank Statement:** {bank_statement.name}")
            st.write(f"• **Resume (CV):** {resume.name}")
            st.write(f"• **Assets/Liabilities File:** {assets_liabilities.name}")
            st.write(f"• **Credit Report:** {credit_report.name}")


        
        from data_ingestion.bank_parser import process_bank_statement

        st.title("Uploaded Bank Statement")
        # bank_file = st.file_uploader("Bank Statement", type=["pdf", "csv", "xlsx"])
        if bank_statement:
            try:


                df_bank = process_bank_statement(bank_statement)
                print('Type of df_bank',type(df_bank))
                # print('Bank Statement-> \n',df_bank)
                print('-'*10)
                st.success("Bank statement parsed successfully!")
                st.dataframe(df_bank.head(10))


            except Exception as e:
                st.error(f"Error parsing bank statement: {e}")


        from data_ingestion.resume_parser import process_resume

        st.title("Uploaded Resume Information Captured: ")
        if resume:
            try:
                df_resume = process_resume(resume)
                print('Type of df_resume',type(df_resume))
                # print('Resume-> \n',df_resume)
                print('-'*10)
                st.success("Resume parsed successfully!")
                st.dataframe(df_resume)
            except Exception as e:
                st.error(f"Error parsing resume: {e}")

        
        from data_ingestion.assets_liabilities_parser import process_assets_liabilities
        st.title("Uploaded Assets/Liabilities File")
        if assets_liabilities:
            try:
                df_assets_liabilities = process_assets_liabilities(assets_liabilities)
                print('Type of df_assets_liabilities',type(df_assets_liabilities))
                print('Assets/Liabilities-> \n',df_assets_liabilities)
                print('-'*10)
                st.success("Assets/Liabilities parsed successfully!")
                st.dataframe(df_assets_liabilities)
            except Exception as e:
                st.error(f"Error parsing assets/liabilities: {e}")

        from data_ingestion.credit_report_parser import process_credit_report
        st.title("Uploaded Credit Report")
        if credit_report:
            try:
                import copy
                raw = process_credit_report(credit_report)
                if(raw is None):
                    st.error("Error parsing credit report: Credit report is empty")
                else:
                    credit_report_raw = copy.deepcopy(raw)
                    st.success("Credit report parsed successfully!")
                    st.dataframe(raw)
            except Exception as e:
                st.error(f"Error parsing credit report: {e}")



        from data_processing.data_validation import validate_details
        response = validate_details(application_form_details,df_bank,df_resume,df_assets_liabilities,credit_report_raw)
        print("Response from data validation",response)
        if response[0] == True:
            st.success("All details are valid")
        else:
            st.error(response[1])
        

        from Utils.langchain_general_method import EmployementHistoryParser
        employement_history_parser = EmployementHistoryParser()
        result_employement = employement_history_parser.parse_employement_history_llm(df_resume['Companies'],'process_employement_history')
        print("Result from employement history parser",result_employement)
        st.dataframe(result_employement)


        from Utils.langchain_general_method import AssetsLiabilitiesParser
        assets_liabilities_parser = AssetsLiabilitiesParser()
        result_assets_liabilities = assets_liabilities_parser.get_wealth_assessment(df_assets_liabilities)
        print("Result from assets/liabilities parser",result_assets_liabilities)
        st.dataframe(result_assets_liabilities)
        

        final_classification_model_inputs = {
             "declared_income": declared_income,
             
             "No_of_companies_worked": result_employement['number_of_companies'],
             "current_employer_tenure_months": result_employement['current_employer_tenure_months'],
             "earliest_start_year": result_employement['earliest_start_year'],
             "total_experience": result_employement['total_experience_months'],
             "average_employement_tenure_months": result_employement['average_tenure_months'],


             "total_assets": result_assets_liabilities['total_assets'],
             "total_liabilities": result_assets_liabilities['total_liabilities'],
             "net_worth": result_assets_liabilities['net_worth'],
             "num_assets": result_assets_liabilities['num_assets'],
             "num_liabilities": result_assets_liabilities['num_liabilities'],
             "asset_to_liability_ratio": result_assets_liabilities['asset_to_liability_ratio'],
             "property_asset_value": result_assets_liabilities['property_asset_value'],
             "investment_asset_value": result_assets_liabilities['investment_asset_value'],

             "family_size": family_size,
             "dependents":dependents,
            


            "Age": (date.today().year - dob.year),
            "Gender": gender,
            "marital_status":marital_status,
            
            "Disability": has_disability,
            "Housing_type": housing_type
         }   


        st.title("Final Classification Model Inputs")
        st.dataframe(final_classification_model_inputs)
        import pandas as pd
        from ML.Predictor import predict_social_support
        new_df = pd.DataFrame([final_classification_model_inputs])
        prediction,pipeline = predict_social_support(new_df)

        # prediction = predict_social_support(new_df)

        print("Prediction from classification model",prediction)
        st.title(f"Final Review Result: {'Eligible' if prediction == 1 else 'Not Eligible'}")


        from ML.FInal_approach import generate_model_explanation_prompt
        
        prompt = generate_model_explanation_prompt(pipeline, new_df, prediction)


        from Utils.llm import Ollama
        from langchain import LLMChain, PromptTemplate

        llm = Ollama(model="gemma3:1b", temperature=0.3, max_tokens=25000)
        
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="{text}"
            )
        )
        # Get the natural-language explanation
        explanation_text = llm_chain.run({"text": prompt})

        st.title("Natural-Language Explanation")
        st.write(explanation_text)        
        


        # from data_ingestion.emirates_id_parser import extract_emirates_id_text
        # st.title("Uploaded Emirates ID")
        # if emirates_id:
        #     try:
        #         df_emirates_id = extract_emirates_id_text(emirates_id)
        #         st.success("Emirates ID parsed successfully!")
        #         st.dataframe(df_emirates_id)
        #     except Exception as e:
        #         st.error(f"Error parsing emirates ID: {e}")

if __name__ == "__main__":
    main()
