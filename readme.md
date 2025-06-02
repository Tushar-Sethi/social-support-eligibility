# Social Support Eligibility Application

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [File Structure](#file-structure)  
7. [Configuration](#configuration)  


## Project Overview

The Social Support Eligibility Application is a web-based system designed to help applicants determine if they qualify for social-support benefits. It combines:

- A **Streamlit** user interface for collecting personal information and file uploads (bank statements, resume, assets/liabilities spreadsheet, credit report).  
- Traditional data-processing routines to parse and normalize tabular documents (CSV, Excel, PDF).  
- LLM-driven parsing of unstructured documents (resume, credit report) into structured JSON fields.  
- Validation logic that checks consistency (e.g., name matching, income vs. bank credits).  
- Feature engineering that computes employment history metrics (via LLM) and wealth metrics (assets/liabilities).  
- A pre-trained **machine-learning pipeline** (preprocessing + classifier) to predict eligibility.  
- An LLM-based explanation generator to produce a human-readable rationale for each decision.  
- A conversational agent that answers follow-up questions in context, guiding users on next steps or explaining validation errors.

While designed as a Streamlit prototype, it can be extended into a standalone API service for broader integration, and retrained on real data for production readiness.


## Features

- **Interactive Web Form**  
  - Collects personal details (name, DOB, contact, demographics, household size, declared income).  
  - File uploaders for bank statements, resume, assets/liabilities data, and credit report.

- **Document Parsing & Normalization**  
  - **Bank Statement**: Reads CSV, Excel, or PDF, outputs a standardized transactions table (Date, Description, Amount, Running Balance).  
  - **Resume**: Converts PDF/DOCX into structured fields (Name, Email, Education, Companies, Projects, Skills) via an LLM.  
  - **Assets & Liabilities**: Reads CSV/Excel, tags each row as “Asset” or “Liability,” and consolidates for metric computation.  
  - **Credit Report**: Extracts text from PDF, uses an LLM to parse credit score, personal details, and account summary.

- **Validation**  
  - **Name Consistency**: Fuzzy-match applicant name, resume name, and credit report name to detect mismatches.  
  - **Income vs. Bank Activity**: Compares declared income to average monthly credits (±10% tolerance).

- **Feature Engineering**  
  - **Employment Metrics**: LLM-driven computation of number of companies, total experience, average tenure, current tenure, earliest start year.  
  - **Wealth Metrics**: Programmatic computation of total assets, total liabilities, net worth, counts, ratio, property and investment breakdown.

- **Eligibility Prediction**  
  - Loads the latest pre-trained ML pipeline (scaling + one-hot encoding + classifier) and returns a binary “Eligible/Not Eligible” label.

- **Explanation Generation**  
  - Inspects model feature importances, selects top features, and constructs a prompt for an LLM to explain the decision in plain language.

- **Conversational Assistance**  
  - Context-aware chat interface that answers follow-up questions about validation errors, eligibility rationale, or next steps.

## architecture

<img src="architecture_diagram.svg" alt="Architecture Diagram" width="800" />


## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-org/social-support-eligibility.git
   cd social-support-eligibility

2. Create & Activate a Virtual Environment
    python3 -m venv venv
    venv/bin/activate

3. Install Dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

4. Ensure Ollama (or Other LLM) Is Running

    If using Ollama locally: start the Ollama daemon/server.
    Confirm connectivity by sending a test prompt or checking the HTTP endpoint (e.g., http://localhost:11434/api/generate).


## Usage
    Launch the Application
    - streamlit run Index.py

    Open the Browser
    By default, Streamlit will open a browser window at http://localhost:8501.
    If it doesn’t open automatically, navigate to that URL manually.

    1. Complete the Form
    2. Fill in Sections 1–3 (Personal Info, Demographics, Household & Income).
    3. Upload the four required documents:
    4. Bank Statement (CSV/Excel/PDF), Resume (PDF/DOCX), Assets & Liabilities (CSV/Excel), Credit Report (PDF)
    5. Submit and Wait
    6. Click Submit Application. The system will parse each file, validate data, run the ML prediction, generate an explanation, and display results.
    7. Expect a few seconds for each LLM‐based parsing step and a short pause for model inference.


## File Structure
```
application/
│
├── Index.py
│   └─ Main Streamlit application (forms, file uploads, state management)
│
├── data_ingestion/
│   ├── bank_parser.py
│   │   └─ Reads CSV/Excel/PDF, normalizes to a transactions table
│   ├── resume_parser.py
│   │   └─ Extracts structured fields (Name, Email, Companies, etc.) via LLM
│   ├── assets_liabilities_parser.py
│   │   └─ Tags and consolidates asset/liability rows from CSV/Excel
│   └── credit_report_parser.py
│       └─ Extracts credit-score and personal fields via LLM
│
├── data_processing/
│   └── data_validation.py
│       └─ Validates name consistency and declared-income vs. bank credits
│
├── Utils/
│   └── langchain_general_method.py
│       └─ LLM integration: prompt templates + Pydantic parsing for resume, credit, employment
│
├── ML/
│   ├── Generate_synthetic_data.py
│   │   └─ Creates synthetic training data for model development
│   ├── Best_model_selection_trainer.py
│   │   └─ Benchmarks classifiers, builds final preprocessing+classifier pipeline
│   ├── Trainer.py
│   │   └─ Entry point to generate data and train/save model
│   ├── Predictor.py
│   │   └─ Loads latest model and returns a prediction
│   ├── FInal_approach.py
│   │   └─ Builds an LLM prompt from feature importances for explanation
│   └── Training_data/
│       └─ synthetic_data.csv  (generated synthetic dataset)
│
├── Models/
│   └─ social_support_model_best<AlgorithmName>.pkl  (serialized pipelines)
│
├── prompts/
│   ├── resume_info_capture.txt
│   ├── credit_report_info_capture.txt
│   ├── process_employement_history.txt
│   ├── validation_template.txt
│   ├── ineligible_template.txt
│   ├── eligible_template.txt
│   ├── followup_template.txt
│   └─ … (all prompt templates for LLM chains)
│
├── parser.py
│   └─ Pydantic models defining expected fields for LLM-driven parsing
│
├── general_utils.py
│   └─ Helper routines (e.g., finding most-recent file in a directory)
│
├── llm.py
│   └─ Wrapper class for local LLM service (e.g., Ollama), HTTP calls
│
├── application_insight_agent.py
│   └─ Conversational agent: selects branch (validation/ineligible/eligible/followup) and queries LLM
│
├── requirements.txt
│   └─ Pinned Python dependencies for easy installation
```

## Configuration
    - LLM Settings

        1. Default uses a local LLM server at http://localhost:11434.
        2. To change, update llm.py (base URL) or set an environment variable LLM_BASE_URL.

    - Prompt Templates

        1. All LLM prompts live under the prompts/ directory.
        2. If you want to refine wording, edit the .txt files (e.g., resume_info_capture.txt, ineligible_template.txt).

    - Model Directory

        1. Pre-trained models must be placed in the Models/ folder.
        2. Models should follow the naming pattern social_support_model_best<AlgorithmName>.pkl.
        3. The predictor automatically picks the most recent file by timestamp.

    - Validation Parameters

        1. Similarity threshold for name matching (default 90) and percentage tolerance for income vs. bank credits (default ±10%) can be tweaked in data_validation.py.