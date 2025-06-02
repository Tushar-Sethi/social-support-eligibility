# application_insight_agent.py

import json
import pandas as pd
from typing import Any, Dict, Optional

from Utils.llm import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class ApplicationInsightAgent:
    """
    An AI agent that routes user queries into one of three conversational modes—
    (A) validation‐failed, (B) ineligible, or (C) eligible—injecting exactly the
    state information needed for each case and returning an LLM‐generated response.
    """

    def __init__(self):
        """
        Initialize the agent with an Ollama LLM chain (gemma3:1b, temperature=0.3).
        """
        # Instantiate the Ollama model
        llm = Ollama(model="gemma3:1b", temperature=0.3, max_tokens=25000)

        # Wrap the Ollama model in an LLMChain that expects a single "text" input
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="{text}"
            )
        )

        # Load all prompt templates for each conversational branch
        self._load_prompts()

    def _load_prompts(self):
        """
        Define three PromptTemplate instances—one for each conversational branch.
        """
        # Branch A: Validation failed
        self.validation_template = PromptTemplate(
            input_variables=[
                "user_query",
                "form_data",
                "parsed_bank",
                "parsed_resume",
                "parsed_assets_liabilities",
                "parsed_credit",
                "validation_errors",
            ],
            template="""
User asks: "{user_query}"

The applicant’s raw form data:
{form_data}

Parsed document excerpts (where available):
- Bank Statement parsing result:
{parsed_bank}

- Resume parsing result:
{parsed_resume}

- Assets & Liabilities parsing result:
{parsed_assets_liabilities}

- Credit Report parsing result:
{parsed_credit}

Validation errors detected:
{validation_errors}

Please respond as an application assistant:
1. Explain, step by step, why each field failed validation.
2. For each missing/invalid item, explicitly tell the user what format or value is expected.
3. Suggest corrections or next steps to fix the validation errors.

IMPORTANT NOTE:
 - The response you will generate will be directly displayed to the user.
 - Be polite, professional and concise.
 - you will be directly referring to the user so respond accordingly and appropriately.
"""
        )

        # Branch B: Valid but Ineligible
        self.ineligible_template = PromptTemplate(
            input_variables=[
                "user_query",
                "form_data",
                "parsed_bank",
                "parsed_resume",
                "parsed_assets_liabilities",
                "parsed_credit",
                "feature_vector",
                "prediction",
                "model_explanation",
            ],
            template="""
User asks: "{user_query}"

The applicant’s raw form data:
{form_data}

Parsed document details:
- Bank Statement parsing result:
{parsed_bank}

- Resume parsing result:
{parsed_resume}

- Assets & Liabilities parsing result:
{parsed_assets_liabilities}

- Credit Report parsing result:
{parsed_credit}

Constructed feature vector for the eligibility model:
{feature_vector}

Model’s binary prediction (0 = Ineligible, 1 = Eligible):
{prediction}

Human‐readable explanation produced by the model:
{model_explanation}

Please respond as an eligibility advisor:
1. Walk the user through each major feature that caused “Ineligible.”
   - For example: “Your declared income was X, which is below the threshold of Y,” etc.
2. Highlight any borderline or critical thresholds they missed.
3. Offer guidance on what could improve their eligibility, if applicable.
4. Keep the tone conversational but precise, and invite any follow‐up questions.

IMPORTANT NOTE:
 - The response you will generate will be directly displayed to the user.
 - Be polite, professional and concise.
 - you will be directly referring to the user so respond accordingly and appropriately.
"""
        )

        # Branch C: Valid and Eligible
        self.eligible_template = PromptTemplate(
            input_variables=[
                "user_query",
                "form_data",
                "parsed_bank",
                "parsed_resume",
                "parsed_assets_liabilities",
                "parsed_credit",
                "feature_vector",
                "prediction",
                "model_explanation",
            ],
            template="""
User asks: "{user_query}"

The applicant’s raw form data:
{form_data}

Parsed document details:
- Bank Statement parsing result:
{parsed_bank}

- Resume parsing result:
{parsed_resume}

- Assets & Liabilities parsing result:
{parsed_assets_liabilities}

- Credit Report parsing result:
{parsed_credit}

Constructed feature vector for the eligibility model:
{feature_vector}

Model’s binary prediction (0 = Ineligible, 1 = Eligible):
{prediction}

Human‐readable explanation produced by the model:
{model_explanation}

Please respond as an eligibility consultant:
1. Emphasize the strengths in their application that led to “Eligible.”
   - For example: “Your declared income of X exceeds the requirement of Y,” etc.
2. Outline any next steps or documentation still needed (e.g., “Submit proof of address”).
3. Encourage them to ask any follow‐up questions about their benefits, timeline, or process.
4. Maintain a friendly, informative tone.

IMPORTANT NOTE:
 - The response you will generate will be directly displayed to the user.
 - Be polite, professional and concise.
 - you will be directly referring to the user so respond accordingly and appropriately.

"""
        )

        # Common follow‐up template if the user’s query is a narrow, context-specific question
        self.followup_template = PromptTemplate(
            input_variables=[
                "user_query",
                "form_data",
                "parsed_bank",
                "parsed_resume",
                "parsed_assets_liabilities",
                "parsed_credit",
                "validation_errors",
                "feature_vector",
                "prediction",
                "model_explanation",
            ],
            template="""
User asks: "{user_query}"

Here is the full application state we have so far:

Raw form data:
{form_data}

Parsed document details:
- Bank Statement:
{parsed_bank}

- Resume:
{parsed_resume}

- Assets & Liabilities:
{parsed_assets_liabilities}

- Credit Report:
{parsed_credit}

Validation errors (if any):
{validation_errors}

Constructed feature vector:
{feature_vector}

Model prediction (0 = Ineligible, 1 = Eligible):
{prediction}

Model’s explanation:
{model_explanation}

Please answer the user’s question by focusing only on the specific aspect mentioned in "{user_query}". 
If they reference bank info, answer only about Bank Statement parsing, etc. 
Be concise, but include any relevant numbers or thresholds.

IMPORTANT NOTE:
 - The response you will generate will be directly displayed to the user.
 - Be polite, professional and concise.
 - you will be directly referring to the user so respond accordingly and appropriately.
"""
        )

    def _stringify(self, obj: Any) -> str:
        """
        Convert different object types (DataFrame, dict, etc.) into readable strings.
        """
        if isinstance(obj, pd.DataFrame):
            # Use a truncated version if the DataFrame is very large
            try:
                return obj.to_string(max_rows=20, max_cols=10)
            except Exception:
                return str(obj)
        elif isinstance(obj, dict):
            return json.dumps(obj, indent=2)
        else:
            return str(obj)

    def _determine_branch(self, state: Dict[str, Any], user_query: str) -> str:
        """
        Decide which conversational branch to use:
          - "validation" if validation_errors exist
          - "ineligible" if valid & prediction == 0
          - "eligible" if valid & prediction == 1
          - "followup" if none of the above
        """
        if state.get("validation_errors"):
            return "validation"
        elif state.get("prediction") == 0:
            return "ineligible"
        elif state.get("prediction") == 1:
            return "eligible"
        else:
            # If no prediction or validation errors, treat as followup
            return "followup"

    def build_prompt(self, state: Dict[str, Any], user_query: str) -> str:
        """
        Construct the appropriate prompt text by stringifying the relevant parts
        of the state and filling in the corresponding PromptTemplate.
        """
        # Stringify each piece of state safely
        form_data_str = self._stringify(state.get("form_data", {}))

        parsed = state.get("parsed_docs", {})
        parsed_bank_str = self._stringify(parsed.get("bank", "No bank‐statement data"))
        parsed_resume_str = self._stringify(parsed.get("resume", "No resume data"))
        parsed_assets_str = self._stringify(parsed.get("assets_liabilities", "No assets/liabilities data"))
        parsed_credit_str = self._stringify(parsed.get("credit", "No credit‐report data"))

        validation_errors_str = self._stringify(state.get("validation_errors", "None"))
        feature_vector_str = self._stringify(state.get("feature_vector", {}))
        prediction_str = self._stringify(state.get("prediction", "No prediction"))
        model_explanation_str = self._stringify(state.get("model_explanation", "No explanation"))

        branch = self._determine_branch(state, user_query)

        if branch == "validation":
            return self.validation_template.format(
                user_query=user_query,
                form_data=form_data_str,
                parsed_bank=parsed_bank_str,
                parsed_resume=parsed_resume_str,
                parsed_assets_liabilities=parsed_assets_str,
                parsed_credit=parsed_credit_str,
                validation_errors=validation_errors_str,
            )

        elif branch == "ineligible":
            return self.ineligible_template.format(
                user_query=user_query,
                form_data=form_data_str,
                parsed_bank=parsed_bank_str,
                parsed_resume=parsed_resume_str,
                parsed_assets_liabilities=parsed_assets_str,
                parsed_credit=parsed_credit_str,
                feature_vector=feature_vector_str,
                prediction=prediction_str,
                model_explanation=model_explanation_str,
            )

        elif branch == "eligible":
            return self.eligible_template.format(
                user_query=user_query,
                form_data=form_data_str,
                parsed_bank=parsed_bank_str,
                parsed_resume=parsed_resume_str,
                parsed_assets_liabilities=parsed_assets_str,
                parsed_credit=parsed_credit_str,
                feature_vector=feature_vector_str,
                prediction=prediction_str,
                model_explanation=model_explanation_str,
            )

        else:  # followup
            return self.followup_template.format(
                user_query=user_query,
                form_data=form_data_str,
                parsed_bank=parsed_bank_str,
                parsed_resume=parsed_resume_str,
                parsed_assets_liabilities=parsed_assets_str,
                parsed_credit=parsed_credit_str,
                validation_errors=validation_errors_str,
                feature_vector=feature_vector_str,
                prediction=prediction_str,
                model_explanation=model_explanation_str,
            )

    def run(self, state: Dict[str, Any], user_query: str) -> str:
        """
        Given the current application state and a user query, build the prompt,
        send it to the LLMChain, and return the generated response.
        """
        prompt_text = self.build_prompt(state, user_query)
        response = self.llm_chain.run(text=prompt_text).strip()
        return response
