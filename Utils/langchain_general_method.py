from Utils.llm import Ollama
from Utils.parser import resume_info_capture,credit_report_info_capture,emirates_id_info_capture,EmploymentHistoryFeatures
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser


class ResumeParser:
    def __init__(self):
        self.llm = Ollama(model="gemma3:1b", temperature=0.3, max_tokens=25000)
        
    def parse_resume_text_llm(self,raw_text,prompt_path):
        with open(f'prompts/{prompt_path}.txt', 'r') as file:
            prompt = file.read()
        inputs = {"final_text": raw_text}
        pydantic_parser = resume_info_capture
        parser = JsonOutputParser(pydantic_object=pydantic_parser)
        prompt_template = PromptTemplate(input_variables=["final_text"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template, output_parser=parser)
        result = chain.invoke(inputs)
        return result["text"]
    

class CreditReportParser:
    def __init__(self):
        self.llm = Ollama(model="gemma3:1b", temperature=0.3, max_tokens=25000)

    def parse_credit_report_text_llm(self,raw_text,prompt_path):
        with open(f'prompts/{prompt_path}.txt', 'r') as file:
            prompt = file.read()
        # print('Prompt from Credit Report Parser----------\n\n',prompt)
        # print('-'*100)
        inputs = {"final_text": raw_text}
        pydantic_parser = credit_report_info_capture
        parser = JsonOutputParser(pydantic_object=pydantic_parser)
        prompt_template = PromptTemplate(input_variables=["final_text"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template,output_parser=parser)
        result = chain.invoke(inputs)
        # print('Result from Credit Report Parser----------\n\n',result['text'])

        # remove ```JSON from the result
        final_json = result["text"]
        print('Response from LLM for Credit report ->\n\n',final_json)
        # first_occurance = final_json.find('{')
        # last_occurance = final_json.rfind('}')
        # final_json = final_json[first_occurance:last_occurance+1]
        # import ast
        # final_json = ast.literal_eval(final_json)
        return final_json
        

class EmiratedIDParser:

    def __init__(self):
        self.llm = Ollama(model="gemma3:1b", temperature=0.7, max_tokens=15000)

    def parse_Emirated_ID_llm(self,raw_text,prompt_path):
        with open(f'prompts/{prompt_path}.txt', 'r') as file:
            prompt = file.read()
        inputs = {"final_text": raw_text}
        pydantic_parser = emirates_id_info_capture
        parser = JsonOutputParser(pydantic_object=pydantic_parser)
        prompt_template = PromptTemplate(input_variables=["final_text"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template, output_parser=parser)
        result = chain.invoke(inputs)
        return result["text"]
    

class EmployementHistoryParser:
    def __init__(self):
        self.llm = Ollama(model="gemma3:1b", temperature=0.3, max_tokens=25000)

    def parse_employement_history_llm(self,raw_text,prompt_path):
        with open(f'prompts/{prompt_path}.txt', 'r') as file:
            prompt = file.read()
        inputs = {"employement_history": raw_text}
        pydantic_parser = EmploymentHistoryFeatures
        parser = JsonOutputParser(pydantic_object=pydantic_parser)
        prompt_template = PromptTemplate(input_variables=["employement_history"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template, output_parser=parser, verbose=True)
        result = chain.invoke(inputs)
        return result["text"]
    
import pandas as pd
class AssetsLiabilitiesParser:
    def get_wealth_assessment(self,features_df: pd.DataFrame) -> dict:
        """
        Calculate numeric wealth assessment features from an assets/liabilities DataFrame.
        
        Args:
            features_df (pd.DataFrame): DataFrame with columns ['Type', 'Category', 'Description', 'Amount'].
                                    'Type' is either 'Asset' or 'Liability'.
                                    'Amount' is numeric, assets positive, liabilities negative.
        
        Returns:
            dict: Dictionary containing numeric features:
                - total_assets (float): Sum of all asset amounts.
                - total_liabilities (float): Sum of absolute values of all liability amounts.
                - net_worth (float): total_assets - total_liabilities.
                - num_assets (int): Count of asset entries.
                - num_liabilities (int): Count of liability entries.
                - asset_to_liability_ratio (float): total_assets / total_liabilities (0 if liabilities are 0).
                - property_asset_value (float): Sum of 'Property' category assets.
                - investment_asset_value (float): Sum of 'Investment' category assets.
        """
        df = features_df.copy()
        
        # Ensure Amount is numeric
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        # Filter assets and liabilities
        assets = df[df['Type'].str.lower() == 'asset'].copy()
        liabilities = df[df['Type'].str.lower() == 'liability'].copy()
        
        # Total assets and liabilities
        total_assets = assets['Amount'].sum()
        total_liabilities = liabilities['Amount'].abs().sum()
        
        # Net worth
        net_worth = total_assets - total_liabilities
        
        # Counts
        num_assets = len(assets)
        num_liabilities = len(liabilities)
        
        # Asset-to-liability ratio
        if total_liabilities == 0:
            asset_to_liability_ratio = float('inf') if total_assets > 0 else 0.0
        else:
            asset_to_liability_ratio = total_assets / total_liabilities
        
        # Category-specific sums
        property_asset_value = assets[assets['Category'].str.lower() == 'property']['Amount'].sum()
        investment_asset_value = assets[assets['Category'].str.lower() == 'investment']['Amount'].sum()
        
        return {
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "net_worth": net_worth,
            "num_assets": num_assets,
            "num_liabilities": num_liabilities,
            "asset_to_liability_ratio": asset_to_liability_ratio,
            "property_asset_value": property_asset_value,
            "investment_asset_value": investment_asset_value
        }



