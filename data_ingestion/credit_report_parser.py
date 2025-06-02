# application/data_ingestion/credit_report_parser.py

import pdfplumber
from io import BytesIO

from Utils.langchain_general_method import CreditReportParser
def process_credit_report(uploaded_file) -> str:
    """
    Reads an uploaded PDF (credit report) and returns a single string
    containing all the text (page by page). If a page has no extractable
    text, it will append an empty line for that page.

    You can pass the returned string straight into an LLM for further parsing.
    """
    # Read the uploaded file’s bytes
    file_bytes = uploaded_file.read()

    all_text = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""   # extract_text() may return None
            all_text.append(page_text)

    # Join with double‐newline between pages
    full_text = "\n\n".join(all_text)

    # print('Full Text from Credit Report Parser----------\n\n',full_text)

    parser_obj = CreditReportParser()
    parsed = parser_obj.parse_credit_report_text_llm(full_text,'credit_report_info_capture')
    # print('Parsed from LLM----------\n\n',parsed)
    # Convert the string to a pandas DataFrame

    return parsed
