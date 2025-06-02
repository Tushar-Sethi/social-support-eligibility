

# application/data_ingestion/emirates_id_parser.py

import pdfplumber
from io import BytesIO
from Utils.langchain_general_method import EmiratedIDParser

def extract_emirates_id_text(uploaded_file) -> str:
    """
    Reads an uploaded Emirates ID PDF and returns all the extractable text (concatenated page by page).
    If a page has no extractable text, it will append an empty string for that page.

    Usage:
        raw_text = extract_emirates_id_text(my_pdf_file)
        # then pass raw_text into your LLM for parsing fields like ID number, name, etc.
    """
    # Read the uploaded PDF’s bytes into memory
    file_bytes = uploaded_file.read()
    all_pages = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_pages.append(text)

    full_text = "\n\n".join(all_pages)
    parser_obj = EmiratedIDParser()
    parsed = parser_obj.parse_Emirated_ID_llm(full_text,'Emirated_ID_info_capture')
    return parsed
