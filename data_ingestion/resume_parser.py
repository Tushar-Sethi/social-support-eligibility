# application/data_ingestion/resume_parser.py

import re
import pdfplumber
import docx
import pandas as pd
from io import BytesIO

from Utils.langchain_general_method import ResumeParser


def process_resume(uploaded_file) -> dict:
    """
    Entry‐point for resume parsing. Detects file extension and
    dispatches to the correct loader. Returns a dictionary of
    extracted fields: {
        "text": <full raw text>,
        "name": <best‐guess name or None>,
        "email": <found email or None>,
        "phone": <found phone number or None>,
        "skills": [list of skills strings] or [],
        "education": [list of education lines] or [],
        "experience": [list of experience lines] or []
    }
    """
    filename = uploaded_file.name.lower()
    # Read the uploaded file’s bytes
    file_bytes = uploaded_file.read()
    if filename.endswith(".pdf"):
        raw_text = _extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        raw_text = _extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported resume file type: {filename}")

    
    parser_obj = ResumeParser()
    parsed = parser_obj.parse_resume_text_llm(raw_text,'resume_info_capture')
    return parsed


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Uses pdfplumber to extract all text from a PDF (page by page).
    Returns a single string with newlines between pages.
    """
    all_pages = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_pages.append(text)
    return "\n\n".join(all_pages)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Uses python‐docx to extract text from a DOCX. Returns the concatenated
    text from all paragraphs, with newlines between paragraphs.
    """
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


# def _parse_resume_text(raw_text: str) -> dict:
#     """
#     Given the full raw text of a resume, run regex/heuristic-based
#     parsing to extract common fields: name, email, phone, skills, education, experience.
#     Returns a dict with those keys.
#     """
#     # Normalize whitespace
#     text = re.sub(r"\r\n?", "\n", raw_text)  # unify line breaks
#     lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

#     # 1) EMAIL: simple regex
#     email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
#     email_matches = email_pattern.findall(text)
#     email = email_matches[0] if email_matches else None

#     # 2) PHONE: look for international or common formats
#     phone_pattern = re.compile(
#         r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"
#     )
#     phone_matches = phone_pattern.findall(text)
#     phone = None
#     if phone_matches:
#         # phone_matches is a list of tuples from the capturing groups; reconstruct first full match
#         # Simplest: re.search full match on original text
#         m = re.search(phone_pattern, text)
#         phone = m.group(0) if m else None

#     # 3) NAME: heuristic—first non‐sectioned line that’s not “Resume” or “Curriculum Vitae”
#     name = None
#     for ln in lines[:5]:  # assume name is in the first 5 non-empty lines
#         if re.search(r"resume|curriculum vitae", ln, flags=re.I):
#             continue
#         # If the line has more than one word and no digits, treat it as name
#         if len(ln.split()) <= 5 and not any(char.isdigit() for char in ln):
#             name = ln
#             break

#     # 4) SKILLS: look for a “Skills” or “Technical Skills” section and gather subsequent bullet lines
#     skills = []
#     for idx, ln in enumerate(lines):
#         if re.search(r"\bskills?\b", ln, flags=re.I):
#             # Collect next few lines until an empty line or another section header (all-caps or ends with colon)
#             for subln in lines[idx + 1:]:
#                 if not subln or re.match(r"^[A-Z\s]{3,}$", subln) or subln.endswith(":"):
#                     break
#                 # Split on commas if comma‐separated list, else take the whole line
#                 if "," in subln:
#                     skills.extend([s.strip() for s in subln.split(",") if s.strip()])
#                 else:
#                     skills.append(subln.strip())
#             break

#     # Deduplicate skill entries
#     skills = list(dict.fromkeys(skills))

#     # 5) EDUCATION: look for “Education” section, gather until next section
#     education = []
#     for idx, ln in enumerate(lines):
#         if re.search(r"\beducation\b", ln, flags=re.I):
#             for subln in lines[idx + 1:]:
#                 if not subln or re.match(r"^[A-Z\s]{3,}$", subln) or subln.endswith(":"):
#                     break
#                 education.append(subln.strip())
#             break

#     # 6) EXPERIENCE: look for “Experience” or “Work Experience” section
#     experience = []
#     for idx, ln in enumerate(lines):
#         if re.search(r"\b(experience|work experience|professional experience)\b", ln, flags=re.I):
#             for subln in lines[idx + 1:]:
#                 if not subln or re.match(r"^[A-Z\s]{3,}$", subln) or subln.endswith(":"):
#                     break
#                 experience.append(subln.strip())
#             break

#     return {
#         "name": name,
#         "email": email,
#         "phone": phone,
#         "skills": skills,
#         "education": education,
#         "experience": experience
#     }
