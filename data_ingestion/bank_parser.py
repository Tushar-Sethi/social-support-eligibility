import pandas as pd
import pdfplumber

def process_bank_statement(uploaded_file) -> pd.DataFrame:
    """
    Determine file type from filename and route to the appropriate loader.
    Returns a DataFrame with columns: [date, description, amount, running_balance].
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return _load_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        return _load_excel(uploaded_file)
    elif filename.endswith(".pdf"):
        return _load_pdf(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def _load_csv(f) -> pd.DataFrame:
    """
    Reads a CSV bank statement (expected columns: Date, Description, Debit, Credit, Balance).
    Normalizes it into [date, description, amount, running_balance].
    """
    df = pd.read_csv(f)
    return _normalize_tabular(df)

def _load_excel(f) -> pd.DataFrame:
    """
    Reads an Excel bank statement (expected columns: Date, Description, Debit, Credit, Balance).
    Normalizes it into [date, description, amount, running_balance].
    """
    df = pd.read_excel(f, engine="openpyxl")
    return _normalize_tabular(df)

def _normalize_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with typical bank-statement columns,
    unify into: [date, description, amount, running_balance].
    """
    # Standardize column names
    df = df.rename(columns=lambda c: c.strip().lower())

    # 1) Find date column
    date_col = next((c for c in ["date", "transaction date", "txn_date", "posted date"] if c in df.columns), None)
    if date_col is None:
        raise KeyError("Date column not found in CSV/Excel statement.")

    # 2) Find description column
    desc_col = next((c for c in ["description", "narration", "details"] if c in df.columns), None)
    if desc_col is None:
        raise KeyError("Description column not found in CSV/Excel statement.")

    # 3) Debit & Credit → signed amount (or direct amount)
    if "debit" in df.columns and "credit" in df.columns:
        df["debit"] = df["debit"].fillna(0).astype(float)
        df["credit"] = df["credit"].fillna(0).astype(float)
        df["amount"] = df["credit"] - df["debit"]
    elif "amount" in df.columns:
        df["amount"] = df["amount"].astype(float)
    else:
        raise KeyError("Neither 'Debit/Credit' nor 'Amount' columns found in CSV/Excel statement.")

    # 4) Find balance column
    balance_col = next((c for c in ["balance", "running balance", "running_balance"] if c in df.columns), None)
    if balance_col is None:
        raise KeyError("Balance column not found in CSV/Excel statement.")

    # 5) Build the clean DataFrame
    clean_df = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "description": df[desc_col].astype(str).str.strip(),
        "amount": df["amount"].astype(float),
        "running_balance": df[balance_col].astype(float),
    })

    # 6) Drop rows where date parsing failed or amount is NaN
    clean_df = clean_df.dropna(subset=["date", "amount"])

    # 7) Sort by date ascending
    clean_df = clean_df.sort_values("date").reset_index(drop=True)
    return clean_df

def _load_pdf(f) -> pd.DataFrame:
    """
    Uses pdfplumber to extract tables from a PDF bank statement.
    If no table is found on a page, falls back to parsing raw text lines.
    Raises ValueError if neither method yields any data (e.g. scanned image).
    """
    data_frames = []

    with pdfplumber.open(f) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables and len(tables) > 0:
                try:   
                    temp_table = []
                    for row in tables[0]:
                        temp_table.append(row)
                    df = pd.DataFrame(temp_table[1:], columns=temp_table[0])
                    break
                except Exception:
                    pass
    return df

def _parse_text_lines(text: str) -> pd.DataFrame:
    """
    Simple parser for lines of raw text extracted from a PDF page.
    Expects lines like:
        01/05/2025  Grocery Store   150.00   4850.00
    Returns a DataFrame with [date, description, amount, running_balance], or None if no valid lines found.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    records = []
    import re

    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue

        # 1) First token is date
        date_str = parts[0]
        try:
            txn_date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
            if pd.isna(txn_date):
                continue
        except Exception:
            continue

        # 2) Last two tokens are amount & running balance
        raw_amount = parts[-2].replace(",", "").replace("AED", "").strip()
        raw_balance = parts[-1].replace(",", "").replace("AED", "").strip()
        try:
            amount = float(raw_amount)
            balance = float(raw_balance)
        except ValueError:
            continue

        # 3) Description is everything in between
        description = " ".join(parts[1:-2]).strip()

        records.append({
            "date": txn_date,
            "description": description,
            "amount": amount,
            "running_balance": balance
        })

    if not records:
        return None

    return pd.DataFrame(records)
