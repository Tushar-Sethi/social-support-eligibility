# application/data_ingestion/assets_liabilities_parser.py

import pandas as pd

def process_assets_liabilities(uploaded_file):
    """
    Reads an uploaded CSV or Excel file containing Assets & Liabilities.
    Returns two DataFrames: (assets_df, liabilities_df).
    
    Expected columns in the file:
      - Type: values "Asset" or "Liability"
      - Category: e.g. "Cash", "Loan", etc.
      - Description: description of the item
      - Amount: numeric (positive for assets, negative or positive for liabilities)
    
    If the file is CSV:
      • pd.read_csv(uploaded_file) is used.
      • Filters rows by 'Type' == "Asset" or "Liability".
    
    If the file is Excel (.xls or .xlsx):
      • pd.read_excel(uploaded_file, sheet_name=None) is used to read all sheets.
      • If there are sheets named "Assets" and "Liabilities", it loads each directly.
      • Otherwise, it reads the first sheet and splits on the "Type" column.
    
    Raises ValueError if the format is unsupported or required columns are missing.
    """
    filename = uploaded_file.name.lower()
    
    # Helper to normalize the DataFrame: ensure 'Type' column exists, drop NaNs
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if 'Type' not in df.columns:
            raise KeyError("Missing required 'Type' column in assets/liabilities file.")
        df = df.dropna(subset=['Type'])  # drop rows without a Type
        # Standardize Type values
        df['Type'] = df['Type'].astype(str).str.strip().str.capitalize()
        return df
    
    # 1) Handle CSV
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            raise ValueError(f"Unable to read CSV: {e}")
        df = _normalize_df(df)
        return df    
    # 2) Handle Excel (.xls or .xlsx)
    elif filename.endswith(('.xls', '.xlsx')):
        try:
            # Read all sheets into a dict of DataFrames
            xls = pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            xls = pd.DataFrame()
            # raise ValueError(f"Unable to read Excel file: {e}")
        # If there are sheets explicitly named "Assets" and "Liabilities", use them
        return xls
        # If there are sheets explicitly named "Assets" and "Liabilities", use them
        # sheets = {name.lower(): df for name, df in xls.items()}
        # if 'assets' in sheets and 'liabilities' in sheets:
            # assets_df = sheets['assets'].copy().reset_index(drop=True)
            # liabilities_df = sheets['liabilities'].copy().reset_index(drop=True)
            
            # No need to filter by Type since sheets are separate
            # return assets_df, liabilities_df
        
        # Otherwise, load the first sheet and split by 'Type'
        # first_sheet_df = list(xls.values())[0]
        # df = _normalize_df(first_sheet_df)
        # assets_df     = df[df['Type'] == 'Asset'].copy().reset_index(drop=True)
        # liabilities_df = df[df['Type'] == 'Liability'].copy().reset_index(drop=True)
        # return assets_df, liabilities_df
    
    else:
        raise ValueError(
            f"Unsupported file type '{uploaded_file.name}'. "
            "Please upload a .csv, .xls, or .xlsx file."
        )
