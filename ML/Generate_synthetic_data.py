import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=500):
    """
    Generate a synthetic dataset with realistic distributions for training.
    
    Columns:
    - declared_income (float)
    - No_of_companies_worked (int)
    - current_employer_tenure_months (int)
    - earliest_start_year (int)
    - total_experience (int)
    - average_employement_tenure_months (float)
    - total_assets (float)
    - total_liabilities (float)
    - net_worth (float)
    - num_assets (int)
    - num_liabilities (int)
    - asset_to_liability_ratio (float)
    - property_asset_value (float)
    - investment_asset_value (float)
    - family_size (int)
    - dependents (int)
    - Age (int)
    - Gender (str)
    - marital_status (str)
    - Disability (str)
    - Housing_type (str)
    - Social_support (int)
    """
    np.random.seed(42)

    # Demographics
    ages = np.random.randint(18, 70, size=n_samples)
    genders = np.random.choice(
        ['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.48, 0.04]
    )
    marital_statuses = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed', 'Other'],
        size=n_samples,
        p=[0.4, 0.45, 0.1, 0.03, 0.02]
    )
    disabilities = np.random.choice(['No', 'Yes'], size=n_samples, p=[0.9, 0.1])
    housing_types = np.random.choice(
        ['Renting', 'Owned', 'Living with Family', 'Other'],
        size=n_samples,
        p=[0.5, 0.3, 0.15, 0.05]
    )

    # Family
    family_sizes = np.random.randint(1, 8, size=n_samples)
    dependents = np.minimum(
        family_sizes - 1,
        np.maximum(0, np.random.poisson(lam=1.5, size=n_samples))
    )

    # Employment history
    No_of_companies = np.random.randint(1, 6, size=n_samples)
    # total_experience in years (approx), clipped to non-negative integer
    total_experience = np.clip(
        (ages - 18) * np.random.uniform(0.5, 0.9, size=n_samples),
        0,
        None
    ).astype(int)
    # average tenure (years) per company
    average_tenure = (total_experience / No_of_companies).round(1)

    # Ensure randint bounds are valid: high must be > low
    # We'll compute an integer high bound = max(floor(average_tenure) + 1, 2)
    avg_tenure_int = np.floor(average_tenure).astype(int)
    high_bounds = np.maximum(avg_tenure_int + 1, 2)

    # Random tenure (years) for current employer, then clamp by total_experience
    rand_tenure = np.array([
        np.random.randint(1, hb) for hb in high_bounds
    ])
    current_employer_tenure = np.minimum(rand_tenure, total_experience)

    # Earliest start year (assuming total_experience is in years)
    earliest_start_year = 2025 - total_experience - np.random.randint(0, 2, size=n_samples)

    # Income (AED per month), correlates with experience
    declared_income = np.clip(
        np.random.normal(loc=5000 + (total_experience * 10), scale=1500),
        1000,
        None
    )

    # Assets & Liabilities (AED)
    total_assets = np.clip(
        np.random.lognormal(
            mean=np.log(50000 + (declared_income * 6)),
            sigma=0.75,
            size=n_samples
        ),
        1000,
        None
    )
    total_liabilities = np.clip(
        total_assets * np.random.uniform(0.1, 0.5, size=n_samples),
        0,
        total_assets
    )
    net_worth = total_assets - total_liabilities
    num_assets = np.random.randint(1, 6, size=n_samples)
    num_liabilities = np.random.randint(0, 4, size=n_samples)
    asset_to_liability_ratio = np.where(
        total_liabilities == 0,
        np.inf,
        total_assets / total_liabilities
    )
    property_asset_value = np.clip(
        total_assets * np.random.uniform(0.3, 0.7, size=n_samples),
        0,
        None
    )
    investment_asset_value = np.clip(
        total_assets * np.random.uniform(0.1, 0.3, size=n_samples),
        0,
        None
    )

    # Social support label: lower net worth & income → higher chance of 1
    support_prob = 1 / (1 + np.exp((net_worth + declared_income * 6 - 200000) / 50000))
    Social_support = (np.random.rand(n_samples) < support_prob).astype(int)

    data = pd.DataFrame({
        'declared_income': declared_income,
        'No_of_companies_worked': No_of_companies,
        'current_employer_tenure_months': current_employer_tenure,
        'earliest_start_year': earliest_start_year,
        'total_experience': total_experience,
        'average_employement_tenure_months': average_tenure,
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'net_worth': net_worth,
        'num_assets': num_assets,
        'num_liabilities': num_liabilities,
        'asset_to_liability_ratio': asset_to_liability_ratio,
        'property_asset_value': property_asset_value,
        'investment_asset_value': investment_asset_value,
        'family_size': family_sizes,
        'dependents': dependents,
        'Age': ages,
        'Gender': genders,
        'marital_status': marital_statuses,
        'Disability': disabilities,
        'Housing_type': housing_types,
        'Social_support': Social_support
    })

    return data
