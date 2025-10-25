import pandas as pd
import numpy as np
import re

# Data cleaning and feature-engineering pipeline
def clean_and_engineer_features(df):
    """
    Comprehensive data cleaning and feature engineering pipeline for real estate data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw real estate dataframe with original column names
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned and feature-engineered dataframe
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # ========== STEP 1: DROP UNNECESSARY COLUMNS ==========
    cols_to_drop = ['MLS#', 'Prop. Cat.', 'Tax', 'Address', 'Price SqFt',
                    'Sld Price Sqft', 'Lot Size', 'Pend. Date', 'List Date', 
                    'CDOM', 'Sold Date', 'Terms']
    
    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=cols_to_drop)
    
    # ========== STEP 2: RENAME COLUMNS ==========
    rename_mapping = {
        "Type": "property_type",
        "Prop. Cond.": "property_condition",
        "City": "city",
        "Zip": "zip_code",
        "Area": "area",
        "BD": "bedrooms",
        "Baths": "bathrooms",
        "# Levels": "num_levels",
        "Apx Sqft": "approx_sqft",
        "DOM": "days_on_market",
        "List Price": "list_price",
        "Price": "price",
        "Yr. Built": "year_built",
        "HOA Dues": "hoa_dues",
        "# Garage": "num_garage",
        "# Fireplaces": "num_fireplaces"
    }
    
    # Only rename columns that exist
    rename_mapping = {k: v for k, v in rename_mapping.items() if k in data.columns}
    data = data.rename(columns=rename_mapping)
    
    # ========== STEP 3: CLEAN AREA COLUMN ==========
    if 'area' in data.columns:
        data['area'] = (
            data['area']
            .astype(str)
            .str.replace("$", " ", regex=False)
            .str.strip()
            .str.normalize('NFKC')
            .str.replace(".00", "", regex=False)
        )
    
    # ========== STEP 4: CLEAN PRICE COLUMNS ==========
    for col in ['list_price', 'price']:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace(".00", "", regex=False)
            )
    
    # ========== STEP 5: REMOVE OUTLIERS ==========
    # Remove rows where num_garage == 14
    if 'num_garage' in data.columns:
        data = data[data["num_garage"] != 14].reset_index(drop=True)
    
    # ========== STEP 6: APPLY LOWERCASE TO STRING COLUMNS ==========
    data = data.map(lambda x: x.lower() if isinstance(x, str) else x)
    
    # ========== STEP 7: NORMALIZE PROPERTY TYPE ==========
    if 'property_type' in data.columns:
        data['property_type'] = data['property_type'].astype(str).str.strip().str.lower()
    
    # ========== STEP 8: CLEAN AND IMPUTE HOA DUES ==========
    if 'hoa_dues' in data.columns:
        # Robust numeric parse
        data['hoa_dues'] = (
            data['hoa_dues'].astype(str)
            .str.replace(r'(?i)\$|,|usd|/mo|per\s*month|monthly', '', regex=True)
            .str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
            .pipe(pd.to_numeric, errors='coerce')
        )
        
        # Compute median per property_type and impute
        if 'property_type' in data.columns:
            medians = data.groupby('property_type')['hoa_dues'].median()
            data['hoa_dues'] = data.apply(
                lambda row: medians.get(row['property_type'], 0) if pd.isna(row['hoa_dues']) else row['hoa_dues'],
                axis=1
            )
    
    # ========== STEP 9: FILL MISSING VALUES ==========
    if 'num_fireplaces' in data.columns:
        data['num_fireplaces'] = data['num_fireplaces'].fillna(0)

    if 'hoa_dues' in data.columns:
        data['hoa_dues'] = data['hoa_dues'].fillna(0)
    
    if 'property_condition' in data.columns:
        data['property_condition'] = data['property_condition'].fillna('resale')
    
    if 'num_levels' in data.columns:
        data['num_levels'] = data['num_levels'].fillna(1)
    
    # ========== STEP 10: DROP ROWS WITH MISSING CRITICAL DATA ==========
    if 'approx_sqft' in data.columns:
        data = data.dropna(subset=['approx_sqft'])
    
    # Drop rows with 'bd' in bedrooms column
    if 'bedrooms' in data.columns:
        data = data[~data['bedrooms'].astype(str).str.contains(r'\bbd\b', case=False, na=False)]
    
    # ========== STEP 11: SPLIT BATHROOMS INTO FULL AND HALF ==========
    if 'bathrooms' in data.columns:
        bath_split = data['bathrooms'].astype(str).str.split('.', expand=True)
        data['full_bath'] = pd.to_numeric(bath_split[0], errors='coerce').astype("Int64")
        data['half_bath'] = pd.to_numeric(bath_split[1], errors='coerce').fillna(0).astype("Int64")
        data = data.drop(columns=['bathrooms'])
    
    # ========== STEP 12: CONVERT COLUMNS TO INTEGER ==========
    cols_to_int = [
        'bedrooms', 'num_levels', 'approx_sqft', 'days_on_market', 'list_price', 
        'price', 'num_garage', 'num_fireplaces'
    ]
    
    for col in cols_to_int:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")
    
    # ========== STEP 13: CONVERT YEAR BUILT ==========
    if 'year_built' in data.columns:
        data['year_built'] = pd.to_numeric(data['year_built'], errors='coerce').astype('Int64')
    
    # ========== STEP 14: CONVERT TO CATEGORICAL ==========
    categorical_cols = ["property_type", "property_condition", "city", "area"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")
    
    # ========== STEP 15: NORMALIZE ZIP CODE ==========
    if "zip_code" in data.columns:
        zip_num = pd.to_numeric(data["zip_code"], errors="coerce").round().astype("Int64")
        data["zip_code"] = zip_num.astype("string").str.zfill(5).astype("category")
    
    # ========== STEP 16: REMOVE UNWANTED PROPERTY TYPES ==========
    if 'property_type' in data.columns:
        types_to_remove = ["in-park", "flthome", "res-mfg", "plncomm"]
        data = data.loc[~data["property_type"].isin(types_to_remove)].copy()
        data["property_type"] = data["property_type"].cat.remove_unused_categories()
    
    # ========== STEP 17: CLEAN CITY COLUMN ==========
    if 'city' in data.columns:
        data['city'] = data['city'].str.strip().str.lower()
        
        # Remove specific cities
        cities_to_drop = [
            "forest grove", "cornelius", "aloha", "gaston",
            "tualatin", "gresham", "gales creek", "milwaukie", "newberg"
        ]
        data = data[~data['city'].isin(cities_to_drop)].reset_index(drop=True)
    
    # ========== STEP 18: FEATURE ENGINEERING - BATH TO BED RATIO ==========
    if 'full_bath' in data.columns and 'half_bath' in data.columns and 'bedrooms' in data.columns:
        total_bathrooms = data['full_bath'] + 0.5 * data['half_bath']
        data['bath_to_bed_ratio'] = np.where(
            data['bedrooms'] > 0,
            total_bathrooms / data['bedrooms'],
            0
        )
        data['bath_to_bed_ratio'] = data['bath_to_bed_ratio'].round(2)
    
    # ========== STEP 19: FEATURE ENGINEERING - PROPERTY AGE ==========
    if 'year_built' in data.columns:
        data['property_age'] = 2025 - data['year_built']
        data['property_age'] = data['property_age'].astype('category')
    
    # ========== STEP 20: FEATURE ENGINEERING - ZIP PREFIX GROUPS ==========
    if 'zip_code' in data.columns:
        # Convert zip_code to numeric for prefix extraction
        zip_numeric = pd.to_numeric(data['zip_code'], errors='coerce')
        zip_prefix = zip_numeric.dropna().astype(int).astype(str).str[:3]
        
        # Classify by prefix
        def classify_zip_prefix(prefix):
            if pd.isna(prefix):
                return 'other'
            elif prefix == '972':
                return 'urban_portland'
            elif prefix == '970':
                return 'suburban_west_south'
            elif prefix == '971':
                return 'suburban_northwest'
            else:
                return 'other'
        
        data['zip_prefix_group'] = zip_prefix.apply(classify_zip_prefix)
        data['zip_prefix_group'] = data['zip_prefix_group'].astype('category')
    
    return data


# === USAGE ===
#Dataset_New = clean_and_engineer_features(Dataset_New)