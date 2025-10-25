import pandas as pd
import re

# data preprocessing function
def preprocessor(data: pd.DataFrame)->pd.DataFrame:
    # Making sure feature shape is correct
    if data.shape[1] != 28:
        raise ValueError(f"Input is missing data. Number of features should be 25. Shape {data.shape}")
    
    # Standaridizing column name format
    data.columns = [re.sub(r"\.","", re.sub(r" ", "_", str(x).lower())) for x in data.columns]
    
    # Making sure features are correct 
    col_names = ['mls#', 'type', 'prop_cat', 'prop_cond', 'tax', 'address', 'city',
       'zip', 'area', 'bd', 'baths', '#_levels', 'apx_sqft', 'price_sqft',
       'sld_price_sqft', 'lot_size', 'pend_date', 'dom', 'cdom', 'list_date',
       'list_price', 'sold_date', 'price', 'yr_built', 'hoa_dues', '#_garage',
       '#_fireplaces', 'terms']
    unknown_cols =  set(col_names) - set(data.columns)
    if unknown_cols:
        raise ValueError(f"Input data columns not recognized: Input cols \nExpected:{col_names}\nData Cols: {data.columns}")    

    # Filtering columns
    try:
        # column names to keep
        keep = ["prop_cond", "type", "city", "zip", "area", "bd", "baths", "#_levels", "apx_sqft", "yr_built", "hoa_dues", "#_garage", "#_fireplaces"]
        data = data[keep] 
    except Exception as e:
        print(f"Error filtering data: {e}")

    # Keeping desires target property types
    prop_types = ["ATTACHD", "DETACHD", "CONDO"]
    data = data[data["type"].isin(prop_types)]
    
    # Keeping rows with desired cities
    cities = ["Portland", "Beaverton", "Hillsboro", "Lake Oswego", "West Linn"]
    data = data[data["city"].isin(cities)]

    # Dropping missing sqft rows
    data = data.dropna(subset="apx_sqft", axis=0)

    # Consolidating prop_cond unique values 
    data["prop_cond"] = data["prop_cond"].replace(
    ["UNKNOWN", "UNDRCON", "REGHIST", "EXISTNG", "PROPOSD"],
    "OTHER"
)

        # Checking for significant feature missing values
    prop_cond_values = ["RESALE", "REMOD", "APPROX", "NEW", "FIXER", "RESTORD"]
    unknown_cond = set(data["prop_cond"]) - set(prop_cond_values)
    if unknown_cond:
        print("WARNING: Submitting missing values for property condition. \nThis may negatively affect prediction accuracy")
        data["prop_cond"] = data["prop_cond"].fillna("OTHER")
    
    # Cities model is trained on
    cities = ["Portland", "Beaverton", "Hillsboro", "Lake Oswego", "West Linn"]
    unknown_cities = set(data["city"]) - set(cities)
    if unknown_cities:                              
        raise ValueError(f"City not recognized. Unrecognized cities: {unknown_cities}")

    if data["zip"].isna().any():
        raise ValueError("Please include zip code(s)") 
    
    if data["area"].isna().any():
        raise ValueError("Please include area value")
        
    if data["bd"].isna().any():
        raise ValueError("Please include number of bedrooms")
    
    if data["baths"].isna().any():
        raise ValueError("Please include number of baths (0.1 for each partial bathroom)")
        
    if data["#_levels"].isna().any():
        print("WARNING: Missing number of levels values. \nThis may negatively affect prediction accuracy")
        data["#_levels"] = data["#_levels"].fillna('1')
    
    if data["apx_sqft"].isna().any():
        raise ValueError("Please include approximate square footage")
    
    if data["yr_built"].isna().any():
        raise ValueError("Please include property year built")

    if data["hoa_dues"].isna().any():
        print("WARNING: Missing hoa dues values. \nThis may negatively affect prediction accuracy")
        data["hoa_dues"] = data["hoa_dues"].fillna('0')
    
    if data["#_garage"].isna().any():
        print("WARNING: Missing number of gaarage values can negatively affect model accuracy")
        data["#_garage"] = data["#_garage"].fillna('0')

    if data["#_fireplaces"].isna().any():
        print("WARNING: Missing number of fireplaces can negatively affect model accuracy")
        data["#_fireplaces"] = data["#_fireplaces"].fillna('0')
    
    # Converting dtypes to appropriate types
    cat = ["prop_cond", "type", "city", "area"]
    flt = "baths"
    num = ["zip", "bd", "#_levels", "apx_sqft", "yr_built", "hoa_dues", "#_garage", "#_fireplaces"]
    
    for col in cat:
        data[col] = data[col].astype("category")
    
    data[flt] = data[flt].astype("float")

    data["area"] = [str(x).strip('$') for x in data["area"]] # Removing '$' found in values

    for col in num:
        data[col] = pd.to_numeric(data[col],downcast="integer")
    
    return data.copy()

# Feature Engineering function
def new_features(data: pd.DataFrame) -> pd.DataFrame:
    # Splitting number of bathrooms to 2 features, `"full_baths"` & `"half_baths"`
    data["full_baths"] = data["baths"].astype("int")
    # Creating `"half_baths"` feature
    data["half_baths"] = ((data["baths"] - data["full_baths"]) * 10).astype("int")
    # Dropping `"baths"` feature
    data = data.drop("baths", axis=1)

    # Generating new feature accounting for `"total_rooms"`
    data["total_rooms"] = (data["bd"] + data["full_baths"] + data["half_baths"]).astype("int")
    
    # Creating new feature calculating aproximate average sqft per room
    data["apx_room_sqft"] = (data["apx_sqft"] / data["total_rooms"]).astype("float")

    # Creating `"avg_level_sqft"` feature 
    data["avg_level_sqft"] = data["apx_sqft"] / data["#_levels"]

    # Converting new column to float dtype
    data["avg_level_sqft"] = data["avg_level_sqft"].astype("float")
    
    return data
