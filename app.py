import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from Data_Cleaning_Pipeline import clean_and_engineer_features
import pickle

# Establishing constants
TARGETS = ["ATTACHD", "CONDO", "DETACHD"]
CITIES = ["Portland", "Beaverton", "Hillsboro", "Lake Oswego", "West Link"]
COMPANY_LOGO = st.image("https://content.mediastg.net/static/RealEstate/company/3/012-logo.png")

st.header("Predict Property Type")

st.write(
    """
    📄 **Instructions:**
    - The file must be a `.csv` format.
    - Include column headers in the first row.
    - Columns should include: 'MLS#', 'Type', 'Prop. Cat.', 'Prop. Cond.', 'Tax', 
        'Address', 'City', 'Zip', 'Area', 'BD', 'Baths', '# Levels', 'Apx Sqft',
        'Price SqFt', 'Sld Price Sqft', 'Lot Size', 'Pend. Date', 'DOM', 'CDOM',
        'List Date', 'List Price', 'Sold Date', 'Price', 'Yr. Built', 'HOA Dues',
        '# Garage', '# Fireplaces', 'Terms'
    """
)
# Loading model and transformers
with open("./model_artifacts.pkl", 'rb') as f:
    model_artifacts = pickle.load(f)

encoder = EncodingPipeline()
with open("./ohe.pkl", 'rb') as f:
    ohe = pickle.load(f)

with open("./scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("./label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

num_cols = model_artifacts["numeric_features"]
cat_cols = model_artifacts["categorical_features"]
feature_order = model_artifacts["feature_names"]

file = st.file_uploader(".csv file with property data", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df)
    
    unknown_cities = df.query("City not in @CITIES")["City"]
    if unknown_cities.count() > 0:
        st.warning(
            f"Found City values not recognized by model. Removing row as consequence.\nCities found: {unknown_cities}",
            icon = "⚠️",
            width = "stretch"
        )
    if df["City"].isna().sum() > 0:
        st.warning(
            "City data is missing values. Rows with missing 'City' values will be removed.",
            icon = "⚠️",
            width="stretch"
        )
    if df["Zip"].isna().sum() > 0:
        st.warning(
            "Zip Code data is missing values. Rows with missing 'Zip' values will be removed.",
            icon = "⚠️",
            width="stretch"
        )
    if df["Area"].isna().sum() > 0:
        st.warning(
            "Area data is missing values. Rows with missing 'Area' values will be removed.",
            icon = "⚠️",
            width="stretch"
        )
    if df["Yr. Built"].isna().sum() > 0:
        st.warning(
            "Year built data is missing values. Rows with missing 'Yr. Built' values will be removed.",
            icon = "⚠️",
            width="stretch"
        )
    if df["Prop. Cond."].isna().sum() > 0:
        st.markdown(
            ":orange-badge[⚠️ Prop. Cond. is missing values can negatively affect model accuracy. Please fill in values if possible.]"
        )
    if df["Tax"].isna().sum() > 0:
        st.markdown(
            ":orange-badge[⚠️ Tax data is missing values can negatively affect model accuracy. Please fill in values if possible.]"
        )
    if df["Apx Sqft"].isna().sum() > 0:
        st.markdown(
            ":orange-badge[⚠️ Aprox. sq footage data is missing values can negatively affect model accuracy. Please fill in values if possible.]"
        )
    if df["HOA Dues"].isna().sum() > 0:
        st.markdown(
            ":orange-badge[⚠️ HOA Dues missing values can negatively affect model accuracy. Please fill in values if possible.]"
        )
    
    if st.button("Run Predictions"):
        st.write("Loading Model...")
        # Asigning model to variable from aodel artifacts
        model = model_artifacts["model"]

        # Cleaning and engineering fewatures
        X = clean_and_engineer_features(df)

        # Transforming features
        X[num_cols] = scaler.transform(X[num_cols])
        ohe_feat = ohe.transform(X[cat_cols])
        ohe_cols = ohe.get_feature_names_out()
        ohe_df = pd.DataFrame(ohe_feat, columns=ohe_cols, index=X.index)
        X = pd.concat([X.drop(cat_cols, axis=1), ohe_df], axis=1)
        X = X.reindex(columns=feature_order) # Re-ordering columns to match trained model column 
        st.write("Running Predictions...")
        st.success("Prediction Probabilities by Property Type")
        df["Predicted Type"] = le.inverse_transform(model.predict(X)) # Decoding results for readability
        st.dataframe(df[["Address", "City", "Zip", "Area", "BD", "Baths", "Predicted Type"]])

        # Download dataframe with predictions to csv
        csv = df.to_csv(index=False) # Converting dataframe to csv for download
        st.download_button(
            label = "Download CSV",
            data = csv,
            file_name = "Dataset_Predictions.csv",
            mime = "text/csv",
            on_click = "ignore",
            icon = ":material/download:"
        )
