import pandas as pd
import numpy as np
import re
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from Data_Cleaning_Pipeline import clean_and_engineer_features
from Data_Encoding_Pipeline import EncodingPipeline
import pickle

df = pd.read_csv("../sample_data/raw_minus_sample.csv")

df = clean_and_engineer_features(df.drop("Unnamed: 0", axis=1))

# Initializing encoder
e_pipe = EncodingPipeline()
e_pipe.fit(df) # Fitting encoder

# Transforming data
df = e_pipe.transform(df)

# Separating features and target
X, y = df.drop("property_type", axis=1), df["property_type"]

# Loading pretrained model 
with open("../pickle/model.pkl", 'rb') as f:
    model = pickle.load(f)

# Extracting trained model hyper paramteres
params = model.get_params()

# Initializing model with parameters
model_deploy = LGBMClassifier(**params)

model_deploy.fit(X, y)

# During training - save everything needed
model_artifacts = {
    'model': model_deploy,
    'feature_names': X.columns.tolist(),
    'categorical_features': e_pipe.ohe_.get_feature_names_out().tolist(),
    'numeric_features': e_pipe.scaler_.feature_names_in_.tolist(),  # if applicable
    'dtypes': X.dtypes.to_dict()
}

with open('../pickle/model_artifacts.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Saving encoders and scaler
with open("../pickle/ohe.pkl", "wb") as f:
    pickle.dump(e_pipe.ohe_, f)

with open("../pickle/label_encoder.pkl", "wb") as f:
    pickle.dump(e_pipe.le_, f)

with open("../pickle/scaler.pkl", "wb") as f:
    pickle.dump(e_pipe.scaler_, f)
