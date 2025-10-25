import preprocessing as pre
import joblib
import lightgbm as lgbm
import pandas as pd
import csv

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, precision_score, recall_score

# Loading fitted model, encodres, and scaler
with open("model.pkl", 'rb') as f:
    model = joblib.load(f)

with open("ordinal_encoder.pkl", 'rb') as f:
    oe = joblib.load(f)

with open("label_encoder.pkl", 'rb') as f:
    le = joblib.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = joblib.load(f)

# Printing config
print(f"Model type: {type(model)}")
print(f"Model parameters: {model.get_params()}")
print(f"Encoder type: {type(oe)}")
print(f"Encoder parameters: {oe.get_params()}")
print(f"Encoder type: {type(le)}")
print(f"Encoder parameters: {le.classes_}")
print(f"Scaler type: {type(scaler)}")
print(f"Scaler parameters: {scaler.get_params()}")

state = 42
test = pd.read_csv("~/externships/Homeservices/Dataset.csv")

test = pre.preprocessor(test)
test = pre.new_features(test)

sample = test.sample(5, random_state=state)
X = sample.sample(5).drop("type",axis=1)
y = sample.loc[X.index]["type"]

cats = ["prop_cond","city"]

X[cats] = oe.transform(X[cats])
X = scaler.transform(X)
X = pd.DataFrame(X, columns=sample.drop("type",axis=1).columns)
y = le.transform(y)


pred = model.predict(X)
score = balanced_accuracy_score(y, pred)#, average="macro")
# conf = confusion_matrix(y, pred)
# recall = recall_score(y, pred, average="macro")
# print(f"Score: {score}\nConfusion Matrix: {conf}, \nRecall: {recall}")
print("y true: \n", y)
print("Prediction: \n", pred)
print("Accuracy Score: \n", score)
