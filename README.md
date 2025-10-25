# 🏠 Task 3: Property Type Prediction

## 📘 Objective
Create an interactive dashboard that takes MLS (Multiple Listing Service) `.csv` input and predicts **property type** — distinguishing between:
- **Detached**
- **Attached**
- **Condo**

This project demonstrates end-to-end data processing, model development, and real-time prediction using Streamlit and gradient boosting models.

---

## 🚀 Features
- **File Upload:** Accepts MLS `.csv` data directly from users.  
- **Automated Data Cleaning & Feature Engineering:** Applies consistent transformations to numeric and categorical features.  
- **Encoding Pipeline:** Uses stored OneHotEncoder, LabelEncoder, and Scaler artifacts for reproducibility.  
- **Model Prediction:** Outputs property type probabilities across three classes.  
- **Interactive Dashboard:** Built in Streamlit with real-time inference and visualization.

---

## 🧠 Machine Learning Workflow
1. **Data Preprocessing:**  
   - Handling missing values, outlier filtering, and feature scaling.  
   - Categorical encoding via `OneHotEncoder` and `LabelEncoder`.

2. **Modeling:**  
   - Primary models: LightGBM, CatBoost, and XGBoost, RandomForest.
   - Comparison of stratification and encoding methodologies prior to modeling.    
   - Hyperparameter tuning with Optuna for performance optimization.
   - Evaluation metrics: Accuracy, Precision, Recall, and F1-Score.

3. **Deployment:**  
   - Streamlit app for accessible, browser-based prediction interface.  
   - Artifacts (`model.pkl`, `ohe.pkl`, `le.pkl`, `scaler.pkl`) stored for versioned reuse.

---

## 🧰 Tools & Technologies
**Languages & Frameworks:**  
- Python 3.11  
- Streamlit  
- scikit-learn  
- LightGBM / XGBoost / CatBoost  

**Libraries:**  
- pandas, numpy, matplotlib, seaborn, plotly  
- imbalanced-learn, shap, optuna, lazypredict  
- joblib, pathlib, openpyxl  

**Development Environment:**  
- Jupyter Notebook for experimentation  
- VS Code for app development  
- GitHub for version control and collaboration

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/property-type-prediction.git
   cd property-type-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Upload your `.csv` file** and view predicted property types.

---

## 🧩 Example Input
A valid MLS CSV must include the following columns:

        'MLS#', 'Type', 'Prop. Cat.', 'Prop. Cond.', 'Tax', 'Address', 'City', 
        'Zip', 'Area', 'BD', 'Baths', '# Levels', 'Apx Sqft', 'Price SqFt', 
        'Sld Price Sqft', 'Lot Size', 'Pend. Date', 'DOM', 'CDOM', 'List Date', 
        'List Price', 'Sold Date', 'Price', 'Yr. Built', 'HOA Dues', '# Garage', 
        '# Fireplaces', 'Terms'

For complete guidelines, please see the application page.

---

## 👥 Contributors

| Name | Role | Email | GitHub | LinkedIn |
|------|------|----------|--------|-------|
| **Mazin Hassan** | Data Scientist | mazinmhassan@gmail.com | [GitHub](https://github.com/Mazindata) | [LinkedIn](https://www.linkedin.com/in/mazin-hassan/) |
| **Paul London** | Data Scientist | palondon@hotmail.com | [GitHub](https://github.com/paul-london/) |  [LinkedIn](https://www.linkedin.com/in/palondon/) |
| **Sabrina McField** | Data Scientist | sabrinamcfield@gmail.com | [GitHub](https://github.com/SabrinaMcField) | [LinkedIn](https://www.linkedin.com/in/sabrina-mcfield/) |
| **Chris Rivera** | Data Scientist | criveraaprg@gmail.com | [GitHub](https://github.com/Chris-Coded-Rivera) | [LinkedIn](https://linkedin.com/chris-rivera-ds) |

---

## 🔑 Keywords
`Machine Learning` • `Real Estate Analytics` • `Streamlit` • `Property Classification` • `LightGBM` • `Feature Engineering` • `Data Science` • `Python`

---

## 📈 Results Summary
- Achieved **96.5% accuracy** on validation data using tuned LightGBM model.  
- Consistent performance across all property types.  
- Explainability verified using SHAP feature importance plots.

---

## 🙌 Acknowledgments
Special thanks to Berkshire Hathaway HomeServices, mentor [Dr. Ernest Bonat](https://github.com/ebonat), other collaborators, and Elvira Chorna at TripleTen.

---

## 🔒 Data Privacy Notice

This repository does not include any proprietary, private, or real-world client data.
All data are synthetic to protect confidentiality.

---

## 📊 Example Visualizations

Below are sample outputs and visualizations from the Property Type Prediction Notebook:

<img width="1349" height="452" alt="violinplot_bd_bath_processed" src="https://github.com/user-attachments/assets/bef0ffb3-d105-4e11-a63d-db0fbcbd24a7" />

<img width="1400" height="1000" alt="distribution_property_type_by_city" src="https://github.com/user-attachments/assets/3ef05538-3221-4d9c-a4ab-991dc317dcd4" />

