# Calude optimization
import pandas as pd
import pickle
from Data_Cleaning_Pipeline import clean_and_engineer_features
from Data_Encoding_Pipeline import EncodingPipeline
from pathlib import Path
from typing import Tuple, Optional


class RunModel:
    """
    Class to run the preprocessing, feature engineering, encoding,
    scaling, and model loading, predicting, and displaying results
    """
    
    CATEGORICAL_FEATURES = ["prop_cond", "city"]
    
    def __init__(self, model_dir: str = "."):
        """
        Initialize the model and load all required artifacts.
        
        Args:
            model_dir: Directory containing model artifacts (default: current directory)
        """
        model_dir = Path(model_dir)
        
        # Load all artifacts at once with error handling
        self.model_artifacts = self._load_artifact(model_dir / "model_artifacts.pkl")
        self.ohe = self._load_artifact(model_dir / "ohe.pkl")
        self.scaler = self._load_artifact(model_dir / "scaler.pkl")
        self.label_encoder = self._load_artifact(model_dir / "label_encoder.pkl") 
        
        # Cache for processed data
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
    
    @staticmethod
    def _load_artifact(filepath: Path):
        """Load a single artifact with error handling."""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Required artifact not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}: {str(e)}")
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data through all transformation steps.
        
        Args:
            data: Raw input dataframe
            
        Returns:
            Tuple of (X, y) - features and target variable
        """
        # Apply preprocessing pipeline
        self.raw = X
        preprocessed = clean_and_engineer_features(X)
        self.preprocessed = preprocessed

        # Initializing encoder and loading encoders and scaler
        encoder = EncodingPipeline()
        encoder.le_ = self.label_encoder
        encoder.ohe_ = self.ohe
        encoder.scaler_ = self.scaler

        # Encode categorical features
        X_clean = encoder.transform(preprocessed)

        # Cache processed data
        self.X_clean = X_clean
        
        return X_clean
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:#, return_dataframe: bool = True) -> pd.DataFrame:
        """
        Generate predictions with probabilities.
        
        Args:
            return_dataframe: If True, returns DataFrame; if False, prints and returns None
            
        Returns:
            DataFrame with prediction probabilities or None
        """
        if X is not None:
            self.X = X
        
        model = self.model_artifacts["model"]
        cat_features = self.model_artifacts["categorical_features"]
        num_features = self.model_artifacts["numeric_features"]
        feature_order = self.model_artifacts["feature_names"]

        # Reordering columns to match model and encoder order
        X = X.reindex(columns=feature_order)

        # Ensuring each feature has correct dtype to match with model training
        for col, dtype in self.model_artifacts["dtypes"].items():
            X[col] = X[col].astype(dtype, errors="ignore")

        self.X = X

        # Get prediction probabilities
        pred = model.predict(self.X)
        
        # Create results dataframe with proper class labels
        pred_labels = self.label_encoder.inverse_transform(pred)

        # Combining raw features with predictions
        self.raw['Predicted Type'] = pred_labels
        
        return self.raw
    
    def predict_new(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to preprocess and predict in one step.
        
        Args:
            data: Raw input dataframe
            
        Returns:
            DataFrame with prediction probabilities
        """
        self.preprocess(data)
        return self.predict()
