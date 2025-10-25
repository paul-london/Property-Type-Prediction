import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

class EncodingPipeline(BaseEstimator, TransformerMixin):
    """
    - OneHotEncodes all categorical dtype columns except `property_type`.
    - If `property_type` exists, label-encodes it and stores class mapping.
    - Ignores missing columns at transform-time (adds NaNs for fitted OHE columns that are absent).
    """
    def __init__(self, property_col="property_type", verbose=True, ohe_dtype="uint8"):
        self.property_col = property_col
        self.verbose = verbose
        self.ohe_dtype = ohe_dtype
        self.cat_cols_ = None            # categorical columns (excluding property_col)
        self.ohe_ = None
        self.scaler_ = None                 # OneHotEncoder
        self.le_ = None                  # LabelEncoder for property_col (optional)
        self.prop_classes_ = None        # classes_ for property_col
        self.prop_class_mapping_ = None  # {class_name: encoded_int}

    # ---------- helpers ----------
    def _is_categorical(self, s: pd.Series) -> bool:
        return s.dtype.name in ("object", "string", "category")

    def _norm_labels(self, s: pd.Series) -> pd.Series:
        # normalize text labels for stable encoding (lower/strip); leave NaN as-is
        return s.astype("string").str.strip().str.lower()

    # ---------- sklearn API ----------
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Detect categorical columns (object/string/category) and exclude property_col
        cat_candidates = [c for c in X.columns if self._is_categorical(X[c])]
        self.cat_cols_ = [c for c in cat_candidates if c != self.property_col]
        self.numeric_cols_ = X.select_dtypes(include=["Int64", "float64"]).columns.to_list()

        # Fit Standard Scaler for numeric columns if any present
        if len(self.numeric_cols_) > 0:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.numeric_cols_])
        else:
            self.scaler_ = None

        # Fit OHE if any categorical columns to encode
        if len(self.cat_cols_) > 0:
            self.ohe_ = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.uint8 if self.ohe_dtype == "uint8" else None
            )
            self.ohe_.fit(X[self.cat_cols_])
        else:
            self.ohe_ = None

        # Fit LabelEncoder on property_col if present
        if self.property_col in X.columns:
            # LabelEncoder needs 1D array; normalize strings to avoid case/space issues
            self.le_ = LabelEncoder()
            labels = self._norm_labels(X[self.property_col])
            self.le_.fit(labels.fillna("nan"))  # treat NaN as string "nan" for stability
            self.prop_classes_ = self.le_.classes_.tolist()
            # Create human-readable mapping (original string class -> int)
            self.prop_class_mapping_ = {
                cls: int(self.le_.transform([cls])[0]) for cls in self.prop_classes_
            }
        else:
            self.le_ = None
            self.prop_classes_ = None
            self.prop_class_mapping_ = None

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        if self.scaler_ is not None:
            self.numeric_cols_ = self.scaler_.get_feature_names_out().tolist()
            missing = [c for c in self.numeric_cols_ if c not in X.columns]
            for m in missing:
                X[m] = pd.NA
            # X_num = X[self.numeric_cols_] 
            X[self.numeric_cols_] = self.scaler_.transform(X[self.numeric_cols_]) 
            scaled_cols = self.scaler_.get_feature_names_out(self.numeric_cols_).tolist()
            # X = pd.concat([X.drop(scaled_cols, axis=1),pd.DataFrame(X_scaled, columns=scaled_cols, index=X.index)], axis=1)

        # Ensure all fitted OHE columns exist; add missing columns as NA so encoder input shape matches
        if self.ohe_ is not None and self.cat_cols_:
            missing = [c for c in self.cat_cols_ if c not in X.columns]
            for m in missing:
                X[m] = pd.NA
            # Order columns exactly as fitted
            X_cat = X[self.cat_cols_]
            # Transform -> numpy
            ohe_arr = self.ohe_.transform(X_cat)
            ohe_cols = self.ohe_.get_feature_names_out(self.cat_cols_).tolist()
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)

            # Drop original categorical cols (except property_col) and concat OHE
            X.drop(columns=[c for c in self.cat_cols_ if c in X.columns], inplace=True, errors="ignore")
            X = pd.concat([X, ohe_df], axis=1)

        # Label-encode property_col if present and encoder fitted
        if (self.le_ is not None) and (self.property_col in X.columns):
            ser = self._norm_labels(X[self.property_col]).fillna("nan")
            X[self.property_col] = self.le_.transform(ser).astype("int64")

        # Minimal sanity prints
        if self.verbose:
            print(f"[EncodingPipeline] Output shape: {X.shape}")
            if self.ohe_ is not None:
                print(f"[EncodingPipeline] One-hot columns added: {len(self.ohe_.get_feature_names_out(self.cat_cols_))}")
                sample_cols = self.ohe_.get_feature_names_out(self.cat_cols_)[:12]
                print(f"[EncodingPipeline] Sample OHE columns: {list(sample_cols)}")
            if self.prop_class_mapping_ is not None:
                print(f"[EncodingPipeline] '{self.property_col}' class mapping: {self.prop_class_mapping_}")
            if self.scaler_ is not None:
                print(f"Columns scaled: {len(scaled_cols)}")
                print(f"Columns scaled: {scaled_cols}")

        return X

    # Convenience accessor
    def get_class_mapping(self):
        return dict(self.prop_class_mapping_) if self.prop_class_mapping_ is not None else None


# === USAGE ===
# Fit on your TRAIN dataset to lock columns/classes, then transform everywhere.

# pipe = EncodingPipeline(property_col="property_type", verbose=True)

# Dataset_New_encoded = pipe.fit_transform(Dataset_New)     # TRAIN/FIT
