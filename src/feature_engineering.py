"""
Feature engineering: Physics-based features + Robust Preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.config import logger, LEAKAGE_COLS, CAT_FEATURES, NUM_FEATURES, RANDOM_STATE

def create_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers physics-based features from raw sensor data.
    """
    logger.info("   -> ⚙️ Generating Physics-Based Features...")
    df_eng = df.copy()
    
    # 1. Delta Temperature
    df_eng['Temp_Diff'] = df_eng['Process temperature [K]'] - df_eng['Air temperature [K]']

    # 2. Power Input (Torque * Speed)
    df_eng['Power'] = df_eng['Torque [Nm]'] * df_eng['Rotational speed [rpm]']

    # 3. Force Interaction (Torque / Speed) with safe division
    df_eng['Force_Ratio'] = df_eng['Torque [Nm]'] / (df_eng['Rotational speed [rpm]'] + 1e-5)
    
    # Sanitize infinities
    df_eng.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_eng

def get_preprocessor():
    """Returns the Scikit-Learn ColumnTransformer."""
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[['L', 'M', 'H']])) 
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_FEATURES),
        ('cat', cat_pipeline, CAT_FEATURES)
    ])
    
    return preprocessor

def build_features_and_split(df: pd.DataFrame, target_col: str):
    logger.info("[3/5] 🛠️ Engineering Features & Splitting...")
    
    # 1. Feature Engineering
    df_enriched = create_physics_features(df)
    
    # 2. Drop Leakage
    df_model = df_enriched.drop(columns=LEAKAGE_COLS, errors='ignore')
    
    # 3. Define X and y
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]
    
    # 4. Stratified Split (Mandatory for Imbalanced Data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"   ✓ Split Complete. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test