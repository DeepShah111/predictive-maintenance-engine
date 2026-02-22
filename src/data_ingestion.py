"""
Data ingestion and cleaning module.
"""
import pandas as pd
import gdown
import os
from src.config import logger, DOWNLOAD_URL, FILEPATH

def load_data(filepath: str = FILEPATH, url: str = DOWNLOAD_URL) -> pd.DataFrame:
    logger.info("[1/5]  Loading Data Source")
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if os.path.exists(filepath):
            logger.info("Local file found. Loading directly.")
        else:
            logger.info("Local file not found. Downloading from Drive.")
            gdown.download(url, filepath, quiet=False)

        raw_data = pd.read_csv(filepath)
        logger.info(f"Data Loaded Successfully. Shape: {raw_data.shape}")
        return raw_data
        
    except Exception as e:
        logger.error(f"Data Load Failed: {e}")
        raise RuntimeError(f"Data Load Failed: {e}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[2/5]  Cleaning Data")
    cleaned_df = df.copy()
    
    # Drop duplicates
    initial_len = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    
    if len(cleaned_df) < initial_len:
        logger.info(f"Dropped {initial_len - len(cleaned_df)} duplicate rows.")
        
    # Reset index is crucial before Cross Validation splitting
    cleaned_df.reset_index(drop=True, inplace=True)
        
    return cleaned_df