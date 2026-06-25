import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
import json
import sys

from .exception import CustomException
from .logger import logging


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary containing YAML data
        
    Raises:
        CustomException: If file doesn't exist or parsing fails
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.info(f"Successfully read YAML file: {path}")
            return config if config is not None else {}
            
    except Exception as e:
        logging.error(f"Error reading YAML file {path}: {str(e)}")
        raise CustomException(e, sys)


def write_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Write data to YAML configuration file.
    
    Args:
        data: Dictionary to write
        path: Path to save YAML file
        
    Raises:
        CustomException: If writing fails
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
            logging.info(f"Successfully wrote YAML file: {path}")
            
    except Exception as e:
        logging.error(f"Error writing YAML file {path}: {str(e)}")
        raise CustomException(e, sys)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 0
) -> Tuple[bool, str]:
    """
    Validate DataFrame for required columns and minimum rows.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            return False, "Invalid DataFrame object"
        
        if len(df) == 0:
            return False, "DataFrame is empty"
        
        if len(df) < min_rows:
            return False, f"DataFrame has {len(df)} rows, minimum required: {min_rows}"
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
        
        return True, "DataFrame validation passed"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistical summary of DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_summary': df.describe().to_dict()
        }
        return stats
        
    except Exception as e:
        logging.error(f"Error calculating statistics: {str(e)}")
        raise CustomException(e, sys)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    numeric_only: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'forward_fill', 'backward_fill', 'drop'
        numeric_only: Apply only to numeric columns
        
    Returns:
        DataFrame with missing values handled
    """
    try:
        df_copy = df.copy()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna()
        elif strategy == 'mean':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == 'forward_fill':
            df_copy = df_copy.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_copy = df_copy.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logging.info(f"Missing values handled using {strategy} strategy")
        return df_copy
        
    except Exception as e:
        logging.error(f"Error handling missing values: {str(e)}")
        raise CustomException(e, sys)


def get_class_distribution(y: Union[pd.Series, np.ndarray]) -> Dict[str, int]:
    """
    Get class distribution in target variable.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary with class counts
    """
    try:
        if isinstance(y, pd.Series):
            counts = y.value_counts().to_dict()
        else:
            unique, indices = np.unique(y, return_counts=True)
            counts = dict(zip(unique, indices.tolist()))
        
        logging.info(f"Class distribution: {counts}")
        return counts
        
    except Exception as e:
        logging.error(f"Error calculating class distribution: {str(e)}")
        raise CustomException(e, sys)


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers. If None, all numeric columns
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for IQR method (default 1.5) or z-score (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    try:
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[columns].fillna(df_copy[columns].mean())))
            df_copy = df_copy[(z_scores < threshold).all(axis=1)]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logging.info(f"Removed outliers using {method} method")
        return df_copy
        
    except Exception as e:
        logging.error(f"Error removing outliers: {str(e)}")
        raise CustomException(e, sys)


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'label'
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        columns: Columns to encode. If None, all non-numeric columns
        method: 'label' or 'onehot'
        
    Returns:
        Tuple of (encoded_df, encoding_mapping)
    """
    try:
        from sklearn.preprocessing import LabelEncoder
        
        df_copy = df.copy()
        encoding_mapping = {}
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoding_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        elif method == 'onehot':
            df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
            encoding_mapping = {'one_hot_encoded': True}
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logging.info(f"Categorical variables encoded using {method} method")
        return df_copy, encoding_mapping
        
    except Exception as e:
        logging.error(f"Error encoding categorical variables: {str(e)}")
        raise CustomException(e, sys)


def print_summary(
    train_shape: Tuple,
    test_shape: Tuple,
    target_dist: Dict[str, int],
    features: int
) -> None:
    """
    Print dataset summary information.
    
    Args:
        train_shape: Shape of training set
        test_shape: Shape of test set
        target_dist: Distribution of target classes
        features: Number of features
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training Set Shape: {train_shape}")
    print(f"Test Set Shape: {test_shape}")
    print(f"Number of Features: {features}")
    print(f"\nTarget Distribution:")
    for class_label, count in target_dist.items():
        print(f"  Class {class_label}: {count} samples")
    print("="*60 + "\n")
