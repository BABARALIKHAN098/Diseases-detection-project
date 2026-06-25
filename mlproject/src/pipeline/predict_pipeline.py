import os
import sys
import pandas as pd
import numpy as np
from typing import Union, List
from dataclasses import dataclass

from ..exception import CustomException
from ..logger import logging
from ..helper import load_object

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


@dataclass
class PredictPipelineConfig:
    """Configuration for prediction pipeline paths."""
    model_file: str = os.path.join(BASE_DIR, "artifacts", "model.pkl")
    preprocessor_file: str = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")


class PredictPipeline:
    """
    Pipeline for making predictions on new data.
    
    This pipeline loads a pre-trained model and preprocessor from the artifacts folder,
    preprocesses input data, and returns predictions.
    """
    
    def __init__(self, config: PredictPipelineConfig = PredictPipelineConfig()):
        """
        Initialize the prediction pipeline.
        
        Args:
            config: Configuration object containing paths to model and preprocessor
            
        Raises:
            CustomException: If model or preprocessor files cannot be loaded
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """
        Load the pre-trained model and preprocessor from artifacts folder.
        
        Raises:
            CustomException: If loading artifacts fails
        """
        try:
            logging.info("Loading model and preprocessor from artifacts")
            
            if not os.path.exists(self.config.model_file):
                raise FileNotFoundError(f"Model file not found: {self.config.model_file}")
            
            if not os.path.exists(self.config.preprocessor_file):
                raise FileNotFoundError(f"Preprocessor file not found: {self.config.preprocessor_file}")
            
            self.model = load_object(self.config.model_file)
            self.preprocessor = load_object(self.config.preprocessor_file)
            
            logging.info("Successfully loaded model and preprocessor")
            
        except Exception as e:
            logging.error(f"Error loading artifacts: {str(e)}")
            raise CustomException(e, sys)
    
    def preprocess_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess input data using the loaded preprocessor.
        
        Args:
            X: Input features as DataFrame or numpy array
            
        Returns:
            Preprocessed features as numpy array
            
        Raises:
            CustomException: If preprocessing fails
        """
        try:
            logging.info("Preprocessing input data")
            
            # Convert to DataFrame if numpy array
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            
            # Get numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logging.warning("No numeric columns found in input data")
                return X.values
            
            # Create a copy to avoid modifying original
            X_processed = X.copy()
            
            # Apply scaling to numeric columns
            X_processed[numeric_cols] = self.preprocessor.transform(X[numeric_cols])
            
            logging.info(f"Data preprocessed successfully. Shape: {X_processed.shape}")
            
            return X_processed.values
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features as DataFrame or numpy array
            
        Returns:
            Predictions as numpy array
            
        Raises:
            CustomException: If prediction fails
        """
        try:
            logging.info(f"Making predictions on data with shape: {X.shape if hasattr(X, 'shape') else len(X)}")
            
            if self.model is None or self.preprocessor is None:
                raise RuntimeError("Model or preprocessor not loaded. Please check artifact files.")
            
            # Preprocess the data
            X_processed = self.preprocess_data(X)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            logging.info(f"Predictions generated successfully. Unique values: {np.unique(predictions)}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities for input data (if model supports it).
        
        Args:
            X: Input features as DataFrame or numpy array
            
        Returns:
            Prediction probabilities as numpy array
            
        Raises:
            CustomException: If prediction fails or model doesn't support probabilities
        """
        try:
            logging.info("Generating prediction probabilities")
            
            if not hasattr(self.model, 'predict_proba'):
                raise AttributeError("Current model does not support predict_proba")
            
            # Preprocess the data
            X_processed = self.preprocess_data(X)
            
            # Get probabilities
            probabilities = self.model.predict_proba(X_processed)
            
            logging.info(f"Probabilities generated successfully. Shape: {probabilities.shape}")
            
            return probabilities
            
        except Exception as e:
            logging.error(f"Error generating probabilities: {str(e)}")
            raise CustomException(e, sys)
