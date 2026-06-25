import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from ..exception import CustomException
from ..logger import logging
from ..components.data_ingestion import DataIngestion, DataIngestionConfig
from ..components.data_transformation import DataTransformation, DataTransformationConfig
from ..components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class TrainPipelineConfig:
    """Configuration for the entire training pipeline."""
    data_ingestion_config: DataIngestionConfig = None
    data_transformation_config: DataTransformationConfig = None
    model_trainer_config: ModelTrainerConfig = None
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.data_ingestion_config is None:
            self.data_ingestion_config = DataIngestionConfig()
        if self.data_transformation_config is None:
            self.data_transformation_config = DataTransformationConfig()
        if self.model_trainer_config is None:
            self.model_trainer_config = ModelTrainerConfig()


class TrainPipeline:
    """
    Complete training pipeline orchestrating data ingestion, transformation, and model training.
    
    This pipeline coordinates all stages of the machine learning workflow:
    1. Data Ingestion: Load raw data and split into train/test sets
    2. Data Transformation: Preprocess and scale features
    3. Model Training: Train multiple models and select the best one
    
    All artifacts are saved to the artifacts folder for later use in predictions.
    """
    
    def __init__(self, config: TrainPipelineConfig = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Optional configuration object for the pipeline.
                   If None, uses default configurations.
        """
        self.config = config if config is not None else TrainPipelineConfig()
        self.pipeline_results = {}
        
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary containing pipeline results including:
            - train_file: Path to training data
            - test_file: Path to test data
            - best_model: The best trained model
            - best_accuracy: Accuracy of the best model
            - pipeline_status: Status of pipeline execution
            
        Raises:
            CustomException: If any stage of the pipeline fails
        """
        try:
            logging.info("="*50)
            logging.info("Starting Training Pipeline")
            logging.info("="*50)
            
            # Stage 1: Data Ingestion
            logging.info("\n[STAGE 1] Data Ingestion")
            train_file, test_file = self._run_data_ingestion()
            
            # Stage 2: Data Transformation
            logging.info("\n[STAGE 2] Data Transformation")
            X_train, X_test, y_train, y_test = self._run_data_transformation(
                train_file, test_file
            )
            
            # Stage 3: Model Training
            logging.info("\n[STAGE 3] Model Training")
            best_model, best_accuracy = self._run_model_training(
                X_train, y_train, X_test, y_test
            )
            
            # Prepare results
            self.pipeline_results = {
                "train_file": train_file,
                "test_file": test_file,
                "best_model": best_model,
                "best_accuracy": best_accuracy,
                "pipeline_status": "SUCCESS"
            }
            
            logging.info("\n" + "="*50)
            logging.info("Training Pipeline Completed Successfully!")
            logging.info(f"Best Model Accuracy: {best_accuracy:.4f}")
            logging.info("="*50)
            
            return self.pipeline_results
            
        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            self.pipeline_results["pipeline_status"] = "FAILED"
            raise CustomException(e, sys)
    
    def _run_data_ingestion(self) -> Tuple[str, str]:
        """
        Execute data ingestion stage.
        
        Loads raw data, performs train/test split, and saves to artifacts folder.
        
        Returns:
            Tuple of (train_file_path, test_file_path)
            
        Raises:
            CustomException: If data ingestion fails
        """
        try:
            logging.info("Initializing Data Ingestion Component")
            data_ingestion = DataIngestion(self.config.data_ingestion_config)
            
            logging.info("Running data ingestion...")
            train_file, test_file = data_ingestion.run()
            
            logging.info(f"✓ Data ingestion completed")
            logging.info(f"  - Train data: {train_file}")
            logging.info(f"  - Test data: {test_file}")
            
            return train_file, test_file
            
        except Exception as e:
            logging.error(f"Data ingestion failed: {str(e)}")
            raise CustomException(e, sys)
    
    def _run_data_transformation(
        self, 
        train_file: str, 
        test_file: str
    ) -> Tuple:
        """
        Execute data transformation stage.
        
        Preprocesses and scales features for both training and test sets.
        Saves preprocessor object to artifacts folder.
        
        Args:
            train_file: Path to training data CSV
            test_file: Path to test data CSV
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            CustomException: If data transformation fails
        """
        try:
            logging.info("Initializing Data Transformation Component")
            data_transformation = DataTransformation(
                self.config.data_transformation_config
            )
            
            logging.info("Running data transformation...")
            X_train, X_test, y_train, y_test = data_transformation.fit_transform(
                train_file, 
                test_file
            )
            
            logging.info(f"✓ Data transformation completed")
            logging.info(f"  - X_train shape: {X_train.shape}")
            logging.info(f"  - X_test shape: {X_test.shape}")
            logging.info(f"  - y_train shape: {y_train.shape}")
            logging.info(f"  - y_test shape: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Data transformation failed: {str(e)}")
            raise CustomException(e, sys)
    
    def _run_model_training(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ) -> Tuple:
        """
        Execute model training stage.
        
        Trains multiple models, evaluates them, and saves the best one to artifacts.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (best_model, best_accuracy)
            
        Raises:
            CustomException: If model training fails
        """
        try:
            logging.info("Initializing Model Trainer Component")
            model_trainer = ModelTrainer(self.config.model_trainer_config)
            
            logging.info("Running model training...")
            best_model, best_accuracy = model_trainer.initiate_model_training(
                X_train,
                y_train,
                X_test,
                y_test
            )
            
            logging.info(f"✓ Model training completed")
            logging.info(f"  - Best model accuracy: {best_accuracy:.4f}")
            logging.info(f"  - Best model type: {type(best_model).__name__}")
            
            return best_model, best_accuracy
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise CustomException(e, sys)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the results from the last pipeline run.
        
        Returns:
            Dictionary with pipeline results, or empty dict if pipeline hasn't run
        """
        return self.pipeline_results
