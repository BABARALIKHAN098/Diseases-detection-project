import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from ..exception import CustomException
from ..logger import logging
from ..helper import save_object, create_directory


@dataclass
class DataTransformationConfig:
    artifacts_dir: str = os.path.join("artifacts")
    preprocessor_obj_file: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def fit_transform(self, train_path: str, test_path: str):
        try:
            logging.info("Loading train and test for transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # assume target column is 'classification' as in notebook
            target_col = 'classification'

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # scale numeric features
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            if numeric_cols:
                X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

            # persist preprocessor
            create_directory(self.config.artifacts_dir)
            save_object(self.config.preprocessor_obj_file, scaler)
            logging.info(f"Saved preprocessor to {self.config.preprocessor_obj_file}")

            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        """Transform a dataframe or numpy array using saved preprocessor.
        (Not implemented fully here; use fit_transform for pipeline.)"""
        raise NotImplementedError("Use fit_transform for now")
