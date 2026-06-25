from dataclasses import dataclass
from typing import Tuple
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from ..exception import CustomException
from ..logger import logging
from ..helper import create_directory


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("notebooks", "datasets", "kidney_disease_cleaned.csv")
    artifacts_dir: str = os.path.join("artifacts")
    train_file: str = os.path.join("artifacts", "train.csv")
    test_file: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def run(self) -> Tuple[str, str]:
        """Read raw CSV, split into train/test and save to artifacts.

        Returns paths to train and test CSV files.
        """
        try:
            logging.info(f"Reading raw data from: {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)

            # ensure artifacts dir exists
            create_directory(self.config.artifacts_dir)

            # simple split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=True,
            )

            train_df.to_csv(self.config.train_file, index=False)
            test_df.to_csv(self.config.test_file, index=False)

            logging.info(f"Saved train to {self.config.train_file} ({train_df.shape})")
            logging.info(f"Saved test to {self.config.test_file} ({test_df.shape})")

            return self.config.train_file, self.config.test_file
        except Exception as e:
            raise CustomException(e, sys)
