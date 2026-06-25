from dataclasses import dataclass
from typing import Optional
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from ..exception import CustomException
from ..logger import logging


@dataclass
class DataCleaningConfig:
	drop_duplicates: bool = True
	missing_threshold: float = 0.5  # drop columns with fraction of missing values > threshold
	numeric_impute_strategy: str = "mean"  # 'mean' | 'median' | 'most_frequent'
	categorical_impute_strategy: str = "most_frequent"
	encode_categorical: str = "onehot"  # 'onehot' | 'label' | 'none'
	outlier_method: Optional[str] = "iqr"  # 'iqr' or None
	outlier_threshold: float = 1.5  # multiplier for IQR


class DataCleaning:
	"""Utility class encapsulating common dataset cleaning steps.

	Usage:
		cfg = DataCleaningConfig()
		cleaner = DataCleaning(cfg)
		df_clean = cleaner.clean(df)
	"""

	def __init__(self, config: DataCleaningConfig = DataCleaningConfig()):
		self.config = config

	def drop_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			if not self.config.drop_duplicates:
				return df

			before = df.shape[0]
			df = df.drop_duplicates().reset_index(drop=True)
			after = df.shape[0]
			logging.info(f"drop_duplicate_rows: removed {before - after} rows")
			return df
		except Exception as e:
			raise CustomException(e, sys)

	def drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			missing_frac = df.isna().mean()
			cols_to_drop = missing_frac[missing_frac > self.config.missing_threshold].index.tolist()
			if cols_to_drop:
				df = df.drop(columns=cols_to_drop)
				logging.info(f"drop_high_missing_columns: dropped columns {cols_to_drop}")
			return df
		except Exception as e:
			raise CustomException(e, sys)

	def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
			cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

			if numeric_cols:
				imputer_num = SimpleImputer(strategy=self.config.numeric_impute_strategy)
				df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
				logging.info(f"impute_missing: imputed numeric columns {numeric_cols} using {self.config.numeric_impute_strategy}")

			if cat_cols:
				imputer_cat = SimpleImputer(strategy=self.config.categorical_impute_strategy)
				df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
				logging.info(f"impute_missing: imputed categorical columns {cat_cols} using {self.config.categorical_impute_strategy}")

			return df
		except Exception as e:
			raise CustomException(e, sys)

	def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			if self.config.encode_categorical == 'none':
				return df

			cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
			if not cat_cols:
				return df

			if self.config.encode_categorical == 'onehot':
				df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
				logging.info(f"encode_categoricals: one-hot encoded {cat_cols}")
				return df

			if self.config.encode_categorical == 'label':
				from sklearn.preprocessing import LabelEncoder
				le = LabelEncoder()
				for c in cat_cols:
					df[c] = le.fit_transform(df[c].astype(str))
				logging.info(f"encode_categoricals: label encoded {cat_cols}")
				return df

			return df
		except Exception as e:
			raise CustomException(e, sys)

	def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			if not self.config.outlier_method:
				return df

			if self.config.outlier_method == 'iqr':
				num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
				if not num_cols:
					return df

				Q1 = df[num_cols].quantile(0.25)
				Q3 = df[num_cols].quantile(0.75)
				IQR = Q3 - Q1

				lower = Q1 - self.config.outlier_threshold * IQR
				upper = Q3 + self.config.outlier_threshold * IQR

				before = df.shape[0]
				mask = pd.Series(True, index=df.index)
				for col in num_cols:
					mask &= df[col].between(lower[col], upper[col], inclusive='both')

				df = df[mask].reset_index(drop=True)
				after = df.shape[0]
				logging.info(f"handle_outliers: removed {before - after} rows based on IQR across {len(num_cols)} numeric columns")

			return df
		except Exception as e:
			raise CustomException(e, sys)

	def clean(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Run full cleaning pipeline on dataframe and return cleaned copy."""
		try:
			df_clean = df.copy()
			df_clean = self.drop_duplicate_rows(df_clean)
			df_clean = self.drop_high_missing_columns(df_clean)
			df_clean = self.impute_missing(df_clean)
			df_clean = self.handle_outliers(df_clean)
			df_clean = self.encode_categoricals(df_clean)

			logging.info(f"clean: completed cleaning pipeline; shape is now {df_clean.shape}")
			return df_clean
		except Exception as e:
			raise CustomException(e, sys)

