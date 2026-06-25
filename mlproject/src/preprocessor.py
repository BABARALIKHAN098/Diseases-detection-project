
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def remove_duplicates(df):
    return df.drop_duplicates()


def fill_missing_values(df):
    imputer = SimpleImputer(strategy="mean")

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
