import os
import sys
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from ..exception import CustomException
from ..logger import logging
from ..helper import save_object, create_directory


@dataclass
class ModelTrainerConfig:
    artifacts_dir: str = os.path.join("artifacts")
    model_file: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_training(self, X_train, y_train, X_test, y_test) -> Tuple[object, float]:
        try:
            models = [
                ("LogisticRegression", LogisticRegression(max_iter=500)),
                ("DecisionTree", DecisionTreeClassifier(random_state=42)),
                ("RandomForest", RandomForestClassifier(random_state=42)),
                ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
                ("AdaBoost", AdaBoostClassifier(random_state=42)),
                ("SVC", SVC(random_state=42)),
                ("KNN", KNeighborsClassifier()),
                ("GaussianNB", GaussianNB()),
                ("BernoulliNB", BernoulliNB()),
                ("MultinomialNB", MultinomialNB()),
                ("LDA", LinearDiscriminantAnalysis()),
                ("QDA", QuadraticDiscriminantAnalysis()),
                ("MLP", MLPClassifier(max_iter=500, random_state=42)),
            ]

            best_model = None
            best_score = -1.0

            for name, model in models:
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    logging.info(f"Model {name} accuracy: {acc:.4f}")
                    if acc > best_score:
                        best_score = acc
                        best_model = model
                except Exception as e:
                    logging.warning(f"Training {name} failed: {e}")

            if best_model is None:
                raise Exception("No model was successfully trained")

            # persist best model
            create_directory(self.config.artifacts_dir)
            save_object(self.config.model_file, best_model)
            logging.info(f"Saved best model to {self.config.model_file} with accuracy {best_score:.4f}")

            return best_model, best_score
        except Exception as e:
            raise CustomException(e, sys)
