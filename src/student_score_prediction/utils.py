import os
import sys
import dill
import numpy as np
from sklearn.metrics import r2_score

from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger


def save_object(file_path, obj):
    """Serialize and save a Python object to disk using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a serialized Python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models, returning a dictionary of R² scores.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of {model_name: model_instance}

    Returns:
        Dictionary of {model_name: r2_score}
    """
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            logger.info(f"{model_name} R² Score: {score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
