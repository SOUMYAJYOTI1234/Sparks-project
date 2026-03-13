import os
import sys

import numpy as np
import pandas as pd

from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger
from student_score_prediction.utils import load_object


class PredictPipeline:
    """Loads the trained model and preprocessor to make predictions."""

    def __init__(self):
        pass

    def predict(self, features):
        """
        Make predictions using the saved model and preprocessor.

        Args:
            features: pandas DataFrame with feature columns

        Returns:
            numpy array of predictions
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logger.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logger.info("Applying preprocessing to input features")
            data_scaled = preprocessor.transform(features)

            logger.info("Making prediction")
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Represents a single prediction request.
    Maps form input to a DataFrame for the prediction pipeline.
    """

    def __init__(self, hours: float):
        self.hours = hours

    def get_data_as_dataframe(self):
        """Convert the input data to a pandas DataFrame."""
        try:
            data_dict = {
                "Hours": [self.hours],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
