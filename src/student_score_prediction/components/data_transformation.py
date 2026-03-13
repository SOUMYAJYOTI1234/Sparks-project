import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger
from student_score_prediction.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation paths."""

    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Handles feature engineering and preprocessing pipeline creation."""

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a preprocessing pipeline with StandardScaler.

        Returns:
            sklearn Pipeline object
        """
        try:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                ]
            )
            logger.info("Preprocessing pipeline created with StandardScaler")
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply preprocessing transformations to training and test data.

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV

        Returns:
            Tuple of (train_array, test_array, preprocessor_path)
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and test data read successfully")
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_column = "Scores"
            feature_columns = ["Hours"]

            # Separate features and target
            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column]

            logger.info("Separated features and target variable")

            # Get preprocessing pipeline and fit-transform
            preprocessing_obj = self.get_data_transformer_object()

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logger.info("Applied preprocessing transformations")

            # Combine features and target into arrays
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save the preprocessing object
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logger.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
