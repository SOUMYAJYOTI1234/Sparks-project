import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""

    raw_data_path: str = os.path.join("artifacts", "data", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")


class DataIngestion:
    """Handles data downloading and train/test splitting."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Download dataset from the source URL, save it locally,
        and split into training and test sets.

        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logger.info("Data ingestion started")

        try:
            # Read dataset from the original source URL
            url = (
                "https://raw.githubusercontent.com/AdiPersonalWorks/Random/"
                "master/student_scores%20-%20student_scores.csv"
            )
            df = pd.read_csv(url)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Save raw data
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(
                f"Raw data saved at: {self.ingestion_config.raw_data_path}"
            )

            # Train-test split
            logger.info("Performing train-test split (80/20)")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logger.info(
                f"Train data saved at: {self.ingestion_config.train_data_path}"
            )
            logger.info(
                f"Test data saved at: {self.ingestion_config.test_data_path}"
            )
            logger.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
