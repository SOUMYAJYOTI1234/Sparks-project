import sys

from student_score_prediction.components.data_ingestion import DataIngestion
from student_score_prediction.components.data_transformation import DataTransformation
from student_score_prediction.components.model_trainer import ModelTrainer
from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger


class TrainingPipeline:
    """Orchestrates the full training pipeline: Ingestion → Transformation → Training."""

    def __init__(self):
        pass

    def run(self):
        """Execute the complete training pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Step 1: Data Ingestion
            logger.info("Step 1/3: Data Ingestion")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logger.info("Step 2/3: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(train_path, test_path)
            )

            # Step 3: Model Training
            logger.info("Step 3/3: Model Training")
            model_trainer = ModelTrainer()
            best_model_name, r2_score = model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"R² Score: {r2_score:.4f}")
            logger.info("=" * 60)

            print("\n" + "=" * 60)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"  Best Model  : {best_model_name}")
            print(f"  R² Score    : {r2_score:.4f}")
            print(f"  Model saved : artifacts/model.pkl")
            print(f"  Preprocessor: artifacts/preprocessor.pkl")
            print("=" * 60 + "\n")

            return best_model_name, r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
