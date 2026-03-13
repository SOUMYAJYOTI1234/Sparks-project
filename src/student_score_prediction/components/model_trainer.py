import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from student_score_prediction.exception import CustomException
from student_score_prediction.logger import logger
from student_score_prediction.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer paths."""

    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Trains and compares multiple regression models, selects the best one."""

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models, evaluate them, and save the best one.

        Args:
            train_array: Numpy array with training features + target (last column)
            test_array: Numpy array with test features + target (last column)

        Returns:
            Tuple of (best_model_name, r2_score)
        """
        try:
            logger.info("Splitting train and test arrays into features and target")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models to compare
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1),
                "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
                "SVR": SVR(kernel="rbf"),
            }

            logger.info(f"Evaluating {len(models)} models")

            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logger.info(
                f"Best model: {best_model_name} with R² Score: {best_model_score:.4f}"
            )

            # Print all model scores for comparison
            logger.info("=" * 60)
            logger.info("MODEL COMPARISON REPORT")
            logger.info("=" * 60)
            for name, score in sorted(
                model_report.items(), key=lambda x: x[1], reverse=True
            ):
                marker = " ★ BEST" if name == best_model_name else ""
                logger.info(f"  {name:25s} → R²: {score:.4f}{marker}")
            logger.info("=" * 60)

            if best_model_score < 0.6:
                raise CustomException(
                    "No model achieved acceptable performance (R² < 0.6)", sys
                )

            # Re-train the best model on training data and save
            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logger.info(
                f"Best model saved at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
