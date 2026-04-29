import os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging
from house_prediction.entity.config_entity import ModelTrainerConfig
from house_prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from house_prediction.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from house_prediction.utils.ml_utils.model.estimator import HouseModel
from house_prediction.utils.ml_utils.metric.metric import calculate_regression_metrics

import dagshub
dagshub.init(repo_owner='diallisouleyman78', repo_name='House-Prediction-dvf-', mlflow=True)

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.data
    MLFLOW_AVAILABLE = True
    logging.info("MLflow imported successfully")
except ImportError as e:
    MLFLOW_AVAILABLE = False
    logging.warning(f"MLflow not installed. MLflow tracking will be disabled. Error: {e}")


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HousePredictionException(e, sys) from e

    def track_mlflow(self, best_model, regression_metric, run_name="train"):
        """
        Track model training with MLflow for regression
        Logs model type, metrics, and model artifact
        """
        try:
            if not MLFLOW_AVAILABLE:
                logging.warning("MLflow not available. Skipping MLflow tracking.")
                return

            logging.info(f"Starting MLflow tracking for {run_name}...")

            # Set experiment
            mlflow.set_experiment(experiment_name="regression")

            with mlflow.start_run(run_name=f"model_training_{run_name}"):
                mlflow.log_param("model_type", best_model.__class__.__name__)

                # Regression metrics
                mlflow.log_metric("r2_score", regression_metric.r2_score)
                mlflow.log_metric("adj_r2_score", regression_metric.adj_r2_score)
                mlflow.log_metric("rmse", regression_metric.rmse)
                mlflow.log_metric("mae", regression_metric.mae)

                mlflow.sklearn.log_model(best_model, "model")

                logging.info(f"MLflow tracking completed for {run_name}")

        except Exception as e:
            raise HousePredictionException(e, sys) from e

    def train_model(self, x_train, y_train, x_test, y_test):
        """
        Train Random Forest model with parameters from test.ipynb
        """
        try:
            logging.info("Training Random Forest model...")

            # Random Forest with same parameters as test.ipynb
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            # Train the model
            model.fit(x_train, y_train)
            logging.info("Model training completed")

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculate metrics
            train_metric = calculate_regression_metrics(y_train, y_train_pred)
            test_metric = calculate_regression_metrics(y_test, y_test_pred)

            logging.info(f"Train R² Score: {train_metric.r2_score:.4f}")
            logging.info(f"Test R² Score: {test_metric.r2_score:.4f}")
            logging.info(f"Model Accuracy: {test_metric.r2_score * 100:.2f}%")

            # Track with MLflow
            self.track_mlflow(model, train_metric, run_name="train")
            self.track_mlflow(model, test_metric, run_name="test")

            # Load preprocessor
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Create combined model object
            house_model = HouseModel(preprocessor=preprocessor, model=model)

            # Save model
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, house_model)

            # Save to final_model directory for inference
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", house_model)

            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

            return model_trainer_artifact

        except Exception as e:
            raise HousePredictionException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate model training
        """
        try:
            logging.info("Initiating model training")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load transformed data
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features and target
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

            # Train model
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)

            logging.info("Model training completed successfully")
            return model_trainer_artifact

        except Exception as e:
            raise HousePredictionException(e, sys) from e
