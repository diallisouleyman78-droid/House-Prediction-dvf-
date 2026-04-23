"""
Main script to test data ingestion, validation and transformation pipeline
"""
import sys
from house_prediction.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from house_prediction.components.data_ingestion import DataIngestion
from house_prediction.components.data_validation import DataValidation
from house_prediction.components.data_transformation import DataTransformation
from house_prediction.logging.logger import logging
from house_prediction.exception.exception import HousePredictionException
import numpy as np

def test_data_ingestion():
    """
    Test the data ingestion pipeline:
    1. Read data from MongoDB
    2. Export to feature store
    3. Split into train/test
    4. Return artifact with file paths
    """
    try:
        logging.info("Starting data ingestion test...")

        # Step 1: Create training pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        logging.info(f"Training pipeline config created: {training_pipeline_config.pipeline_name}")

        # Step 2: Create data ingestion config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        logging.info(f"Data ingestion config created")
        logging.info(f"Database: {data_ingestion_config.database_name}")
        logging.info(f"Collection: {data_ingestion_config.collection_name}")
        logging.info(f"Train/Test split ratio: {data_ingestion_config.train_test_split_ratio}")

        # Step 3: Initialize data ingestion
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Data ingestion component initialized")

        # Step 4: Run data ingestion
        logging.info("Starting data ingestion from MongoDB...")
        # Use limit=1000 for testing, remove limit=None for production
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion(limit=1000)

        # Step 5: Print results
        logging.info("="*50)
        logging.info("DATA INGESTION COMPLETED SUCCESSFULLY!")
        logging.info("="*50)
        logging.info(f"Train file path: {data_ingestion_artifact.train_file_path}")
        logging.info(f"Test file path: {data_ingestion_artifact.test_file_path}")
        logging.info("="*50)

        print("\n" + "="*50)
        print("DATA INGESTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Train file: {data_ingestion_artifact.train_file_path}")
        print(f"Test file: {data_ingestion_artifact.test_file_path}")
        print("="*50)

        return data_ingestion_artifact

    except Exception as e:
        raise HousePredictionException(e, sys)


def test_data_validation(data_ingestion_artifact):
    """
    Test the data validation pipeline:
    1. Validate column count
    2. Validate numerical columns exist
    3. Check data drift
    4. Return validation artifact
    """
    try:
        logging.info("Starting data validation test...")

        # Step 1: Create training pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        logging.info(f"Training pipeline config created: {training_pipeline_config.pipeline_name}")

        # Step 2: Create data validation config
        data_validation_config = DataValidationConfig(training_pipeline_config)
        logging.info(f"Data validation config created")

        # Step 3: Initialize data validation
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Data validation component initialized")

        # Step 4: Run data validation
        logging.info("Starting data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()

        # Step 5: Print results
        logging.info("="*50)
        logging.info("DATA VALIDATION COMPLETED!")
        logging.info("="*50)
        logging.info(f"Validation status: {data_validation_artifact.validation_status}")
        logging.info(f"Valid train file: {data_validation_artifact.valid_train_file_path}")
        logging.info(f"Valid test file: {data_validation_artifact.valid_test_file_path}")
        logging.info(f"Drift report: {data_validation_artifact.drift_report_file_path}")
        logging.info("="*50)

        print("\n" + "="*50)
        print("DATA VALIDATION COMPLETED!")
        print("="*50)
        print(f"Validation status: {data_validation_artifact.validation_status}")
        print(f"Valid train file: {data_validation_artifact.valid_train_file_path}")
        print(f"Valid test file: {data_validation_artifact.valid_test_file_path}")
        print(f"Drift report: {data_validation_artifact.drift_report_file_path}")
        print("="*50)

        return data_validation_artifact

    except Exception as e:
        raise HousePredictionException(e, sys)


def test_data_transformation(data_validation_artifact):
    """
    Test the data transformation pipeline:
    1. Clean data
    2. Apply feature engineering
    3. Apply target encoding
    4. Scale and impute
    5. Return transformation artifact
    """
    try:
        logging.info("Starting data transformation test...")

        # Step 1: Create training pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        logging.info(f"Training pipeline config created: {training_pipeline_config.pipeline_name}")

        # Step 2: Create data transformation config
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info(f"Data transformation config created")

        # Step 3: Initialize data transformation
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Data transformation component initialized")

        # Step 4: Run data transformation
        logging.info("Starting data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        # Step 5: Print results
        logging.info("="*50)
        logging.info("DATA TRANSFORMATION COMPLETED!")
        logging.info("="*50)
        logging.info(f"Transformed train file: {data_transformation_artifact.transformed_train_file_path}")
        logging.info(f"Transformed test file: {data_transformation_artifact.transformed_test_file_path}")
        logging.info(f"Transformed object: {data_transformation_artifact.transformed_object_file_path}")
        logging.info("="*50)

        print("\n" + "="*50)
        print("DATA TRANSFORMATION COMPLETED!")
        print("="*50)
        print(f"Transformed train file: {data_transformation_artifact.transformed_train_file_path}")
        print(f"Transformed test file: {data_transformation_artifact.transformed_test_file_path}")
        print(f"Transformed object: {data_transformation_artifact.transformed_object_file_path}")
        print("="*50)

        return data_transformation_artifact

    except Exception as e:
        raise HousePredictionException(e, sys)

if __name__ == "__main__":
    try:
        # Step 1: Test data ingestion
        print("="*50)
        print("STEP 1: DATA INGESTION")
        print("="*50)
        data_ingestion_artifact = test_data_ingestion()
        print("\n✓ Data ingestion test passed!")

        # Step 2: Test data validation
        print("\n" + "="*50)
        print("STEP 2: DATA VALIDATION")
        print("="*50)
        data_validation_artifact = test_data_validation(data_ingestion_artifact)
        print("\n✓ Data validation test passed!")

        # Step 3: Test data transformation
        print("\n" + "="*50)
        print("STEP 3: DATA TRANSFORMATION")
        print("="*50)
        data_transformation_artifact = test_data_transformation(data_validation_artifact)
        print("\n✓ Data transformation test passed!")

        print("\n" + "="*50)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
