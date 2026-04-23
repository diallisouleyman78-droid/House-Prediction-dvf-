import os
import sys
import pandas as pd
import numpy as np

"""
    defining common constants for the training pipeline
"""

TARGET_COLUMN = "Valeur fonciere"
PIPELINE_NAME: str = "house_prediction"
ARTIFACT_DIR: str = "Artifact"
FILE_NAME: str = "filtered_data.csv"

TRAINING_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

"""
Data ingestion related constants
"""

DATA_INGESTION_COLLECTION_NAME: str = "house_data"
DATA_INGESTION_DATABASE_NAME: str = "house_prediction"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2



SCHEMA_FILE_PATH: str = os.path.join("house_prediction", "data_schema", "schema.yaml")
MODEL_FILE_NAME = "model.pkl"

"""
Data validation related constants
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
TRANSFORM_OBJECT_FILE_NAME: str = "transformer.pkl"

"""
Data transformation related constants
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

