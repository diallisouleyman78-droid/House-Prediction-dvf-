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
