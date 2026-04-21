"""
Main script to test data ingestion pipeline
"""
import sys
from house_prediction.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from house_prediction.components.data_ingestion import DataIngestion
from house_prediction.logging.logger import logging
from house_prediction.exception.exception import HousePredictionException

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
        # Process all records (this will take time for 650K records)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion(limit=None)
        
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

if __name__ == "__main__":
    try:
        artifact = test_data_ingestion()
        print("\n✓ Data ingestion test passed!")
    except Exception as e:
        print(f"\n✗ Data ingestion test failed: {e}")
        sys.exit(1)
