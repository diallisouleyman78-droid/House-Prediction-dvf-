from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging

from house_prediction.entity.config_entity import DataIngestionConfig
from house_prediction.entity.artifact_entity import DataIngestionArtifact

import os,sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import certifi
load_dotenv()


MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HousePredictionException(e, sys)  


    def export_collection_as_dataframe(self, limit=None):
        try:
            """Read data from database and export as dataframe"""
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            print(f"Connecting to MongoDB: {database_name}.{collection_name}...")
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            collection = self.mongo_client[database_name][collection_name]
            print(f"Connected! Reading data from collection...")

            # Set limit for testing (None = all records)
            if limit:
                print(f"Fetching {limit} records for testing...")
                cursor = collection.find().limit(limit)
            else:
                print(f"Fetching all documents from MongoDB (this may take time for 650K records)...")
                cursor = collection.find()

            df = pd.DataFrame(list(cursor))
            print(f"Fetched {len(df)} records")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            print(f"Dataframe created with shape: {df.shape}")
            return df
        except Exception as e:
            raise HousePredictionException(e, sys)


    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            print(f"Saving to feature store: {feature_store_file_path}")
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            print(f"Feature store saved successfully")
        except Exception as e:
            raise HousePredictionException(e, sys)   

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            print(f"Splitting data into train/test (ratio: {self.data_ingestion_config.train_test_split_ratio})...")
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed train and test split on the data")
            print(f"Train set: {train_set.shape}, Test set: {test_set.shape}")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            print(f"Saving train file: {self.data_ingestion_config.train_file_path}")
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            print(f"Saving test file: {self.data_ingestion_config.test_file_path}")
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            
            logging.info("saved the train and test data in the ingested directory")
            print(f"Train/test files saved successfully")

        except Exception as e:
            raise HousePredictionException(e, sys)                



    def initiate_data_ingestion(self, limit=None):
        try:
            print("="*50)
            print("INITIATING DATA INGESTION")
            print("="*50)
            dataframe = self.export_collection_as_dataframe(limit=limit)
            self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )
            print("="*50)
            print("DATA INGESTION COMPLETED")
            print("="*50)
            return data_ingestion_artifact

        except Exception as e:
            raise HousePredictionException(e, sys)       
        