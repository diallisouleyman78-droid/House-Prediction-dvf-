import os
from dotenv import load_dotenv
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging
import certifi


load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where() #get the path to the certificate bundle

class HousePredictionExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise HousePredictionException(e, sys)

    def csv_to_json(self, file_path):
        try:
            data = pd.read_csv(file_path, low_memory=False)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise HousePredictionException(e, sys)
               

    def insert_data_to_mongo(self, records, database, collection, batch_size=10000):
        try:
            self.records = records
            self.database = database
            self.collection = collection

            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            
            self.database = client[self.database]
            self.collection = self.database[self.collection]
            
            # Insert in batches for better performance
            total_inserted = 0
            for i in range(0, len(self.records), batch_size):
                batch = self.records[i:i + batch_size]
                self.collection.insert_many(batch)
                total_inserted += len(batch)
                print(f"Inserted {total_inserted}/{len(self.records)} records...")

            return total_inserted

        except Exception as e:
            raise HousePredictionException(e, sys)      

if __name__ == "__main__":    
    FILE_PATH = "house_data/filtered_data.csv"
    DATABASE = "house_prediction"
    COLLECTION = "house_data"
    houseobj = HousePredictionExtract()
    records = houseobj.csv_to_json(FILE_PATH)

    no_of_records = houseobj.insert_data_to_mongo(records, DATABASE, COLLECTION)
    print(f"Inserted {no_of_records} records into {DATABASE}.{COLLECTION}")
    
    
