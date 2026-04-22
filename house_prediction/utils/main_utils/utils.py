import os, sys
import numpy
import pickle
from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging
import yaml

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousePredictionException(e, sys) from e   

def write_yaml_file(file_path: str, data: dict):
    try:
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise HousePredictionException(e, sys) from e          