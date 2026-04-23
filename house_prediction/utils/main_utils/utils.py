import os, sys
import numpy as np
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

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HousePredictionException(e, sys) from e

def save_object(file_path: str, obj: object):
    """
    Save a python object to file
    file_path: str location of file to save
    obj: object to save
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to file: {file_path}")
    except Exception as e:
        raise HousePredictionException(e, sys) from e                