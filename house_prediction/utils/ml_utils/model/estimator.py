import os, sys
from house_prediction.exception.exception import HousePredictionException
from house_prediction.logging.logger import logging
from house_prediction.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class HouseModel:
    def __init__(self, preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise HousePredictionException(e, sys)
    
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise HousePredictionException(e, sys)
