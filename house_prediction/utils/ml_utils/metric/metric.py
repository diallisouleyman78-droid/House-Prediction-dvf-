import sys
from house_prediction.exception.exception import HousePredictionException
from house_prediction.entity.artifact_entity import RegressionMetricArtifact
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_regression_metrics(y_true, y_pred, n_features=19) -> RegressionMetricArtifact:
    """
    Calculate regression metrics
    y_true: true values
    y_pred: predicted values
    n_features: number of features used in the model (default 19 for this project)
    """
    try:
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        p = n_features
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return RegressionMetricArtifact(r2_score=r2, adj_r2_score=adj_r2, rmse=rmse, mae=mae)
    except Exception as e:
        raise HousePredictionException(e, sys) from e
