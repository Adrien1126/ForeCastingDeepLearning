import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) for predictions.
    Args:
        y_true (numpy array): True values.
        y_pred (numpy array): Predicted values.
    Returns:
        float: RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_nse(y_true, y_pred):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for predictions.
    Args:
        y_true (numpy array): True values.
        y_pred (numpy array): Predicted values.
    Returns:
        float: NSE value.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)
