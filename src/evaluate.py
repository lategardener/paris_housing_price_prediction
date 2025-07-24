from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation(y_true, y_pred, metric):
    """
    Evaluate regression predictions using the specified metric.

    Parameters:
    - y_true: array-like of true target values
    - y_pred: array-like of predicted values
    - metric: str, one of ['mse', 'mae', 'r2']

    Returns:
    - score: float, evaluation metric result
    """
    if metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    if metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    return None

def run_all_metrics(y_true, y_pred, model_name=None):
    """
    Compute and print multiple regression metrics for given predictions.

    Parameters:
    - y_true: array-like of true target values
    - y_pred: array-like of predicted values
    - model_name: str or None, optional name of the model for display
    """
    regression_mae = evaluation(y_true, y_pred, "mae")
    regression_mse = evaluation(y_true, y_pred, "mse")
    regression_r2 = evaluation(y_true, y_pred, "r2")

    print("="*(35 + len(model_name) if model_name else 35))
    print(f"📊  Regression model evaluation ({model_name})" if model_name else "📊  Regression model evaluation")
    print("="*(35 + len(model_name) if model_name else 35))
    print(f"🔹 MAE (Mean Absolute Error) : {regression_mae:.2f}")
    print(f"🔹 MSE (Mean Squared Error)  : {regression_mse:.2f}")
    print(f"🔹 R2  (Coefficient of Determination)  : {regression_r2:.2f}")
