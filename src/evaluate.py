from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation(y_true, y_pred, metric):
    if metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    if metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    return None

def run_all_metrics(y_true, y_pred, model_name=None):
    regression_mae = evaluation(y_true, y_pred, "mae")
    regression_mse = evaluation(y_true, y_pred, "mse")
    regression_r2 = evaluation(y_true, y_pred, "r2")
    print("="*(35 + len(model_name)))
    print(f"📊  Regression model evaluation ({model_name})")
    print("="*(35 +len(model_name)))
    print(f"🔹 MAE (Mean Absolute Error) : {regression_mae:.2f}")
    print(f"🔹 MSE (Mean Squared Error)  : {regression_mse:.2f}")
    print(f"🔹 R2  (Coefficient of Determination)  : {regression_r2:.2f}")

