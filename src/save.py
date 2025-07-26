import os

import joblib
import pandas as pd

def save_model_metrics(model_names, mae_list, rmse_list, r2_list, output_dir="../outputs/metrics_outputs"):
    """
    Saves model evaluation metrics (MAE, RMSE, R2) into a CSV file for further use in a Streamlit application.

    Parameters:
    - model_names (list): List of model name strings.
    - mae_list (list): List of Mean Absolute Error values.
    - rmse_list (list): List of Root Mean Squared Error values.
    - r2_list (list): List of R² scores.
    - output_dir (str): Path to the directory where the CSV file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        "Model": model_names,
        "MAE": mae_list,
        "RMSE": rmse_list,
        "R2": r2_list
    })

    # Save the DataFrame to a CSV file
    output_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    print(f"✅ Metrics saved to: {output_path}")



def save_trained_models(model_list, model_names, output_dir="../outputs/models"):
    """
    Saves each trained machine learning model to disk using Joblib for later use in a Streamlit app.

    Parameters:
    - model_list (list): List of trained model objects.
    - model_names (list): List of corresponding model name strings (must match in length).
    - output_dir (str): Path to the directory where models will be stored.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each model with its corresponding name
    for model, name in zip(model_list, model_names):
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"✅ Model '{name}' saved to: {model_path}")

