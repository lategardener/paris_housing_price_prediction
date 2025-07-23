import json
import os

def register_param(file_path, model_name, score, params):
    """
    Saves a model's best parameters and score to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        model_name (str): Unique name of the model (key in the JSON).
        score (float): Current model score (higher = better).
        params (dict): Dictionary of model parameters.

    Behavior:
        - If the file doesn't exist, it is created.
        - If the model already exists, the scores are compared and the best is kept.
    """
    # Load or create the JSON content
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Check if model already present and compare scores
    if model_name in data:
        old_score = data[model_name].get("score", float("-inf"))
        if score <= old_score:
            print(f"Current score ({score}) <= saved score ({old_score}), no update performed.")
            return
        else:
            print(f"New better score ({score} > {old_score}), updating parameters.")

    else:
        print(f"Adding new model {model_name} with score {score}.")

    # Update data
    data[model_name] = {
        "score": score,
        "params": params
    }

    # Save to JSON file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {file_path}")
