from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import numpy as np

def residual_distribution(y_test, y_pred):
    """
    Plot the distribution of residuals (errors) between true and predicted values.

    Parameters:
    - y_test: array-like, true target values
    - y_pred: array-like, predicted values by the model

    This function shows a histogram with a KDE (Kernel Density Estimate)
    to visualize how prediction errors are distributed, along with summary statistics.
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='skyblue', bins=30)

    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    # Add a vertical line at zero residual
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')

    # Add text box with summary stats
    mean_res = residuals.mean()
    std_res = residuals.std()
    textstr = f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}'
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.legend()
    plt.show()


def plot_learning_curve(model, X_train, y_train):
    """
    Plots the learning curve of a model using training and cross-validation sets.
    Useful to detect underfitting or overfitting.
    """

    # Compute training and test scores using cross-validation
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring='neg_mean_squared_error',  # Use negative MSE (sklearn convention), will reverse sign later
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )

    # Calculate mean errors across the folds (note: errors are negated to be positive)
    train_errors_mean = -np.mean(train_scores, axis=1)
    test_errors_mean = -np.mean(test_scores, axis=1)

    # Plot the learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors_mean, 'o-', label="Training Error", color="blue")
    plt.plot(train_sizes, test_errors_mean, 'o-', label="Validation Error", color="green")

    # Axis labels and title
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
