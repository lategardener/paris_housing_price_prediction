from matplotlib import pyplot as plt
import seaborn as sns

def residual_distribution(y_test, y_pred):
    """
    Plot the distribution of residuals (errors) between true and predicted values.

    Parameters:
    - y_test: array-like, true target values
    - y_pred: array-like, predicted values by the model

    This function shows a histogram with a KDE (Kernel Density Estimate)
    to visualize how prediction errors are distributed.
    """
    # Calculate residuals (errors)
    residuals = y_test - y_pred

    # Set figure size for better readability
    plt.figure(figsize=(8, 5))

    # Plot histogram with KDE overlay for residuals distribution
    sns.histplot(residuals, kde=True)

    # Set the title and axis labels for clarity
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    # Display the plot
    plt.show()
