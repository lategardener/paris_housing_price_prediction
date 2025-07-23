from matplotlib import pyplot as plt
import seaborn as sns

def residual_distribution(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()