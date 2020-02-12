import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor


def show_plot(x, y, axis: str):
    """
    Plots the predications.
    Not used right now
    """
    plt.plot([0, 2.5], [0, 2.5], 'k', lw=0.5)  # reference diagonal
    plt.plot(x, y, '.')
    plt.axis(axis)
    plt.show()


def plot_validation_curve(X, y):
    train_sizes, train_scores, valid_scores = validation_curve(
        validation_curve(RandomForestRegressor()), X, y, "alpha", 1, cv=5)
