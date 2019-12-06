import matplotlib.pyplot as plt

def show_plot(x, y, axis: str):
    plt.plot([0, 2.5], [0, 2.5], 'k', lw=0.5)  # reference diagonal
    plt.plot(x, y, '.')
    plt.axis(axis)
    plt.show()