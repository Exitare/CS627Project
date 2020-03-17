import matplotlib.pyplot as plt


def plot_box(df, folder, action):
    plt.figure()
    ax = df.boxplot()
    labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94', '95', '96', '97',
              '98' '99']
    ax.set_ylabel('R2 Score')
    ax.set_xlabel("Percentage of rows removed")
    ax.set_title('R2 Values for 5K Fold')
    ax.set_xticklabels(labels)
    plt.savefig(f"{folder}/{action}.jpg", dpi=None, format='png')
