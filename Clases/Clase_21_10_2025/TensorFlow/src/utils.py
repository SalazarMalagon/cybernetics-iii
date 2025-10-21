def plot_results(y, y2, classes2, classes3):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.bar(range(len(y[0])), y[0], color=[.6, .6, .6], edgecolor='k')
    plt.title('Output for Dog Class (Training)')

    plt.subplot(4, 1, 2)
    plt.bar(range(len(y[1])), y[1], color=[.6, .6, .6], edgecolor='k')
    plt.title('Output for Cat Class (Training)')

    plt.subplot(4, 1, 3)
    plt.bar(range(len(y2[0])), y2[0], color=[.6, .6, .6], edgecolor='k')
    plt.title('Output for Dog Class (Testing)')

    plt.subplot(4, 1, 4)
    plt.bar(range(len(y2[1])), y2[1], color=[.6, .6, .6], edgecolor='k')
    plt.title('Output for Cat Class (Testing)')

    plt.tight_layout()
    plt.show()

def convert_to_classes(y):
    import numpy as np
    return np.argmax(y, axis=0)

def calculate_performance(y_true, y_pred):
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Flatten arrays to handle different shapes like (n,) vs (n,1)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    if y_true.size == 0:
        return 0.0

    return float((y_true == y_pred).sum()) / float(y_true.size)