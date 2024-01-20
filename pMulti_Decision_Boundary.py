# Importing 'ListedColormap' from matplotlib's colors module
from matplotlib.colors import ListedColormap


# Importing the numpy library with alias as np
import numpy as np

# Function to plot decision regions for a classifier
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    import matplotlib.pyplot as plt
    # Setup markers for different classes
    markers = ('o', 's', '^', 'v', '<')

    # Define colors for different classes
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    # Create a colormap using colors for the unique class labels in 'y'
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Define the range for the first feature to plot
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # Define the range for the second feature to plot
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid to represent the feature space
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Use the classifier to predict the label for each point in the mesh grid
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # Reshape the predicted labels to have the same structure as the mesh grid
    lab = lab.reshape(xx1.shape)

    # Create a contour plot using the predicted labels and the colormap
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap = cmap)

    # Set the x-axis limits to match the first feature range
    plt.xlim(xx1.min(), xx1.max())

    # Set the y-axis limits to match the second feature range
    plt.ylim(xx2.min(), xx2.max())

    # Plot the data points for each unique class
    for idx, cl in enumerate(np.unique(y)):
        # Create a scatter plot for the data points of the current class
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker = markers[idx],
                    label=f'Class{cl}',
                    edgecolor='black')

    # If test indices are provided, highlight these examples
    if test_idx:
        # Get the test examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        # Create a scatter plot for the test examples
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')
