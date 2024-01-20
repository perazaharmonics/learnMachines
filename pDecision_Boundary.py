import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')  # Set of marker styles for the scatter plot
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # Set of colors for the scatter plot
    cmap = ListedColormap(colors[:len(np.unique(y))])  # Create a color map from the list of colors
    
    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Find minimum and maximum values for the first feature (sepal length)
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Find minimum and maximum values for the second feature (petal length)

    # Create a mesh grid for the decision surface
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # Create a mesh grid from the minimum and maximum values of both features

    # Predict the class labels for all samples in the grid using the trained model (classifier)
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # ravel() function returns a flattened one-dimensional array, and T (transpose) is used to make the array shape compatible with classifier.predict()
    lab = lab.reshape(xx1.shape)  # Reshape the predicted labels to have the same structure as xx1 or xx2

    # Draw the decision boundary
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)  # contourf() function draws filled contours, alpha sets the transparency level
    plt.xlim(xx1.min(), xx1.max())  # Set limits for x-axis based on the minimum and maximum values of xx1
    plt.ylim(xx2.min(), xx2.max())  # Set limits for y-axis based on the minimum and maximum values of xx2

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):  # Loop over each unique class label in y
        plt.scatter(x=X[y == cl, 0],  # Scatter plot for feature 1 (sepal length) where class label is cl
                    y=X[y == cl, 1],  # Scatter plot for feature 2 (petal length) where class label is cl
                    alpha=0.8,  # Alpha blending value, between 0 (transparent) and 1 (opaque)
                    c=colors[idx],  # Color for each class
                    marker=markers[idx],  # Marker style for each class
                    label=cl,  # The label to be added to the legend
                    edgecolor='black')  # Color of the edge of the marker
