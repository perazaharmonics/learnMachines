# Import numpy library
from matplotlib import markers
from pErceptron import Perceptron
from pDecision_Boundary import plot_decision_regions
from pAdaline import AdalineGD as AGD
from pAdalineSGD import AdalineSGD as ASGD
import numpy as np
import os
import pandas as pd

# URL of the dataset
s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

# Print the URL
print('From URL:', s)

# Read the dataset from the URL into a pandas DataFrame
df = pd.read_csv(s, header=None, encoding='utf-8')

# Print the last 5 rows of the DataFrame
df.tail()

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# You import numpy again, but assign it to a different name (py) which you don't use later
import numpy as py

# Select Setosa and Versicolor from the species column and convert them to 0 and 1 respectively
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract Sepal Length and Petal Length
X = df.iloc[0:100, [ 0, 2]].values

# Plot Sepal Length and Petal Length for Setosa
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'Setosa')

# Plot Sepal Length and Petal Length for Versicolor
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 's', label = 'Versicolor')

# Set labels for the x and y axis
plt.xlabel('Sepal length[cm]')
plt.ylabel('Petal length[cm]')

# Add a legend at the upper left corner
plt.legend(loc='upper left')

# Display any pending plots. In your previous code, this would likely show a scatter plot of Iris data.
plt.show()

# Instantiate a Perceptron object named ppn with learning rate (eta) of 0.1 and 10 iterations (n_iter).
ppn = Perceptron(eta=0.1, n_iter=10)

# Fit the Perceptron to your data. The X input data and y output labels were defined earlier in your code.
ppn.fit(X, y)

# Create a plot of the errors that the Perceptron made in each iteration (epoch).
# The range() function creates a list of epoch numbers, which is used as the x-values in the plot.
# The y-values are the number of errors (misclassifications) in each epoch.
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')

# Set the label for the x-axis to 'Epochs'.
plt.xlabel('Epochs')

# Set the label for the y-axis to 'Number of updates', referring to the number of weight updates.
plt.ylabel('Number of updates')

# Display the plot, which shows how the number of errors decreased over the epochs.
plt.show()
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

# We import pyplot's subplots function for creating multiple layouts of subplots on a single figure
from matplotlib.pyplot import subplots

# We create a figure and two subplots (1 row, 2 columns) and specify the figure size
fig, ax = subplots(nrows=1, ncols = 2, figsize =(10, 4))

# We create an AdalineGD classifier (AGD) with a learning rate of 0.1 and fit the data
ada1 = AGD(n_iter=15, eta = 0.1 ).fit(X,y)

# We plot the log of the Mean Squared Error (MSE) for each epoch on the first subplot
ax[0].plot(range(1, len(ada1.losses_)+1), np.log10(ada1.losses_), marker = 'o')

# We set the labels of the x and y axes for the first subplot
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean Squared Error)')

# We set the title for the first subplot
ax[0].set_title('log(Adaline - Learning Rate = 0.1)')

# We create another AdalineGD classifier with a learning rate of 0.0001 and fit the data
ada2 = AGD(n_iter=15, eta = 0.001).fit(X,y)

# We plot the log of the MSE for each epoch on the second subplot
ax[1].plot(range(1, len(ada2.losses_)+1), np.log10(ada2.losses_), marker = 'o')

# We set the labels of the x and y axes for the second subplot
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean Squared Error)')

# We set the title for the second subplot
ax[1].set_title('log(Adaline - Learning Rate = 0.0001)')

# Display the figure with its two subplots
plt.show()

# We create a copy of our feature matrix X
X_std = np.copy(X)

# We standardize the first feature (sepal length) by subtracting its mean and dividing by its standard deviation
X_std[:,0] = (X[:,0]-X[:,0].mean()) / X[:,0].std()

# We standardize the second feature (petal length) in the same way
X_std[:,1] = (X[:,1]-X[:,1].mean()) / X[:,1].std()

# We create an AdalineGD classifier with a learning rate of 0.5
ada_gd = AGD(n_iter=15, eta=0.5)

# We train the classifier using the standardized features and the target values
ada_gd.fit(X_std, y)

# We use our function to plot the decision regions, providing the standardized features,
# the target values, and the trained classifier
plot_decision_regions(X_std, y, classifier = ada_gd)

# We set the title of the plot and the labels of the x and y axes
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length (Normalized)')
plt.ylabel('Petal Length (Normalized)')

# We add a legend to the plot
plt.legend(loc = 'upper left')

# We adjust the padding between and around the subplots
plt.tight_layout()

# Display the plot
plt.show()

# We plot the mean squared errors recorded in each epoch during the training of the classifier
plt.plot(range(1, len(ada_gd.losses_)+1), ada_gd.losses_, marker='o')

# We set the labels of the x and y axes
plt.xlabel('Epochs')
plt.ylabel('log(Mean Squared Error)')

# We adjust the padding between and around the subplots
plt.tight_layout()

# Finally, we display this plot
plt.show()

# We create an AdalineGD classifier with a learning rate of 0.01
ada_sgd = ASGD(n_iter=15, eta=0.01, random_state = 1)

# We train the classifier using the standardized features and the target values
ada_sgd.fit(X_std, y)

# We use our function to plot the decision regions, providing the standardized features,
# the target values, and the trained classifier
plot_decision_regions(X_std, y, classifier = ada_sgd)
 
# We set the title of the plot and the labels of the x and y axes
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal Length (Normalized)')
plt.ylabel('Petal Length (Normalized)')

# We add a legend to the plot
plt.legend(loc = 'upper left')

# We adjust the padding between and around the subplots
plt.tight_layout()

# Display the plot
plt.show()

# We plot the mean squared errors recorded in each epoch during the training of the classifier
plt.plot(range(1, len(ada_sgd.losses_)+1), ada_sgd.losses_, 
         marker='o')

# We set the labels of the x and y axes
plt.xlabel('Epochs')
plt.ylabel('Average Loss')

# We adjust the padding between and around the subplots
plt.tight_layout()

# Finally, we display this plot
plt.show()