# Importing necessary libraries and modules
# To run on command line
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pLogReg import LogisticRegressionGD
from pMulti_Decision_Boundary import plot_decision_regions
""" 
1. Import Necessary Libraries and Modules:
 The script begins by importing necessary Python libraries such as NumPy, Matplotlib, and machine learning libraries from scikit-learn.

2. Load and Prepare the Dataset:
The script uses the Iris dataset from scikit-learn. It selects the third and fourth features, Petal Length and Petal Width, and splits the dataset into training and testing sets. The dataset is then standardized using the StandardScaler from scikit-learn for better model performance.
Perceptron Model:
A perceptron model is trained using the standardized training dataset. The perceptron is a simple algorithm suitable for binary classification tasks. The model parameters are the learning rate (eta0) and the random_state for reproducibility. The model is then used to predict the class labels for the test set. The number of misclassified examples and the accuracy of the model are printed out.

3. Logistic Regression Models:
Two types of logistic regression models are implemented: a custom one using gradient descent (for binary classification) and one from scikit-learn (for multi-class classification). Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. The parameters are the learning rate (eta), the number of iterations (n_iter), and the random_state for reproducibility.

4. Support Vector Machine (SVM) Model:
A linear SVM model is trained on the standardized dataset. SVM is a type of learning model that separates data by maximizing the margin between different classes. The parameters include the kernel function (kernel), the regularization parameter (C), and the random_state for reproducibility.

5. SGDClassifier Models:
Three types of classifiers (Perceptron, Logistic Regression, and SVM) are implemented using Stochastic Gradient Descent (SGD). SGD is an iterative method for optimizing an objective function with suitable smoothness properties. It's used when the dataset is large to speed up learning.

6. Decision Tree Model:
A decision tree model is trained on the dataset. Decision trees are a type of model that makes decisions based on certain conditions. It is very interpretable and doesn't require feature scaling. The tree is visualized using the plot_tree function from scikit-learn. The parameters are the criterion (gini), the maximum depth of the tree (max_depth), and the random_state for reproducibility.

7. Random Forest Model:
A random forest model, which is an ensemble of decision trees, is trained on the dataset. Random forests help to overcome the overfitting problem of decision trees by averaging multiple deep decision trees, trained on different parts of the same training set. The parameters are the number of trees (n_estimators), the random_state for reproducibility, and n_jobs for parallel computing.

8. K-Nearest Neighbors (KNN) Model:
A KNN model is trained on the dataset. KNN is a type of instance-based learning where the function is only approximated locally, and all computation is deferred until classification. The parameters are the number of neighbors (n_neighbors), the power parameter for the Minkowski metric (p), and the distance metric (metric).

Note: The script also includes a plot of the decision boundaries for each model using a custom plot_decision_regions function. This helps to visualize how well each model separates the classes in the feature space. Furthermore, some statistical measures like Gini impurity, entropy, and misclassification error are calculated and plotted to provide insights into the impurity of classifications."""

# Loading iris dataset from sklearn datasets
iris = datasets.load_iris()

# Extracting only the 3rd and 4th features (petal length and petal width)
X = iris.data[:, [2,3]]

# Extracting the corresponding labels
y = iris.target

print('Class labels', np.unique(y))  

# Importing the train_test_split function from sklearn model_selection module
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))  
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))  

# Importing StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier = ppn,
                      test_idx=range(105, 150))  

plt.xlabel('Petal Length [ standardized ]')
plt.ylabel('Petal Width [ standardized ]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)

lrgd.fit(X=X_train_01_subset,y=y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier = lrgd)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, solver='lbfgs', 
                        multi_class='multinomial')
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = lr,
                      test_idx=range(105,150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

from sklearn.svm import SVC
svm = SVC( kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std , y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = svm,
                      test_idx=range(105,150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss = 'perceptron')
lr = SGDClassifier(loss = 'log')
svm = SGDClassifier(loss = 'hinge')

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue', marker = 's',
            label = 'Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato', marker = 'o',
            label = 'Class 0')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

svm = SVC( kernel='rbf', C=.0001, gamma = 0.31333333, random_state=1)
svm.fit(X_xor , y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Defining function to calculate Gini impurity
def gini(p):
    return p * (1-p) + (1 - p)*(1-(1-p)) 

# Defining function to calculate Entropy
def entropy(p):
    return -p * np.log2(p) - ( 1 - p ) * np.log2((1 -p))

# Defining function to calculate Misclassification error
def error(p):
    return 1 - np.max([p, 1 - p])

# Create an array of equally spaced values from 0.01 to 1.0
x = np.arange(0.01, 1.0, 0.01)

# Apply the entropy function to each element of x (excluding zero)
ent = [entropy(p) if p !=0 else None for p in x]

# Scale the entropy
sc_ent = [e * 0.5 if e else None for e in ent]

# Apply the error function to each element of x
err = [error(i) for i in x]

# Creating a new figure
fig = plt.figure()

# Adding a subplot
ax = plt.subplot(111)

# Plotting entropy, scaled entropy, gini impurity and misclassification error
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy(scale)', 'Gini impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgreen', 'red', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, color=c)

# Setting legend location and format
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)

# Drawing horizontal lines at 0.5 and 1.0
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

# Setting y limit and labels for x and y axes
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')

# Displaying the plot
plt.show()

# Importing DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier as DCS

# Creating a decision tree model
tree_model = DCS(criterion='gini',
                 max_depth=4,
                 random_state=1)
# Training the model
tree_model.fit(X_train, y_train)

# Combining training and testing data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Plotting the decision regions
plot_decision_regions(X_combined, 
                      y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))

# Setting labels for x and y axes
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()

# Displaying the plot
plt.show()

# Importing tree from sklearn for visualizing the decision tree
from sklearn import tree

# Setting the feature names
feature_names = ['Sepal length', 'Sepal width',
                 'Petal length', 'Petal Width']

# Plotting the decision tree
tree.plot_tree(tree_model,
               feature_names=feature_names,
                filled=True )

# Displaying the plot
plt.show()

###############################################################################################################################################################################
# Random Forest Classifier using ensebles of decision trees
###############################################################################################################################################################################
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25,
                                random_state=2,
                                n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, 
                      classifier = forest, test_idx=range(105,150))
plt.xlabel('Petal Length(cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(loc='upper left')
plt.show()

#################################################################################################################################################################################
# Implementation for a k-nearest-neighbor (KNN) classification algorithm.
################################################################################################################################################################################
from sklearn.neighbors import KNeighborsClassifier as KNC
knn = KNC(n_neighbors=5,p=2,
          metric= 'minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('Petal length (standarized)')
plt.ylabel('Petal Width (standarized)')
plt.legend(loc='upper left')
plt.show()



