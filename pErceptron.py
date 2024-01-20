import numpy as np

####################################################################################################
#The Perceptron is a binary classifier that uses a linear decision boundary to separate classes.
#It takes a set of input features and assigns a class label based on the weighted sum of the inputs.
####################################################################################################

class Perceptron:
    """Perceptron classifier.
    
    Parameters
    ---------
    eta: float
        Learning rate (between 0.00 and 1.00)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    b_ : Scalar
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit training data
    
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        
        y : array-like, shape = [n_examples]
            Target values.
    
        Returns
        -------
        self : object
        """
            # Algorithm:
    # 1. Initialize the weights randomly and bias unit to 0.
    # 2. For each epoch (repeat n_iter times):
    #    a. Initialize the errors counter to 0.
    #    b. For each training example xi and target value in zip(X, y):
    #       i. Calculate the update based on the prediction and target:
    #          update = learning_rate * (target - predict(xi))
    #       ii. Update the weights:
    #           weights += update * xi
    #       iii. Update the bias unit:
    #           bias += update
    #       iv. Count the number of updates (misclassifications):
    #           errors += 1 if update != 0 else 0
    #    c. Append the number of errors in this epoch to the errors list.

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # Initialize weights randomly
        self.b_ = np.float_(0.)  # Initialize bias unit
        self.errors_ = []  # Initialize errors list

        for _ in range(self.n_iter):  # Iterate for the specified number of epochs
            errors = 0
            for xi, target in zip(X, y):  # Iterate over each training example and its target value
                update = self.eta * (target - self.predict(xi))  # Calculate the update based on the prediction and target
                self.w_ += update * xi  # Update weights
                self.b_ += update  # Update bias unit
                errors += int(update != 0)  # Count the number of updates (misclassifications)
            self.errors_.append(errors)  # Append the number of errors in this epoch to the errors list
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_  # Calculate the net input by taking the dot product of weights and input features, and adding the bias

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Apply the unit step function and return the predicted class label
