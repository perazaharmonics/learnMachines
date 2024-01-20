import numpy as np

class AdalineGD:
    """Adaptive Linear Neuron classifier.

    Parameters
    ---------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    errors_ : list
        Mean Square Error in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
    
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

        Algorithm:
        1. Initialize weights to a random normal distribution and bias unit to 0.
        2. For each epoch:
            a. Compute the net input (sum of the product of input and weights, plus bias).
            b. Compute the activation function (linear activation function - identity function in this case).
            c. Compute the errors (difference between true and predicted values).
            d. Update the weights based on the gradient descent method.
            e. Update the bias unit based on the average errors.
            f. Compute the Mean Square Error (MSE) and store it in a list.
        3. Repeat steps 2a - 2f for n_iter times.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # Initialize weights
        self.b_ = 0.0  # Initialize bias
        self.losses_ = []  # Initialize error list

        for _ in range(self.n_iter):
            net_input = self.net_input(X)  # Compute the net input
            output = self.activation(net_input)  # Compute the activation function
            errors = (y - output)  # Compute the errors
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]  # Update weights
            self.b_ += self.eta * 2.0 * errors.mean()  # Update bias
            loss = (errors**2).mean()  # Compute MSE
            self.losses_.append(loss)  # Store the MSE

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
