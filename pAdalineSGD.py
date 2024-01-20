import numpy as np

class AdalineSGD:
    """Adaptive Linear Neuron classifier.
    ################################################################################
    AdalineSGD machine:

    1. Initialization: You create an instance of the AdalineSGD class, setting the learning rate (`eta`), number of iterations (`n_iter`), whether to shuffle the training data (`shuffle`), and a seed for random weight initialization (`random_state`).

    2. Fit method: The `fit` method takes in the feature matrix (`X`) and target values (`y`). It first initializes the weights by calling the `_initialize_weights` method. Then, for the number of iterations specified (`n_iter`), it shuffles the data (if `shuffle` is set to True) and updates the weights for each sample in the dataset. It calculates the loss for each sample, then stores the average loss for that epoch.

    3. Partial fit method: The `partial_fit` method works similarly to `fit`, but it doesn't re-initialize the weights if they are already initialized. This is useful for online learning where you might receive data in chunks and want to update your model incrementally.

    4. Shuffle method: The `_shuffle` method randomizes the order of the samples in the training data. This is important for Stochastic Gradient Descent to ensure that the samples' order doesn't influence the learning.

    5. Initialize weights method: The `_initialize_weights` method sets the weights (`w_`) to small random numbers and bias unit (`b_`) to zero.

    6. Update weights method: The `_update_weights` method applies the Adaline learning rule to adjust the weights based on the calculated error, which is the difference between the actual and predicted outputs. This is the core of the gradient descent optimization algorithm.

    7. Net input method: The `net_input` method calculates the weighted sum of the inputs and the bias unit, which is the net input to the neuron.

    8. Activation method: The `activation` method in this implementation just returns the input it receives. This is because Adaline uses an identity function for its activation, meaning the output of the neuron is a linear combination of the inputs.

    9. Predict method: The `predict` method returns the predicted class labels by applying a threshold function to the neuron's output. If the output is greater or equal to 0, it returns 1, otherwise, it returns 0.

    In terms of the general algorithm for Adaline Stochastic Gradient Descent:

    1. Initialize the weights to small random numbers and the bias to 0.
    2. For each epoch (an epoch being one pass through the entire dataset):
        1. If `shuffle` is True, shuffle the training data.
        2. For each sample in the dataset:
            1. Compute the output of the Adaline neuron (the weighted sum of the inputs and the bias).
            2. Calculate the error (the difference between the actual and predicted outputs).
            3. Update the weights and bias according to the Adaline learning rule, which involves multiplying the learning rate, the input value, and the error, then adding this to the current weight (and for the bias, we just consider the learning rate and error).
   3. Compute the average loss for the epoch.
3. Repeat the above steps for the specified number of iterations.
4. When predicting, compute the output of the Adaline neuron for the given input and apply a threshold function (if the output is greater or equal to 0, return 1, otherwise return 0).


    ################################################################################
    Parameters
    ---------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    shuffle: bool (default: true)
        Shuffles training data every epoch if true to prevent
        cycles
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean Square Error in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle = True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit training data.
        Parameters:
        -----------
        X: {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features
        y: array-like, shape = [n_examples]

        Returns: 
        self : object 
     
        """
        
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self
    

    def partial_fit(self, X, y):
       """ Fit training data without reinitializing the weights"""
       if not self.w_initialized: 
           self._initialize_weights(X.shape[1])
       if y.ravel().shape[0] > 1:           
           for xi, target in zip(X,y):
               self._update_weights(xi, target)
       else:
           self._update_weights(X, y)
       return self
       

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    

    def _initialize_weights(self, m):
       """Initialize weights to small random numbers """
      
       self.rgen = np.random.RandomState(self.random_state)
       self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, 
                                  size =m)
       self.b_ = np.float_(0.)
       self.w_initialized = True

    def _update_weights(self, xi, target):
       """Apply Adaline learning rule to update the weights"""
       output = self.activation(self.net_input(xi))
       error = (target - output)
       self.w_ += self.eta * 2.0 * xi * error
       self.b_ += self.eta * 2.0 * error
       loss =  error**2
       return loss
       
    def net_input(self, X):
       """Calculate net input"""
       return np.dot(X, self.w_) + self.b_
       
    def activation(self, X):
       """Compute linear activation"""
       return X
    
    def predict(self, X):
       """Return class label after unit step"""
       return np.where(self.activation(self.net_input(X))
                        >= 0.0, 1, 0)
