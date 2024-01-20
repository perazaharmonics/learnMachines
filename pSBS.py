from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
"""
This Sequential Backward Selection (SBS) algorithm aims to reduce the dimensionality of the initial dataset 
with a minimum decay in performance of the classifier 
to improve upon computational efficiency. 
It does this by successively removing features.
Developer: Enrique Peraza
"""
class SBS:
    # Initialization function
    def __init__(self, estimator, k_features,
                scoring=accuracy_score,
                    test_size=0.25, random_state=1):
        self.scoring=scoring  # Metric function for evaluation
        self.estimator=clone(estimator)  # ML algorithm to use
        self.k_features = k_features  # Desired number of features
        self.test_size = test_size  # Proportion of dataset to include in the test split
        self.random_state=random_state  # Controls the shuffling applied to the data before applying the split

    # Fitting function
    def fit(self, X, y):
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        
        # Initialize necessary variables
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))  # Starting with all features
        self.subsets_ = [self.indices_]  
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)  # Calculate initial score
        self.scores_ = [score]

        # Sequentially remove features
        while dim > self.k_features:
            scores=[]
            subsets=[]

            # Iterate over all combinations of features of length dim - 1
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)  # Calculate score
                scores.append(score)
                subsets.append(p)

            # Find best score and associated subset
            best=np.argmax(scores)
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        
        # Final score
        self.k_score_ = self.scores_[-1]

        return self

    # Function to reduce dataset to selected features
    def transform(self, X):
        return X[:, self.indices_]

    # Function to compute score given a feature set
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)  # Fit model with given feature set
        y_pred = self.estimator.predict(X_test[:, indices])  # Make predictions
        score = self.scoring(y_test, y_pred)  # Compute score of predictions
        return score
 