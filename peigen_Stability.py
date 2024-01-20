import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, header=None)

# Preprocessing: Extract features and labels
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# Encode labels (Malignant 'M' as class 1 and Benignant 'B' as class 0)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Create a pipeline with StandardScaler and Logistic Regression
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l2', solver='liblinear'))])

# Define a range of values for the regularization parameter (C)
C_values = np.logspace(-6, 6, 13)  # Now there are 13 values from 10^-6 to 10^6.


# Calculate negative cross-validated accuracy for each value of C
negative_mean_scores = []
for C in C_values:
    pipe_lr.set_params(clf__C=C)
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores.append(-np.mean(scores))

# Take the absolute value to make the stability curve positive
positive_mean_scores = np.abs(negative_mean_scores)

# Plot the positive stability curve for Logistic Regression
plt.plot(C_values, positive_mean_scores, marker='o')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Cross-validated accuracy')
plt.title('Positive Stability Curve for Logistic Regression')
plt.grid(True)
plt.show()

# Fit the model with the best C value on the training data
best_C = C_values[np.argmin(positive_mean_scores)]
pipe_lr.set_params(clf__C=best_C)
pipe_lr.fit(X_train, y_train)

# Evaluate the model on the training data
train_accuracy = pipe_lr.score(X_train, y_train)

# Evaluate the model on the test data
test_accuracy = pipe_lr.score(X_test, y_test)

# Plot the model's fit on the training data and test data
plt.figure(figsize=(8, 6))
plt.bar(['Training Data', 'Test Data'], [train_accuracy, test_accuracy])
plt.ylim(0.9, 1.0)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Model Fit on Training and Test Data')
plt.show()
