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

# Create a pipeline with StandardScaler and Logistic Regression classifier
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l2', solver='liblinear'))])

# Define a range of values for the regularization parameter (C)
C_values = np.logspace(-4, 4, 9)

# Calculate positive stability curve for Logistic Regression
negative_mean_scores_lr = []
for C in C_values:
    pipe_lr.set_params(clf__C=C)
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_lr.append(-np.mean(scores))

positive_mean_scores_lr = np.abs(negative_mean_scores_lr)

# Plot the stability curve for Logistic Regression
plt.figure(figsize=(8, 6))
plt.plot(C_values, positive_mean_scores_lr, marker='o', label='Logistic Regression Stability')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Stability')
plt.title('Logistic Regression Stability')
plt.legend()
plt.grid(True)
plt.show()

# Fit the best model on the training data
best_C_lr = C_values[np.argmin(positive_mean_scores_lr)]
pipe_lr.set_params(clf__C=best_C_lr)
pipe_lr.fit(X_train, y_train)

# Evaluate the model on the training and test data
train_accuracy_lr = pipe_lr.score(X_train, y_train)
test_accuracy_lr = pipe_lr.score(X_test, y_test)

# Bar graph for model performance on the training and test data for Logistic Regression
plt.figure(figsize=(6, 4))
bar_width = 0.35
index = np.arange(2)
plt.bar(index, [train_accuracy_lr, test_accuracy_lr], bar_width, label=['Training Data', 'Test Data'], color=['blue', 'orange'])
plt.xticks(index + bar_width/2, ['Training Data', 'Test Data'])
plt.ylim(0.9, 1.0)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Model Performance on Training and Test Data')
plt.legend()
plt.grid(True)
plt.show()
