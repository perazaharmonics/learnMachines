import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset (your code for data loading here)

# Your preprocessing code here
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

# Create pipelines with StandardScaler and classifiers
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l2', solver='liblinear'))])
pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf',  C=1.0))])
pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=1))])

# Define a range of values for the regularization parameter (C)
C_values = np.logspace(-4, 4, 9)

# Calculate positive stability curve for Logistic Regression
negative_mean_scores_lr = []
for C in C_values:
    pipe_lr.set_params(clf__C=C)
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_lr.append(-np.mean(scores))

positive_mean_scores_lr = np.abs(negative_mean_scores_lr)

# Calculate positive stability curve for SVM
negative_mean_scores_svm = []
for C in C_values:
    pipe_svm.set_params(clf__C=C)
    scores = cross_val_score(estimator=pipe_svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_svm.append(-np.mean(scores))

positive_mean_scores_svm = np.abs(negative_mean_scores_svm)

# Calculate positive stability curve for Random Forest
negative_mean_scores_rf = []
for C in C_values:
    pipe_rf.set_params(clf__n_estimators=C)
    scores = cross_val_score(estimator=pipe_rf, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_rf.append(-np.mean(scores))

positive_mean_scores_rf = np.abs(negative_mean_scores_rf)

# Plot the stability curves for all classifiers
plt.figure(figsize=(10, 6))
plt.plot(C_values, positive_mean_scores_lr, marker='o', label='Logistic Regression Stability')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Stability')
plt.title('Logistic Regression Stability')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(C_values, positive_mean_scores_svm, marker='o', label='SVM Stability')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Stability')
plt.title('SVM Stability')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(C_values, positive_mean_scores_rf, marker='o', label='Random Forest Stability')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Stability')
plt.title('Random Forest Stability')
plt.legend()
plt.grid(True)
plt.show()

# Fit the best models on the training data
best_C_lr = C_values[np.argmin(positive_mean_scores_lr)]
pipe_lr.set_params(clf__C=best_C_lr)
pipe_lr.fit(X_train, y_train)

best_C_svm = C_values[np.argmin(positive_mean_scores_svm)]
pipe_svm.set_params(clf__C=best_C_svm)
pipe_svm.fit(X_train, y_train)

best_n_estimators_rf = C_values[np.argmin(positive_mean_scores_rf)]
pipe_rf.set_params(clf__n_estimators=best_n_estimators_rf)
pipe_rf.fit(X_train, y_train)

# Evaluate the models on the training and test data
train_accuracy_lr = pipe_lr.score(X_train, y_train)
test_accuracy_lr = pipe_lr.score(X_test, y_test)

train_accuracy_svm = pipe_svm.score(X_train, y_train)
test_accuracy_svm = pipe_svm.score(X_test, y_test)

train_accuracy_rf = pipe_rf.score(X_train, y_train)
test_accuracy_rf = pipe_rf.score(X_test, y_test)

# Bar graph for model performance on the training and test data for Logistic Regression
plt.figure(figsize=(8, 6))
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

# Bar graph for model performance on the training and test data for SVM
plt.figure(figsize=(8, 6))
plt.bar(index, [train_accuracy_svm, test_accuracy_svm], bar_width, label=['Training Data', 'Test Data'], color=['blue', 'orange'])
plt.xticks(index + bar_width/2, ['Training Data', 'Test Data'])
plt.ylim(0.9, 1.0)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('SVM Model Performance on Training and Test Data')
plt.legend()
plt.grid(True)
plt.show()

# Bar graph for model performance on the training and test data for Random Forest
plt.figure(figsize=(8, 6))
plt.bar(index, [train_accuracy_rf, test_accuracy_rf], bar_width, label=['Training Data', 'Test Data'], color=['blue', 'orange'])
plt.xticks(index + bar_width/2, ['Training Data', 'Test Data'])
plt.ylim(0.9, 1.0)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Random Forest Model Performance on Training and Test Data')
plt.legend()
plt.grid(True)
plt.show()
