import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Create a pipeline with StandardScaler and SVM classifier
pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', gamma=0.1, C=1.0))])

# Define a range of values for the regularization parameter (C)
C_values = np.logspace(-4, 4, 9)

# Calculate positive stability curve for SVM
negative_mean_scores_svm = []
for C in C_values:
    pipe_svm.set_params(clf__C=C)
    scores = cross_val_score(estimator=pipe_svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_svm.append(-np.mean(scores))

positive_mean_scores_svm = np.abs(negative_mean_scores_svm)

# Plot the stability curve for SVM
plt.figure(figsize=(8, 6))
plt.plot(C_values, positive_mean_scores_svm, marker='o', label='SVM Stability')
plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Stability')
plt.title('SVM Stability')
plt.legend()
plt.grid(True)
plt.show()

# Fit the best model on the training data
best_C_svm = C_values[np.argmin(positive_mean_scores_svm)]
pipe_svm.set_params(clf__C=best_C_svm)
pipe_svm.fit(X_train, y_train)

# Evaluate the model on the training and test data
train_accuracy_svm = pipe_svm.score(X_train, y_train)
test_accuracy_svm = pipe_svm.score(X_test, y_test)

print(f"Best C value for SVM: {best_C_svm}")
print(f"SVM Train Accuracy: {train_accuracy_svm:.2f}")
print(f"SVM Test Accuracy: {test_accuracy_svm:.2f}")

# Apply PCA to reduce the features to 2 components for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Fit the best model on the reduced training data
best_C_svm = C_values[np.argmin(positive_mean_scores_svm)]
pipe_svm.set_params(clf__C=best_C_svm)
pipe_svm.fit(X_train_pca, y_train)

# Evaluate the model on the training and test data
train_accuracy_svm = pipe_svm.score(X_train_pca, y_train)
test_accuracy_svm = pipe_svm.score(X_test_pca, y_test)

print(f"Best C value for SVM: {best_C_svm}")
print(f"SVM Train Accuracy: {train_accuracy_svm:.2f}")
print(f"SVM Test Accuracy: {test_accuracy_svm:.2f}")

# Scatter plot with the decision boundary for SVM on the test data
plt.figure(figsize=(8, 6))
h = .02  # step size in the mesh
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = pipe_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot class samples
plt.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], c='blue', label='Benignant', edgecolors='k')
plt.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], c='red', label='Malignant', edgecolors='k')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary on Test Data (PCA)')
plt.legend()
plt.show()
