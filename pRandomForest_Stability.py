import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

# Create a pipeline with StandardScaler and Random Forest classifier
pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=150, random_state=1))])

# Define a range of values for the number of estimators
n_estimators_values = [50, 100, 150, 200, 250]

# Calculate positive stability curve for Random Forest
negative_mean_scores_rf = []
for n_estimators in n_estimators_values:
    pipe_rf.set_params(clf__n_estimators=n_estimators)
    scores = cross_val_score(estimator=pipe_rf, X=X_train, y=y_train, cv=10, n_jobs=-1)
    negative_mean_scores_rf.append(-np.mean(scores))

positive_mean_scores_rf = np.abs(negative_mean_scores_rf)

# Plot the stability curve for Random Forest
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_values, positive_mean_scores_rf, marker='o', label='Random Forest Stability')
plt.xlabel('Number of Estimators')
plt.ylabel('Stability')
plt.title('Random Forest Stability')
plt.legend()
plt.grid(True)
plt.show()

# Fit the best model on the training data
best_n_estimators_rf = n_estimators_values[np.argmin(positive_mean_scores_rf)]
pipe_rf.set_params(clf__n_estimators=best_n_estimators_rf)
pipe_rf.fit(X_train, y_train)

# Evaluate the model on the training and test data
train_accuracy_rf = pipe_rf.score(X_train, y_train)
test_accuracy_rf = pipe_rf.score(X_test, y_test)

# Bar graph for model performance on the training and test data for Random Forest
plt.figure(figsize=(6, 4))
bar_width = 0.35
index = np.arange(2)
plt.bar(index, [train_accuracy_rf, test_accuracy_rf], bar_width, label=['Training Data', 'Test Data'], color=['blue', 'orange'])
plt.xticks(index + bar_width/2, ['Training Data', 'Test Data'])
plt.ylim(0.9, 1.0)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Random Forest Model Performance on Training and Test Data')
plt.legend()
plt.grid(True)
plt.show()
