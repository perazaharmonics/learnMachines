# Chapter 4: Conditioning Data 
# 
# To elaborate, in Python, a dictionary is a collection of key-value pairs. In the dictionary size_mapping, each key k maps to a value v. 
# The code is creating a new dictionary where the keys and values are swapped: each v from size_mapping is a key in inv_size_mapping and maps to a corresponding k
# Mapping Ordinal Features
import pandas as pd
import numpy as np
df = pd.DataFrame([['green', 'M', 10.1, 
                    'class2'],
                   ['red', 'L', 13.5,
                    'class1' ],
                   ['blue', 'XL', 15.3,
                     'class2']])
df.columns = ['color', 'size', 'price',
               'classlabel']


size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
inv_size_mapping={v: k for k, v in size_mapping.items()}

# Here, class_mapping is a dictionary that maps each unique label in df['classlabel'] to a unique index number. 
# The enumerate function is used to generate these index numbers, and np.unique(df['classlabel']) returns an array of all unique values in the column 
# 'classlabel' from the pandas DataFrame df.
# Next, we show how to reverse this index placement practice
# REMEMBER: Classlabels aren't ordinal, but nominal since they can't be ordered in a particular fashion

df['size'].map(inv_size_mapping)

class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}

df['classlabel'] = df['classlabel'].map(class_mapping)
inv_class_mapping = {v: k for k, v in class_mapping.items()}

df['classlabel']=df['classlabel'].map(inv_class_mapping)

from sklearn.preprocessing import LabelEncoder as LE
class_le = LE()
y = class_le.fit_transform(df['classlabel'].values)
# >>> Y
class_le.inverse_transform(y)
df['size'].map(inv_size_mapping)
df['classlabel']=df['classlabel'].map(class_mapping)

from sklearn.preprocessing import OneHotEncoder as OHE
X = df[['color', 'size', 'price']].values
color_le=LE()
X[:, 0] = color_le.fit_transform(X[: , 0])
X = df[['color', 'size', 'price ']].values
color_ohe=OHE()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# >>> X

from sklearn.compose import ColumnTransformer as CT
X = df[['color', 'size', 'price']].values
c_transf= CT([
      ('onehot', OHE(), [0]),
      ('nothing','passthrough', [1, 2] ) 
            ])
c_transf.fit_transform(X).astype(float)
pd.get_dummies(df[['price', 'color', 'size']])
pd.get_dummies(df[['price', 'color', 'size']],
                drop_first=True)

# HOW TO: drop a redundant feature column which causes high collinearity and therefore reduces estimation on some models
color_ohe = OHE(categories='auto', drop='first')
c_transf= CT([
      ('onehot', OHE(), [0]),
      ('nothing','passthrough', [1, 2] ) 
            ])
c_transf.fit_transform(X).astype(float)

df = pd.DataFrame([['green', 'M', 10.1, 
                    'class2'],
                   ['red', 'L', 13.5,
                    'class1' ],
                   ['blue', 'XL', 15.3,
                     'class2']])
df.columns = ['color', 'size', 'price',
               'classlabel']

# >>> df


# this line creates a new column that indicates whether the corresponding value
# in the 'size' column is either 'L' or 'XL' using 1 and 0 as binary indicators.

df['x > M'] = df['size'].apply(
    lambda x: 1 if x in {'L', 'XL'} else 0)

# This line creates a column that indicates whether the corresponding value in the 'size' column is 'XL'. 


df['x > L'] = df['size'].apply(
    lambda x: 1 if x == 'XL' else 0
)
#This line deletes the 'size' column from the DataFrame df. It removes the column from the DataFrame, so you will no longer have the original 'size' column in your DataFrame 
# after executing this line.
del df['size']
# >>> df

## Pull the notorious wine data set
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns=['Class label', 'Alcohol',
                 'Malic Acid', 'Ash',
                 'Alcalinity of Ash', 'Magnesium',
                 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols','Proanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280 / OD315 of diluted wines',
                 'Prolines']
print('Class labels', np.unique(df_wine['Class label']))

""" Here we first assign the NumPy array representation of the feature columns 1-13 to the variable X and we assigned the class labels from the first column to the variable y
Then, we used the train_test_split function to randomly split x and y into separate training test and data sets. By setting test_size=0.3, 30 percent of the wine for testing purposes 
and the remaining 70 wine examples were assigned to X_train and y_train, respectively. Providing the class label array y to stratify as an argument, ensures that both the traning 
anf test datasets have the same class proportions as the original dataset. """
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Normalization as we know it in signal processing
from sklearn.preprocessing import MinMaxScaler as MMS
mms = MMS()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Explanation of standardization
# Standardization is a preprocessing technique used to transform the values of features to have zero mean and unit variance.
# It helps in bringing the features to a similar scale, which can be beneficial for certain machine learning algorithms and statistical analyses.
# By standardizing the features, we ensure that they have comparable ranges and prevent any particular feature from dominating the learning process.
# Here, we use the StandardScaler from the scikit-learn library to perform standardization on the numeric features in our dataset.
# We fit the scaler on the training set and then use it to transform both the training and testing sets.

ex = np.array(([0, 1, 2, 3, 4, 5]))
standrd=(ex - ex.mean()) / ex.std()
print('standardized', (ex - ex.mean()) / ex.std())
print('normalized', (ex - ex.min()/(ex.max() - ex.min())))

from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)

## For models that tend to overfit, try to use RobustScaler 

# If a model performs much better on the training dataset than on the test dataset, that means the model is prone to overfitting
# Overfitting implies that the model has no generalization abilities. We say the model has high variance

from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import pandas as pd
lr=LR(penalty='l1',
      C=1.0,
      solver='liblinear',
      multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test Accuracy: ', lr.score(X_test_std, y_test))

# In scikit-learn, intercept_ corresponds to the bias unit b and coef_ corresponds to the values wj
# z = w1x1 + w2x2 + ... + wm*xm + b = w.t*x + b
print(lr.intercept_)
print(lr.coef_)

# Enforcing sparsity by increasing the regularization strength
import matplotlib.pyplot as plt
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()





X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # Bringing features onto the same scale




mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)





stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# A visual example:



ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))



# # Selecting meaningful features

# ...

# ## L1 and L2 regularization as penalties against model complexity

# ## A geometric interpretation of L2 regularization









# ## Sparse solutions with L1-regularization





# For regularized models in scikit-learn that support L1 regularization, we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution:




LR(penalty='l1')


# Applied to the standardized Wine data ...




lr = LR(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))




lr.intercept_




np.set_printoptions(8)




lr.coef_[lr.coef_!=0].shape




lr.coef_





fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LR(penalty='l1', C=10.**c, solver='liblinear', 
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

#plt.savefig('figures/04_08.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)

plt.show()