# Priyanka Kasture | pkasture2010@gmail.com
# Multiple Linear Regression with Backward Elimination on the '50_Startups' Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the - dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

# Fitting the line
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

# For backward elimination - Used for reducing the number of features/independent variables
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values = X,axis=1)
sig = 0.05 # Significance level

X_optimal = X[:,[0,1,2,3,4,5]] # X_optimal initilized as the original matrix of features (6)
regressor_ols = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_ols.summary()

# Remove the feature that shows the highest p-value
# Lesser the p-value - lesser significant the feature
X_optimal = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_ols.summary()
# X_optimal now remains a feature matrix with just 2 significant features
# The process mentioned above can be automated by the use of loops
