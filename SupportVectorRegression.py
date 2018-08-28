# SVR : Support Vector Regression
# Priyanka Kasture | pkasture2010@gmail.com

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# We won't be splitting the dataset into the Training set and Test set, because the dataset is small
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling - because there's a huge difference in the range of y as compared to X
# We reduce the values of X and y between -1 and +1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
# Scaled X and Y
X = sc_X.fit_transform(X)
Y = y.reshape((len(y),1))
y = sc_y.fit_transform(Y)

'''# Inverse transforms are used to get back the original values
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)'''

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result (If you want to know what the salary of a person with 6.5 years of experience should be)
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) 
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
