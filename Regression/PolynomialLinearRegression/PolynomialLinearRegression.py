# Priyanka Kasture | pkasture2010@gmail.com
# Polynomial Regression on 'Position_Salaries' Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # 1:2 because we always want X to be a matrix (feature matrix, not a vector)
y = dataset.iloc[:, 2].values

# Won't be splitting the data into train and test set because the dataset is a small one (10 samples)
"""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)"""

# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# Polynomial Linear Regression Model
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4) # Transform X into X_Poly which contains X, X squared, X cubed, etc
X_Poly = polyreg.fit_transform(X)
regressor1 = LinearRegression() # Fit X_Poly in our Linear Regression model
regressor1.fit(X_Poly,y)

# Note : Greater the degree, greater the accuracy

# Visualizing Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor1.predict(polyreg.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor1.predict(polyreg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
