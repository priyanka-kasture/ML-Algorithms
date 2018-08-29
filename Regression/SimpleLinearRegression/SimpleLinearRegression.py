
# Importing the libraries
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values # All rows, all columns except last
y = dataset.iloc[:, 1].values # All rows, first column (index starts at 0) column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Line fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_train)

# Plotting the plot on training data
plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Training Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Plotting the plot on testing data
plt.scatter(X_test,y_test,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Test Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
