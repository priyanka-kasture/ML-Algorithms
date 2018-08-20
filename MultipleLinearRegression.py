import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,metrics,linear_model

# Boston-House-Pricing Dataset
boston_pricing = datasets.load_boston(return_X_y=False)
X = boston_pricing.data
y = boston_pricing.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)

regression = linear_model.LinearRegression()
regression.fit(X_train,y_train)

print('Multiple Linear Regression Coefficients: ',regression.coef_)
print('Variance : {}',format(regression.score(X_test,y_test)))

plt.style.use('fivethirtyeight')
plt.scatter(regression.predict(X_train),regression.predict(X_train)-y_train,color="red",s=8,label='Training Data')
plt.scatter(regression.predict(X_test),regression.predict(X_test)-y_test,color="blue",s=8,label='Testing Data')
plt.hlines(y=0,xmin=0,xmax=50,linewidth=1.5)
plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.show()
