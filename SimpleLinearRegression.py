import numpy as np      # Numpy is a fundamental package for scientific computing.
import matplotlib.pyplot as plt     # Matplotlib is a 2D plotting library.
 
def estimate_coefficients(x, y):
    
    n = np.size(x)      # Number of observations/points. 10 in our case, since 'x' is an array of 10 elements.
    
    meanx, meany = np.mean(x), np.mean(y)       # Mean ((first element + last element)/2) of x and y vector/array.
 
    ssxy = np.sum(y*x - n*meany*meanx)         # Sum of cross deviations of x and y.
    ssxx = np.sum(x*x - n*meanx*meanx)         # Sum of squared deviations of x.
 
    b1 = ssxy / ssxx             # Calculating regression coefficients b0 (y-intercept) and b1 (slope).
    b0 = meany - b1*meanx
 
    return(b0, b1)
 
def plot_regression_line(x, y, b):

    plt.scatter(x, y, color = "m", marker = "o", s = 30)        # Plotting the actual points as scatter plot.
 
    ypred = b[0] + b[1]*x          # Predicted regression line or response vector.
 
    plt.plot(x, ypred, color = "g")        # Plotting the response vector.
 
    plt.xlabel('x - axis')         # Specifying axes labels.
    plt.ylabel('y - axis')
 
    plt.show()      # Function that displays the plotted graph.
 
def main():
    
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])        # Explicitly specified observations.
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
 
    b = estimate_coefficients(x, y)         # Call to function that returns regression coefficients.
    print("\nEstimated regression coefficients :\nb0 = {}  \nb1 = {}".format(b[0], b[1]))
 
    plot_regression_line(x, y, b)       # Call to function that plots the response vector/ regression line.
 
if __name__ == "__main__":
    main()
