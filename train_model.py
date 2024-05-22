import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def estimate_price(mileage, mileage_normalized, theta0, theta1):
    '''Function to estimate the price for a given mileage'''

    estimated_price = theta0 + (theta1 * mileage_normalized)
    print(f"EstimatedPrice for {mileage} km: {estimated_price} $")
    return estimated_price


def gradient_descent(x, y, theta0, theta1, learning_rate, iterations):
    '''Gradient descent algorithm to update theta0 and theta1'''

    for _ in range(iterations):
        predictions = theta0 + theta1 * x
        error = predictions - y

        theta1 -= learning_rate * (1 / len(x)) * np.dot(x.T, error)
        theta0 -= learning_rate * (1 / len(x)) * np.sum(error)
    
    return theta0, theta1


def plot_scatter_and_regression(x, y, y_pred):
    '''Scatter plot draw function'''

    # Create a scatter plot for the data points
    plt.scatter(x, y, color='blue', label='Data points')

    # Plot the linear regression line
    plt.plot(x, y_pred, color='red', linewidth=1, label='Linear regression')

    # Add titles and axis labels
    plt.title('Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()

    # Display the plot
    plt.show()


def error_f(string: str):
    '''Error function'''

    print(f"\033[91m{string}\033[0m")
    exit(1)


def main() -> int:
    '''Main function'''

    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        error_f("usage: python3 estimated_price.py mileage")

    # Get the mileage from command-line argument
    mileage = int(sys.argv[1])

    # Read data from the CSV file
    data = pd.read_csv("data.csv")

    # Check if the data is empty
    if data.empty:
        error_f("error: data.csv is empty")

    # Extract 'km' and 'price' columns and convert to NumPy arrays
    x = data['km'].values.reshape(-1, 1)
    y = data['price'].values.reshape(-1, 1)

    # Check for NaN or Inf values in x and y
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        error_f("error: x or y contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        error_f("error: x or y contains Inf values.")

    # Calculate average values for 'km' and 'price'
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    x_max = np.max(x)
    x_min = np.min(x)

    x_normalized = (x - x_min) / (x_max - x_min)
    mileage_normalized = (mileage - x_min) / (x_max - x_min)

    # Initialize theta1 and theta0 using linear regression equations
    theta1 = np.sum((x_normalized - np.mean(x_normalized)) * (y - np.mean(y))) / np.sum((x_normalized - np.mean(x_normalized)) ** 2)
    theta0 = np.mean(y) - (theta1 * np.mean(x_normalized))

    iterations = 10000   # Number of iterations for gradient descent
    learning_rate = 0.1  # Learning rate for gradient descent

    # Gradient descent to optimize theta0 and theta1
    theta0, theta1 = gradient_descent(x_normalized, y, theta0, theta1, learning_rate, iterations)

    # Calculate predicted values with the model for all 'x' values
    y_pred = estimate_price(mileage, x_normalized, theta0, theta1)

    # Calculate R-squared (coefficient of determination) with the model
    ss_residual = np.sum((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    print(f"R-squared (accuracy): {r_squared * 100} %")

    # Plot the linear regression line
    plot_scatter_and_regression(x, y, y_pred)

    return 0


if __name__ == '__main__':
    main()
