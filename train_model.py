import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def create_theta_csv(t0, t1):
    '''Create theta CSV method'''

    with open("theta.csv", "w") as file:
        file.write(f"{t0},{t1}")

def error_f(string: str):
    '''Error function'''

    print(f"\033[91m{string}\033[0m")
    exit(1)


def gradient_descent(t0, t1, data, L):
    '''Gradient descent method'''

    t0_gradient = 0
    t1_gradient = 0

    n = len(data)

    for i in range(n):
        x = data.iloc[i].km
        y = data.iloc[i].price

        t0_gradient += -(2 / n) * x * (y - (t0 * x + t1))
        t1_gradient += -(2 / n) * (y - (t0 * x + t1))

    new_t0 = t0 - t0_gradient * L
    new_t1 = t1 - t1_gradient * L

    return new_t0, new_t1


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


def main() -> int:
    '''Main function'''

    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        error_f("usage: python3 estimated_price.py mileage")

    # Get the mileage from command-line argument
    mileage = int(sys.argv[1])

    # Read data from the CSV file
    try:
        data = pd.read_csv("data.csv")
    except:
        error_f("error: cannot access to file")

    # Check if the data is empty
    if data.empty:
        error_f("error: data.csv is empty")

    # Extract 'km' and 'price' columns and convert to NumPy arrays
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    # Check for NaN or Inf values in x and y
    if len(x) == 0 or len(y) == 0:
        error_f("error: km and price columns can't be empty")

    if (np.isnan(x).any() or np.isnan(y).any()):
        error_f("error: invalid value in km or price column")

    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        error_f("error: x or y contains Inf values.")

    t0, t1 = 0, 0
    learning_rate = 0.001
    iterations = 100

    for i in range(iterations):
        t0, t1 = gradient_descent(t0, t1, data, learning_rate)

    # Create theta CSV
    create_theta_csv(t0, t1)

    y_pred = t0 * x + t1

    plot_scatter_and_regression(x, y, y_pred)

    return 0


if __name__ == '__main__':
    main()
