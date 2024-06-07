import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def custom_train_test_split(x, y, test_size):
    '''Custom train_test_split method from sklearn lib'''

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    
    split_index = int((1 - test_size) * len(x))
    x_train, x_test = x[indices[:split_index]], x[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return x_train, x_test, y_train, y_test


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

    x_train, x_test, y_train, y_test = custom_train_test_split(x, y, test_size=1.0/3)

    return 0


if __name__ == '__main__':
    main()
