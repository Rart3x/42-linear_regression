import numpy as np
import pandas as pd
import sys


def calculate_cost(x, y, theta0, theta1):
    '''Function to calculate the cost (avg squared error)'''

    predictions = theta0 + theta1 * x
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * len(x))
    return cost


def estimate_price(mileage, theta0, theta1):
    '''Function to estimate the price for a given mileage'''

    estimated_price = theta0 + (theta1 * mileage)
    estimated_price = round(estimated_price, 2)
    print(f"EstimatedPrice for {mileage} km: {estimated_price} $")


def error_f(string: str):
    '''Error function'''

    print(f"\033[91m{string}\033[0m")
    exit(1)


def main() -> int:
    '''Main function'''

    theta0, theta1 = 0, 0

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

    # Calculate average values for 'km' and 'price'
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    # Initialize theta1 and theta0 using linear regression equations
    theta1 = np.sum((x - x_avg) * (y - y_avg)) / np.sum((x - x_avg) ** 2)
    theta0 = y_avg - (theta1 * x_avg)

    # Estimate the price for the provided mileage using the trained model
    estimate_price(mileage, theta0, theta1)

    return 0


if __name__ == '__main__':
    main()
