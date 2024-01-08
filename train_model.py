import numpy as np
import pandas as pd
import sys

# Function to calculate the cost (avg squared error)
def calculate_cost(x, y, theta0, theta1):
    predictions = theta0 + theta1 * x
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * len(x))
    return cost

# Function to estimate the price for a given mileage
def estimate_price(mileage, theta0, theta1):
    estimated_price = theta0 + (theta1 * mileage)
    print(f"EstimatedPrice for {mileage} km: {estimated_price} $")
    return estimated_price

# Gradient descent algorithm to update theta0 and theta1
def gradient_descent(x, y, theta0, theta1, learning_rate, iterations):
    for _ in range(iterations):
        predictions = theta0 + theta1 * x
        error = predictions - y

        theta1 -= learning_rate * (1 / len(x)) * np.dot(x.T, error)
        theta0 -= learning_rate * (1 / len(x)) * np.sum(error)
    
    return theta0, theta1

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("\033[91musage: python3 estimated_price.py mileage\033[0m")
    exit()

# Get the mileage from command-line argument
mileage = int(sys.argv[1])

# Read data from the CSV file
data = pd.read_csv("data.csv")

# Check if the data is empty
if data.empty:
    print("\033[91merror: data.csv is empty\033[0m")
    exit()

# Extract 'km' and 'price' columns and convert to NumPy arrays
x = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

# Calculate average values for 'km' and 'price'
x_avg = np.mean(x)
y_avg = np.mean(y)

# Initialize theta1 and theta0 using linear regression equations
theta1 = np.sum((x - x_avg) * (y - y_avg)) / np.sum((x - x_avg) ** 2)
theta0 = y_avg - (theta1 * x_avg)

iterations = 100     # Number of iterations for gradient descent
learning_rate = 0.1  # Learning rate for gradient descent

# Perform gradient descent to optimize theta0 and theta1
theta0, theta1 = gradient_descent(x, y, theta0, theta1, learning_rate, iterations)

# Estimate the price for the provided mileage using the trained model
estimate_price(mileage, theta0, theta1)