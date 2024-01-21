import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Function to estimate the price for a given mileage
def estimate_price(mileage, mileage_normalized, theta0, theta1):
    estimated_price = theta0 + (theta1 * mileage_normalized)
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

def plot_scatter_and_regression(x, y, y_pred):
    # Create a scatter plot for the data points
    plt.scatter(x, y, color='blue', label='Data points')

    # Plot the linear regression line
    plt.plot(x, y_pred, color='red', linewidth=2, label='Linear regression')

    # Add titles and axis labels
    plt.title('Linear Regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()

    # Display the plot
    plt.show()

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

# Check for NaN or Inf values in x and y
if np.any(np.isnan(x)) or np.any(np.isnan(y)):
    print("\033[91merror: x or y contains NaN values.")
if np.any(np.isinf(x)) or np.any(np.isinf(y)):
    print("\033[91merror: x or y contains Inf values.")

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

iterations = 100     # Number of iterations for gradient descent
learning_rate = 0.1  # Learning rate for gradient descent

# Gradient descent to optimize theta0 and theta1
theta0, theta1 = gradient_descent(x_normalized, y, theta0, theta1, learning_rate, iterations)

# Calculate predicted values with the model for all 'x' values
y_pred = estimate_price(mileage, x_normalized, theta0, theta1)

# Calculate R-squared
ss_residual = np.sum((y - y_pred)**2)
ss_total = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_residual / ss_total)
print(f"Accuracy: {r_squared * 100} %")

# Plot the linear regression line
plot_scatter_and_regression(x, y, y_pred)