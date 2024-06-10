import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def calculate_r2(y_true, y_pred):
    '''Calculate R^2'''
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate R^2
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) **2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2


def create_theta_csv(t0, t1):
    '''Create theta CSV method'''

    with open("theta.csv", "w") as file:
        file.write("theta0, theta1\n")
        file.write(f"{t0},{t1}")


def error_f(string: str):
    '''Error method'''

    print(f"\033[91m{string}\033[0m")
    exit(1)


def gradient_descent(t0, t1, data, L):
    '''Gradient descent method'''

    x = data['km_n']
    y = data['price_n']

    predictions = t0 + t1 * x
    error = predictions - y

    t1 -= L * (1 / len(x)) * np.dot(x.T, error)
    t0 -= L * (1 / len(x)) * np.sum(error)

    return t0, t1


def plot_scatter_and_regression(x, y, y_pred):
    '''Scatter plot draw method'''

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
    '''Main method'''

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

    #Normalize x and y columns
    x_normalized = (x - np.min(x)) / (np.max(x)- np.min(x))
    data["km_n"] = x_normalized
    y_normalized = (y - np.min(y)) / (np.max(y)- np.min(y))
    data["price_n"] = y_normalized

    t0, t1 = 0, 0
    learning_rate = 0.1
    iterations = 1000

    for i in range(iterations):
        t0, t1 = gradient_descent(t0, t1, data, learning_rate)

    # Calculate R^2, MSE, and MAE
    y_true_normalized = data["price_n"]
    y_pred_normalized = (t0 + t1 * data["km_n"])

    # Calcul du R^2 avec les données normalisées
    r2_normalized = calculate_r2(y_true_normalized, y_pred_normalized)
    r2_percentage = r2_normalized * 100
    print(f'R^2 (accuracy) : {r2_percentage:.2f}%')

    #Denormalized thetas
    t0 = t0 * (np.max(y) - np.min(y)) + np.min(y)
    t1 = t1 * (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))

    # Create theta CSV
    create_theta_csv(t0, t1)
    
    #Stock predictions vlues
    y_pred = t0 + (t1 * x)



    plot_scatter_and_regression(x, y, y_pred)

    return 0


if __name__ == '__main__':
    main()
