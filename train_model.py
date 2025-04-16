import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.widgets import Slider


def accuracy(data, t0, t1):
    """Calculate R^2, MSE, and MAE"""
    y_true_normalized = data["price_n"]
    y_pred_normalized = t0 + t1 * data["km_n"]

    r2_normalized = calculate_r2(y_true_normalized, y_pred_normalized)
    r2_percentage = r2_normalized * 100
    print(f"R^2 (accuracy) : {r2_percentage:.2f}%")


def calculate_r2(y_true, y_pred):
    """Calculate R^2"""
    # Calculate residuals
    residuals = y_true - y_pred

    # Calculate R^2
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2


def create_theta_csv(t0, t1):
    """Create theta CSV method"""
    with open("theta.csv", "w") as file:
        file.write("theta0,theta1\n")
        file.write(f"{t0},{t1}")


def error_f(string: str):
    """Error method"""
    print(f"\033[91m{string}\033[0m")
    exit(1)


def gradient_descent(t0, t1, data, L):
    """Gradient descent method"""
    x = data["km_n"]
    y = data["price_n"]

    predictions = t0 + t1 * x
    error = predictions - y

    t1 -= L * (1 / len(x)) * np.dot(x.T, error)
    t0 -= L * (1 / len(x)) * np.sum(error)

    return t0, t1


def plot_scatter_and_regression(x, y, predictions, data, thetas0, thetas1):
    """Plot scatter and regression"""
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Scatter all datas values from CSV
    ax.scatter(x, y, color="red", label="Data points")

    ax.plot(x, predictions[-1], label=f"Iteration {len(predictions)}")
    ax.legend()
    accuracy(data, thetas0[-1], thetas1[-1])

    # Slider creation
    color = "lightgoldenrodyellow"
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=color)
    slider = Slider(
        ax_slider,
        "Iteration",
        0,
        len(predictions) - 1,
        valinit=len(predictions) - 1,
        valstep=1,
    )

    def update(val):
        """Update cursor method"""

        iteration_index = int(slider.val)
        ax.clear()

        ax.scatter(x, y, color="red", label="Data points")
        ax.plot(
            x,
            predictions[iteration_index],
            label=f"Iteration {iteration_index + 1}",
        )
        ax.legend()

        fig.canvas.draw_idle()

        accuracy(data, thetas0[iteration_index], thetas1[iteration_index])

    # Call on changed method from Slider to update the PLOT
    slider.on_changed(update)

    plt.show()


def check_data_validity(x, y):
    """
    Check if x and y are valid NumPy arrays
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        error_f("ERROR: x or y is not a NumPy array")

    if x.size == 0 or y.size == 0:
        error_f("ERROR: x or y is empty")

    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(
            y.dtype, np.number
    ):
        error_f("ERROR: x or y contains non-numeric values")

    if np.isnan(x).any() or np.isnan(y).any():
        error_f("ERROR: x or y contains NaN values")


def main() -> int:
    """Main method"""
    # Read data from the CSV file
    data = pd.DataFrame()

    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        error_f("ERROR: cannot access to CSV file")
        raise
    except pd.errors.EmptyDataError:
        error_f("ERROR: CSV file is empty")
        raise
    except Exception as e:
        error_f(f"ERROR: {e}")
        raise

    # Extract 'km' and 'price' columns and convert to NumPy arrays
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    check_data_validity(x, y)

    # Check for NaN or Inf values in x and y
    if len(x) == 0 or len(y) == 0:
        error_f("ERROR: km and price columns can't be empty")

    # Normalize x and y columns
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    data["km_n"] = x_normalized
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
    data["price_n"] = y_normalized

    learning_rate = 0.1
    iterations = 1000
    predictions = []

    t0, t1 = 0, 0
    t0_denormalized, t1_denormalized = 0, 0
    thetas0, thetas1 = [], []

    for i in range(iterations):
        t0, t1 = gradient_descent(t0, t1, data, learning_rate)

        # Denormalized thetas
        t0_denormalized = t0 * (np.max(y) - np.min(y)) + np.min(y)
        t1_denormalized = (
                t1 * (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))
        )

        y_pred = t0_denormalized + (t1_denormalized * x)

        predictions.append(y_pred)
        thetas0.append(t0)
        thetas1.append(t1)

    # Create theta CSV
    create_theta_csv(t0_denormalized, t1_denormalized)

    # Stock predictions views
    plot_scatter_and_regression(x, y, predictions, data, thetas0, thetas1)

    return 0


if __name__ == "__main__":
    main()
