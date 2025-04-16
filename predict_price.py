import pandas as pd
import sys


def estimate_price(mileage, theta0, theta1):
    """Function to estimate the price for a given mileage"""
    estimated_price = theta0 + (theta1 * mileage)
    estimated_price = round(estimated_price, 2)
    print(
        f"EstimatedPrice for a car with "
        f"{mileage}"
        f" km mileage: {estimated_price} $"
    )


def error_f(string: str):
    """Error function"""
    print(f"\033[91m{string}\033[0m")
    exit(1)


def error_f_without_exit(string: str):
    """Error function without exit"""
    print(f"\033[91m{string}\033[0m")


def get_thetas():
    """Get thetas method"""
    theta0, theta1 = 0, 0

    try:
        thetas = pd.read_csv("theta.csv")

        if "theta0" in thetas.columns and "theta1" in thetas.columns:
            theta0 = thetas.iloc[0]["theta0"]
            theta1 = thetas.iloc[0]["theta1"]
        else:
            raise KeyError("Any theta columns in theta.csv")

    except FileNotFoundError:
        print(
            "\033[94mINFO: Theta file not found,"
            "by default they will be set to 0\033[0m"
        )
    except KeyError as e:
        error_f_without_exit(f"Error: {str(e)}")
    except Exception as e:
        error_f_without_exit(f"Error: {e}")

    return theta0, theta1


def main() -> int:
    """Main function"""

    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        error_f("USAGE : python3 predict_price.py mileage")

    # Get thetas from the CSV file
    theta0, theta1 = get_thetas()

    # Get the mileage from command-line argument
    mileage = int(sys.argv[1])

    # Initialize the data DataFrame
    data = pd.DataFrame()

    # Read data from the CSV file
    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        error_f("ERROR: cannot access to datas CSV")
        raise
    except pd.errors.EmptyDataError:
        error_f("ERROR: datas CSV file is empty")
        raise
    except Exception as e:
        error_f("ERROR: cannot access to datas CSV")
        raise

    # Check if the data is empty
    if data.empty:
        error_f("ERROR: data.csv is empty")

    # Extract 'km' and 'price' columns and convert to NumPy arrays
    x = data["km"].values.reshape(-1, 1)
    y = data["price"].values.reshape(-1, 1)

    if len(x) == 0 or len(y) == 0:
        error_f("ERROR: km and price columns can't be empty")

    estimate_price(mileage, theta0, theta1)

    return 0


if __name__ == "__main__":
    main()
