import pandas, sys

def calculate_theta0(x_avg, y_avg, theta1):
    # EstimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
    return y_avg - theta1 * x_avg

def calculate_theta1(x, y):
    x_avg = sum(x) / len(x)
    y_avg = sum(y) / len(y)

    theta0 = 0
    theta1 = 0

    for i in range(len(x)):
        theta0 += (x[i] - x_avg) * (y[i] - y_avg)
        theta1 += (x[i] - x_avg) ** 2

    return theta0 / theta1

def estimate_price(mileage):
    print("EstimatedPrice for " + str(mileage) + "km : " + str(theta0 + (theta1 * mileage)))
    return theta0 + (theta1 * mileage)

def print_mileage(x):
    for i in range(len(x)):
        print(str(x[i][0]) + " km")

def print_prices(y):
    for i in range(len(y)):
        print(str(y[i][0]) + " €")

if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("\033[91musage: python3 estimated_price.py mileage\033[0m")
    exit()

mileage = int(sys.argv[1])
data = pandas.read_csv("data.csv")

x = data['km'].values.reshape(-1,1)
y = data['price'].values.reshape(-1,1)

x_avg = sum(x) / len(x)
y_avg = sum(y) / len(y)

theta0 = 0 # Coefficient d'interception (θ0)
theta1 = 0 # Coefficient de pente (θ1)

theta1 = calculate_theta1(x, y)
theta0 = calculate_theta0(x_avg, y_avg, theta1)

estimate_price(mileage)