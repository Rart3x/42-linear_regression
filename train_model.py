import pandas, sys
from estimated_price import call, ret_estimate_price

mileage = int(sys.argv[1])
data = pandas.read_csv("data.csv")

x = data['km'].values.reshape(-1,1)
y = data['price'].values.reshape(-1,1)

learning_rate = 0.1

tmp0 = 0
tmp1 = 0

for i in range(len(x)):
    tmp0 += (ret_estimate_price(x[i][0]) - y[i][0])
    tmp1 += ((ret_estimate_price(x[i][0]) - y[i][0]) * x[i][0])

mean_tmp0 = tmp0 / len(x)
mean_tmp1 = tmp1 / len(x)

theta0 = mean_tmp0 * learning_rate
theta1 = mean_tmp1 * learning_rate