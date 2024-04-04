# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Perch data
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5, 44.0] )

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0] )

# LR
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

reg = LinearRegression()
reg.fit(train_input, train_target)
print(reg.coef_,reg.intercept_)
print(reg.score(train_input, train_target))
print(reg.score(test_input, test_target))
print(reg.predict([[50]]))

# visualization
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*reg.coef_+reg.intercept_,50*reg.coef_+reg.intercept_])
plt.scatter(50, 1241.84, marker= '^')
plt.show()

# MSE
train_pred = reg.predict(train_input)
mse_train = mean_squared_error(train_target, train_pred)
print(mse_train)

test_pred = reg.predict(test_input)
mse_test = mean_squared_error(test_target, test_pred)
print(mse_test)


# practice
import seaborn as sns
iris = sns.load_dataset('iris')

input1 = pd.DataFrame(iris['petal_length'])
target1 = pd.DataFrame(iris['sepal_length'])

input1_train, input1_test, target1_train, target1_test = train_test_split(input1, target1)
print(target1_test)
lr = LinearRegression()
lr.fit(input1_train, target1_train)
print(lr.coef_, lr.intercept_)

iris_pred = pd.DataFrame(lr.predict(input1_test))
iris_pred.head()
target1_test.head()

print(lr.score(input1_train, target1_train))
print(lr.score(input1_test, target1_test))

plt.scatter(input1_train, target1_train)
plt.plot([0, 7], [lr.intercept_, lr.intercept_ + 7 * lr.coef_[0]], color='red')
plt.show()