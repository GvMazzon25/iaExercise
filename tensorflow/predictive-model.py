from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

dataset = fetch_california_housing()

print(dataset['target'][0])

X = dataset['data']
y = dataset['target']

model = LinearRegression()
model.fit(X, y)

p = model.predict(X)

mae = mean_absolute_error(y, p)
print('MAE: ', mae)
print('mean y: ', np.mean(y))


