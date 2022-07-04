from sklearn.linear_model import LinearRegression
import numpy as np

apple = np.array([1, 3, 5, 7, 9])
n = len(apple)

model = LinearRegression().fit(np.arange(n).reshape((n,1)), apple)

print(model.predict([[5],[6]]))