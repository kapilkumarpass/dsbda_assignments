import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston = load_boston()
print(boston.data.shape)
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

print(data.head(10))
# Adding 'Price' (target) column to the data
print(boston.target.shape)
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

print(data.head(10))
# Adding 'Price' (target) column to the data
print(boston.target.shape)
data['Price'] = boston.target
print(data.head())
print(data.describe())
print(data.info())
# Input Data
x = boston.data

# Output Data
y = boston.target

# splitting data to training and testing dataset.

# from sklearn.cross_validation import train_test_split
# the submodule cross_validation is renamed and reprecated to model_selection
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,
                                                random_state=0)

print("xtrain shape : ", xtrain.shape)
print("xtest shape : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape : ", ytest.shape)

# Fitting Multi Linear regression model to training model

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# predicting the test set results
y_pred = regressor.predict(xtest)

# Plotting Scatter graph to show the prediction
# results - 'ytrue' value vs 'y_pred' value
plt.scatter(ytest, y_pred, c='green')
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value : Linear Regression")
plt.show()

# Results of Linear Regression.


mse = mean_squared_error(ytest, y_pred)
print("Mean Square Error : ", mse)
