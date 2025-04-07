# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start

STEP 2. Data preparation

STEP 3. Hypothesis Definition

STEP 4. Cost Function

STEP 5. Parameter Update Rule

STEP 6. Iterative Training

STEP 7. Model evaluation

STEP 8. End


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ibrahim fedah s
RegisterNumber:  212223240056
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Fetch the data
data = fetch_california_housing()
x = data.data[:, :3]
y = np.column_stack((data.target, data.data[:, 6]))

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

# Standardize target variables
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Create and train the model
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)

# Make predictions
y_pred = multi_output_sgd.predict(x_test)

# Inverse transform the predictions and the true values
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Print the predictions
print("Predictions:\n", y_pred)
print("True values:\n", y_test)

# Optional: Print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```
## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
![image](https://github.com/user-attachments/assets/b011f18e-b51f-46b1-b7e3-a58662bc965f)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
