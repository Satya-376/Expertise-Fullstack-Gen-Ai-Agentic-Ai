import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as stl

# load files
data1=pd.read_csv(r'C:\Users\satya\Downloads\Salary_Data.csv')
data1
                  
path=r'C:\Users\satya\Downloads\Salary_Data.csv'
path=path.encode('ascii','ignore').decode()
handle= open(path,'r')
data=pd.read_csv(handle)

# Split the data into independent and dependent variables
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values 

# Split the dataset into training and testing sets (80-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set
y_pred = regressor.predict(x_test)

# Visualize the training set
plt.scatter(x_train, y_train, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")


import pickle
# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

# Open the pickel file
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

stl.title("Employee Attrition Prediction")

# Input fields
YearsExperience=stl.number_input("YearsExperience",min_value=1,max_value=30)
#Salary=stl.number_input("Salary")

# Prediction
if stl.button("Predict"):
    input_data = [[Salary]]
    prediction = model.predict(input_data)
    stl.success(f"Prediction: {prediction[1]}")


