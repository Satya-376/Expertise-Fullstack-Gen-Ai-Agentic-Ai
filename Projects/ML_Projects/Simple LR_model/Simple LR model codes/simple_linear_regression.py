
# Import necessary libraries
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


# Here decode is use for remove hidden charecter for Path access
#data = pd.read_csv(r"‪C:\Users\satya\Downloads\Data.csv")

clean_path = r"‪‪‪C:\Users\satya\Downloads\Salary_Data.csv"
clean_path = clean_path.encode("ascii", "ignore").decode()
handle = open(clean_path,"r")

data=pd.read_csv(handle)

# Make independent and Depended variable. 
x=data.iloc[:,:-1]  # Independed variable
y=data.iloc[:,-1]   # Depended variable

# split data in train and test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# fit the leanar regression model to the training set


regressor=LinearRegression()
regressor.fit(x_train, y_train)

# predicting the result set for the test set
y_pred=regressor.predict(x_test)

 data.mean()
 data['Salary'].mean()

data.median()

data['Salary'].mode()

data.describe()
data.var()
data.std()
data.corr()

# SSR

y_mean=np.mean(y)

ssr=np.sum((y_pred-y_mean)**2)
print(ssr)

#sse
y=y[0:6]
sse=np.sum((y-y_pred)**2)
print(sse)

#sst
mean_total=np.mean(data.values)

sst=np.mean((data.values-mean_total)**2)
print(sst)

#r2
r_square=1-ssr/sst
print(r_square)

# Visualizing the training result set
plt.scatter(x_train, y_train, color = 'red')  # Real salary data (training)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test result set
plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#compare predicted and actula salary from the set

comparison = pd.DataFrame({'Actual':y_test,'Predict':'y_test'})
print(comparison) 

# Statistics for machine learning 


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

#output the coefficients in the Linear Model
print(f"Intercept:{regressor.intercept_}")
print(f"Coefficient:{regressor.coef_}")




# Save the trained model to disk
filename = 'Simple_LR_model1.pkl'
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












