import pandas as pd
import numpy as np
import streamlit as stl
import pickle

# Open the pickle file which you created from Simple_LR_Model file

with open("Simple_LR_model.pkl",'rb')as file:
    model=pickle.load(file)
    
stl.title("Employee salary prediction based on their experience")


YearsExperience=stl.number_input("YearsExperience",min_value=1,max_value=30)
#Salary=stl.number_input("Salary")

# Prediction Salary
if stl.button("Predict"):
    
    input_data = np.array([[YearsExperience]])
    prediction = model.predict(input_data)
    stl.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    
    