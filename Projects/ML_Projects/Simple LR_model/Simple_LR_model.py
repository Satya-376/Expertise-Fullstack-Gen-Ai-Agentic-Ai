# Import all Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# import data from source
data=pd.read_csv(r"C:\Users\satya\Downloads\Salary_Data.csv")

# find independent and depended variable
x=data.iloc[:,:-1].values # Independed variabe
y=data.iloc[:,1].values   #depended variable

# split data into traint set and test set
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#Train the model
LR_model=LinearRegression()
LR_model.fit(x_train,y_train)

# predict  the Test set

y_pred=LR_model.predict(x_test)

# Visualize the training set for check Bais
plt.scatter(x_train, y_train, color='red') 
plt.plot(x_train, LR_model.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set for check variance
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, LR_model.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Calculates stats fuctions
data['Salary'].mean()
data['Salary'].median()
data['Salary'].mode()

data.describe()
data.std()
data.var()
data.corr()


# find SSR
y_mean=np.mean(y)
ssr=np.sum((y_pred-y_mean)**2)
print(ssr)

# find sse
y=y[0:6]
sse=np.sum((y-y_pred)**2)
print(sse)

#find sst
mean_total=np.mean(data.values)
sst=np.mean((mean_total-data.values)**2)
print(sst)

# find r2(r_square)
r2=1-ssr/sst
print(r2)

#pridict the salary for 12 and 15 years experience for trained model
y_12=LR_model.predict([[12]])
y_15=LR_model.predict([[15]])

print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 15 years of experience: ${y_15[0]:,.2f}")

#Check Model Performance
bais=LR_model.score(x_train, y_train)
variance=LR_model.score(x_test, y_test)

print(f"Traning(Bais) score r2:{bais:.2f}")
print(f"Testing variance score r2:{variance:.2f}")

# find the cofiecient and intercept in the Linear model
print(f"cofficient:{LR_model.coef_}")
print(f"intercept:{LR_model.intercept_}")

#Save to the Trained model into the disk
Filename='Simple_LR_model.pkl'
with open(Filename,'wb')as file:
    pickle.dump(LR_model,file)
    
print("Simple_LR_model.pkl name file has been successfully saved")
    













