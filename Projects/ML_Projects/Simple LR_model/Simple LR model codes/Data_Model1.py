import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd	

# IMPORT THE DATASET

#dataset = pd.read_csv(r"‪C:\Users\satya\Downloads\Data.csv")

# Here strip is use for remove hidden charecter for Path access
clean_path = r"C:\Users\satya\Downloads\Data.csv".strip()
handle = open(clean_path, "r")
data=pd.read_csv(handle)

#independent variable
x=data.iloc[:,:-1].values
#depended variable
y=data.iloc[:,3]

#sklearn fill numerical missing value

from sklearn.impute import SimpleImputer
imputer =SimpleImputer()
imputer=imputer.fit(x[:,1:3])

x[:,1:3]=imputer.transform(x[:,1:3])

# Imput catgorical values for depended

from sklearn.preprocessing import LabelEncoder

LabelEncoder_x=LabelEncoder()
LabelEncoder_x.fit_transform(x[:,0])
x[:,0]=LabelEncoder_x.fit_transform(x[:,0])

#impute categorical data for dependent

labalencoder_y=LabelEncoder()
y=labalencoder_y.fit_transform(y)

# Split the data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


data.mean()
data.median()
data['Salary'].mode()

data.var()

data.std()
data["Salary"].std()
data.corr()

data["Salary"].corr(data['YearsExperience'])

bias=regressor.score(x_train,y_train)
variance=regressor.score(x_test,y_test)

print(bias)
print(variance)


y_mean=np.mean(y)

ssr=np.sum((y_pred-y_mean)**2)
print(ssr)

y=y[0:6]
sse=np.sum((y-y_pred)**2)
print(sse)

mean_total=np.mean(data.values)

sst=np.sum((data.values-mean_total)**2)
print(sst)













