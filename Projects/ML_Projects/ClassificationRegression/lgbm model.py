# import all library
import pandas as pd
import numpy as np
 
# import dataset
data=pd.read_csv(r'C:\Users\satya\Downloads\Breast_cancer_data.csv')

data.head(2)

X=data.iloc[:,0:-1].values
y=data.iloc[:,-1].values

# splting dataset into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


import lightgbm as lgbm
classifier=lgbm.LGBMClassifier()
classifier.fit(X_train, y_train)


# Model Prediction

y_pred=classifier.predict(X_test)

#Find accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print('ac',ac)

# view confusion metix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print('cm',cm)


 #find bias and variance score
 
bias=classifier.score(X_train,y_train)
print('Bias',bias)
 
 var=classifier.score(X_test, y_test)
 print('Var',var)
 
 # for comparing now train set
 
 y_pred_train=classifier.predict(X_train)
 
 ac1=accuracy_score(y_train, y_pred_train)
 print('ac_train',ac1)
 
 #classification report
 from sklearn.metrics import classification_report
 clr=classification_report(y_test, y_pred)
 print('Clr',clr)
 