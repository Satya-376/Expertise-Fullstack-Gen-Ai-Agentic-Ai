# Logistic regression
# import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
data1=pd.read_csv(r'C:\Users\satya\Downloads\29th, 31st - Logistic Regression\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Social_Network_Ads.csv')
data1

# indentify depended variable and independed variable

x=data1.iloc[:,[2,3]].values
y=data1.iloc[:,-1].values

# spiliting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# Feature scaling 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Traning the logistic regression model on the trainig set

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train, y_train)

#Predictive the test set result
y_predict=classifier.predict(x_test)

#we build our logistic model and fit it to the training set & we predict our test set result 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)

# from confusion metrix calculate accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_predict)
ac

# This is to get classification report
from sklearn.metrics import classification_report
cr=classification_report(y_test, y_predict)
cr

# find the classifier bais and variance score

bias=classifier.score(x_train, y_train)
bias

variance=classifier.score(x_test, y_test)
variance



#-----------------FUTURE PREDICTION ------------

dataset1 = pd.read_csv(r"C:\Users\satya\Downloads\29th, 31st - Logistic Regression\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")

d2 = dataset1.copy() 

dataset1 = dataset1.iloc[:, [2, 3]].values 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1) 

y_pred1 = pd.DataFrame() 


d2 ['y_pred1'] = classifier.predict(M) 
 
d2.to_csv('final1.csv')

'''
# To get the path 
import os
os.getcwd()
'''

#********************************************************************************************************************

from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
print(auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
 

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show() 
#============



#Now we gonna see the visulization 

# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

