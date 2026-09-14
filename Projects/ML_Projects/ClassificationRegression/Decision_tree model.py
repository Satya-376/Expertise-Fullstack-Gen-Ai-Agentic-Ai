
# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\satya\Gen Ai or Agentic Ai\Projects\Classification Regression\logit classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20,
                                                    random_state=0)
# standerd Scaling for scale labaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

  
#from sklearn.tree import DecisionTreeClassifier
#dtc=DecisionTreeClassifier()
#dtc=DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=None,min_samples_leaf=5,max_features=3)
#dtc.fit(X_train,y_train)


#apply decision tree with hyper parameter or without parameter
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=None,min_samples_leaf=8)
dtc.fit(X_train,y_train)

#classifier.get_params()  

# Predict the model on X_test
y_pred = dtc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('CM',cm)

#from sklearn.metrics import classification_report
#cr = classification_report(y_test,y_pred)
#print(cr) 

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('ac:',ac) 

bias = dtc.score(X_train, y_train)
print('Bias:',bias)

var = dtc.score(X_test, y_test)
print('Variance:',var) 

# Apply Random forest tree
#from sklearn.ensemble import RandomForestClassifier
#rfc=RandomForestClassifier(n_estimators=30,random_state=0)
#rfc.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print('cm',cm)













