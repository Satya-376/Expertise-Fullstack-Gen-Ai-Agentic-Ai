
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
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

  

#use Bernuali bayes algorithum
#from sklearn.naive_bayes import BernoulliNB
#classifier=BernoulliNB()
#classifier.fit(X_train, y_train)

#Use multinominal bays algorithum
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)

#Use Gaussian bays algorithum
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)


classifier.get_params()  

# Predict the model on X_test
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr) 

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('ac',ac) 


bias = classifier.score(X_train, y_train)
print(bias)

var = classifier.score(X_test, y_test)
print(var) 



















