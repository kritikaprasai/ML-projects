# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:16:03 2018

@author: anu
"""
##calculates eucledian distance metrics to classify the tumor cells
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd
df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
X= np.array(df.drop(['class'],1))
df.shape
Y=np.array(df['class'])

##training here
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2,random_state=42)
clf= neighbors.KNeighborsClassifier(metric="euclidean" ,n_neighbors=3) 
clf.fit(X_train,Y_train)
Y_pred= clf.predict(X_test)
accuracy=clf.score(Y_test,Y_pred)
print("The accuracy result of using KNN algorithm is: ",accuracy)
example_measures= np.array([[9,9,8,8,7,10,9,7,1]])
example_measures= example_measures.reshape(-1,1)              #not accuracy but confidence
prediction= clf.predict(example_measures)
print("The class of cancer is",prediction)
if prediction== 4:
 print ("You have Malignant class of tumour")
else:
    print ("You have Benign class of tumour")
 ##confusion matrix calculation   
    from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

                  