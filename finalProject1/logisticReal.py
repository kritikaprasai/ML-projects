##all require modules is imported here....................................
import numpy as np
import pandas as pd 

##geting dataset here and arranging them accordingly...................
df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
X= np.array(df.drop(['class'],1))
Y=np.array(df['class'])

##scaling the features...
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)
##tarining the data sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 
##20% data used for training

##accuracy test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model= LogisticRegression(penalty='l2',C=1)
model.fit(X_train,Y_train)
Y_pred= model.predict(X_test)
LogisticRegression(C=1,class_weight=None,dual=False,fit_intercept=True,
intercept_scaling=1,penalty='l2',random_state=42,tol=0.0001)
accuracy= accuracy_score(Y_test,Y_pred) ##accuracy
print ("the accuracy obtained by using logistic regression is: ",accuracy)

##prediicts the class of tumor based on the example measures provided

example_measures= np.array([[9,9,8,8,7,10,9,7,1]])
example_measures= example_measures.reshape(1,-1)              #not accuracy but confidence
prediction= model.predict(example_measures)
print(prediction)
print("The class of cancer is",prediction)
if prediction== 4:
 print ("You have Malignant class of tumour")
else:
    print ("You have Benign class of tumour")
    
##confusion matrix calculation   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

                  
