import os
import numpy as ny
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from feature_extraction import extractor

Feature_List_Filepath = os.curdir + "\\data\\feature_list.csv"


df = pd.read_csv(Feature_List_Filepath)
data = df
X = data.iloc[:,1:].values
Y = data.iloc[:,0:1].values
print(X)
sc = StandardScaler()
X = sc.fit_transform(X)
logreg = LogisticRegression(C=1e-6, multi_class='ovr', penalty='l2', random_state=0)
logreg.fit(X,Y.ravel())
Y_predict = logreg.predict(X)
cm = confusion_matrix(Y, Y_predict)
print(cm)
print(metrics.accuracy_score(Y,Y_predict))


'''clf = SVC(kernel = 'rbf') 
Y_t=ny.transpose(Y)
clf.fit(X, Y.ravel()) 
Y_ppredict = clf.predict(X)
cmm = confusion_matrix(Y, Y_ppredict)
print(cmm)
print(metrics.accuracy_score(Y,Y_ppredict))'''

path1 = os.curdir + "\\data\\test_data.csv"
path2 = os.curdir + "\\data\\test_list.csv"

extractor(path1,path2)

dff = pd.read_csv(path2)
X_test = dff
print(X_test)
ans = logreg.predict(X_test)
print(ans)