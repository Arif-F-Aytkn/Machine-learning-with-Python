# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
veriler = pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\veriler.csv")
print(veriler)

x = veriler.iloc[:,1:4].values 
y = veriler.iloc[:,4:].values 
print(y)

#Separating data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)
print(cm1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm2=confusion_matrix(y_test,y_pred)
print(cm2)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred= svc.predict(X_test)

cm3=confusion_matrix(y_test,y_pred)
print(cm3)

#NON LÝNEAR SVM
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred= svc.predict(X_test)

cm4=confusion_matrix(y_test,y_pred)
print(cm4)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

cm5=confusion_matrix(y_test,y_pred)
print(cm5)

#DECÝSÝON TREE 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm6=confusion_matrix(y_test,y_pred)
print(cm6)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)
y_pred= rfc.predict(X_test)

cm7 = confusion_matrix(y_test,y_pred)
print(cm7)

