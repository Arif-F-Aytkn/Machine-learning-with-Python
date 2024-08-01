# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\iris.xls")
print(data)

x = data.iloc[:,1:4].values 
y = data.iloc[:,4:].values 
print(y)

#Separating data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# 1. LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)
print("1. LOGISTIC REGRESSION")
print(cm1)

# 2. KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm2=confusion_matrix(y_test,y_pred)
print("KNN")
print(cm2)

# 3. SVC
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred= svc.predict(X_test)

cm3=confusion_matrix(y_test,y_pred)
print("SVC")
print(cm3)

# 4. NON LINEAR SVM
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred= svc.predict(X_test)

cm4=confusion_matrix(y_test,y_pred)
print("NON LINEAR SVM")
print(cm4)

# 5. NAIVE BAYES
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

cm5=confusion_matrix(y_test,y_pred)
print("NAIVE BAYES")
print(cm5)

# 6. DECISION TREE 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm6=confusion_matrix(y_test,y_pred)
print("DECISION TREE")
print(cm6)

# 7. RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)
y_pred= rfc.predict(X_test)

cm7 = confusion_matrix(y_test,y_pred)
print("RANDOM FOREST")
print(cm7)

#ROC, TPR, FPR 
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# Example predictions and true values
y_proba = rfc.predict_proba(X_test)  # Ensure your 'rfc' is defined
y_test = np.array(y_test)  # True labels

print("True values (y_test):", y_test)
print("Predicted probabilities (y_proba[:,1]):", y_proba[:, 1])

pos_label_value = 'Iris-versicolor'

# Converting positive label to numeric value
unique_labels = np.unique(y_test)
if pos_label_value in unique_labels:
    pos_label_index = np.where(unique_labels == pos_label_value)[0][0]
    
    # Converting labels to numeric values
    y_test_numeric = np.where(y_test == pos_label_value, pos_label_index, 1 - pos_label_index)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test_numeric, y_proba[:, 1], pos_label=pos_label_index)
    auc_score = metrics.auc(fpr, tpr)  # Calculate AUC
    print("FPR:", fpr)
    print("TPR:", tpr)
    print("Thresholds:", thresholds)
    print("AUC:", auc_score)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
else:
    print(f"Positive label ({pos_label_value}) not found in y_test.")