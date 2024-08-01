# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

Reviews = pd.read_csv(r'C:\Users\Arif Furkan\OneDrive\Belgeler\Python_kullanirken\Restaurant_Reviews.csv')
print(Reviews)

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

collection = []
for i in range(len(Reviews)):
    Comment = re.sub('[^a-zA-Z]',' ',Reviews['Review'][i])
    Comment = Comment.lower()
    Comment = Comment.split()
    english_stopwords = set(stopwords.words('english'))
    Comment = [ps.stem(Word) for Word in Comment if Word not in english_stopwords]
    Comment = ' '.join(Comment)
    collection.append(Comment)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1996)
X = cv.fit_transform(collection).toarray()
y = Reviews.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train)
print(y_train)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

