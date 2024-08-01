# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# File path and data reading
try:
    Reviews = pd.read_csv(r'C:\Users\Arif Furkan\OneDrive\Belgeler\Python_kullanirken\Restaurant_Reviews.csv', on_bad_lines='skip')
    print(Reviews.head())  
except pd.errors.ParserError as e:
    print("An error occurred while reading the CSV file:", e)
    exit()

# Checking and removing missing values
print(Reviews.isnull().sum())  
Reviews = Reviews.dropna()  
print(Reviews.isnull().sum())  

# NLTK stopwords download
nltk.download('stopwords')
ps = PorterStemmer()

# Preprocessing
collection = []
for i in range(len(Reviews)):
    Comment = re.sub('[^a-zA-Z]', ' ', Reviews['Review'].iloc[i])
    Comment = Comment.lower()
    Comment = Comment.split()
    Comment = [ps.stem(Word) for Word in Comment if not Word in set(stopwords.words('english'))]
    Comment = ' '.join(Comment)
    collection.append(Comment)

# Feature Extraction - Bag of Words (BOW)
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(collection).toarray()  # Independent variable
y = Reviews['Liked'].values  # The dependent variable

# Separating the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Naive Bayes model training
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Guess
y_pred = gnb.predict(X_test)

# Creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion matrix analysis
accuracy = np.trace(cm) / np.sum(cm)
print(f"Accuracy: {accuracy * 100:.2f}%")