# SpamClassifier  

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the Dataset
dataset = pd.read_csv('SMSSpamCollection',delimiter = '\t', names = ["label ", "message"])

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    message = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus.append(message)
    
# Creating the bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
cv = CountVectorizer(max_features = 5500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0].values
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)

#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting Random_Forest_Classification to the Training Set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0 )
classifier.fit(X_train, y_train)

#Predecting the results of Test Set
y_pred = classifier.predict(X_test)

#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy = (953 + 135)/1100 #98.9%


