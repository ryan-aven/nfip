# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:57:28 2021

@author: raven002
"""


print("importing base libraries...")
import pandas as pd
import re
import nltk
print("base libraries imported")

#importing the data set
print("reading in Data...")
df = pd.read_excel ("C:/Users/raven002/OneDrive - Guidehouse/Desktop/FIMA NLP Project/reason_justification.xlsx", converters={'Justification': lambda x: str(x)})
print("data loaded successfully")

#setting x and y values
y = df.ReasonRequestID
x = df.Justification

print("removing special characters and lemmatization...")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')

documents = []

lemmatizer = WordNetLemmatizer()

def preprocess(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    #tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    document = tokenizer.tokenize(document)  
    document = [w for w in document if len(w) > 2 if not w in stopwords.words('english')]

    # Lemmatization
    document = [lemmatizer.lemmatize(word) for word in document]
    return " ".join(document)

df['Justification']=df['Justification'].map(lambda s:preprocess(s)) 
    
print("special characters removed, lemmatization complete")

#setting x and y values
y = df.ReasonRequestID
x = df.Justification

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=200, max_df=.8, stop_words=stopwords.words('english'))
x = vectorizer.fit_transform(x).toarray()

#reweighting based on tf-idf values
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()

#splitting the data sets up into training and test data sets
print("splitting data sets into testing and training sets")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

#importing the gradient boosted classifier
print("uploading gradient boosted classifier")
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=8, learning_rate=0.2, random_state=0)
classifier.fit(x_train, y_train)

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
#classifier.fit(x_train, y_train)

#making our predictions
print("fitting model...")
#just predictions
y_pred = classifier.predict(x_test)
#a separate probability generator
y_pred2 = classifier.predict_proba(x_test)

#producing summary metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))