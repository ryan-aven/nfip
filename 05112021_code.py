# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:13:50 2021

@author: raven002
"""

print("importing base libraries...")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

print("base libraries imported")

#importing the data set
print("reading in Data...")
df = pd.read_excel ("C:/Users/raven002/OneDrive - Guidehouse/Desktop/FIMA NLP Project/reason_justification.xlsx", converters={'Justification': lambda x: str(x)})
print("data loaded successfully")

#dropping nas
df['Justification']=df['Justification'].fillna("blank")


#load spacy
import spacy 
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stops = stopwords.words("english")

#make a normalizer/lemmatization function
def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)


#applying the lemmatized function
df['Justification_cleaned'] = df['Justification'].apply(normalize, lowercase=True, remove_stopwords=True)

#setting x and y values
y = df.ReasonRequestID
x = df.Justification_cleaned

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=100, max_df=.8, stop_words=stopwords.words('english'))
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
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(max_depth=8, learning_rate=0.3, random_state=0)
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

predictions_truths = pd.concat([y_pred, y_test], axis=1).reset_index()


#this is really annoying lol
y_test = y_test.to_frame()
predictions_truths = pd.concat(y_pred, y_test)

