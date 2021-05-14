# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:31:56 2021

@author: raven002
"""
print("reading in Data...")
df = pd.read_excel ("C:/Users/raven002/OneDrive - Guidehouse/Desktop/FIMA NLP Project/reason_justification.xlsx", converters={'Justification': lambda x: str(x)})
print("data loaded successfully")

#dropping nas
df['Justification']=df['Justification'].fillna("blank")


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import xgboost
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


df['Justification']=df['Justification'].map(lambda s:preprocess(s)) 


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
classifier = XGBClassifier(max_depth=8, learning_rate=0.3, random_state=0)
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