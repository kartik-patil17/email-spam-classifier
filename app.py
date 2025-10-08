import pickle
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Classifier')

input_sms = st.text_area('Enter the message')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

if st.button('predict'):

#1.preprocess
#2.vectorize 
#3.predict
#4.display



    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])
    res = model.predict(vector_input)[0]

    if res == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')





