import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import plotly.graph_objects as go
from gensim import models
from gensim.corpora.dictionary import Dictionary
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.phrases import Phrases
import re
from nltk.corpus import stopwords
import nltk
import urllib.request, urllib.error, urllib.parse
import ssl
# nltk.download('stopwords')
# nltk.download('wordnet')
st.set_page_config(layout='wide')
from datetime import datetime

st.title('News classifier - LAMFO & University of Essex & Microsoft AI for Health')


# @st.cache
def importdados():
        
    stop_words = set(stopwords.words('english')) 
    # classificacao = False
    lda_model = models.LdaModel.load("ldamodel")
    common_dictionary = Dictionary.load("ldadic")
    phrase_model = Phrases.load("phaser")

    return stop_words, lda_model, common_dictionary, phrase_model

now = datetime.now()
stop_words, lda_model, common_dictionary, phrase_model = importdados()
later = datetime.now()
difference = (later - now).total_seconds()
st.write(difference)

def classificalda(x):
    x = remove_stopwords(x)
    x =  re.sub(r'\W', ' ', x)
    x =  re.sub(r' \w ', ' ', x)

    x =  x.lower()
    x =  x.split()
    lemmatizer = WordNetLemmatizer()
    x =  [lemmatizer.lemmatize(token) for token in x]
    x = [w for w in x if not w in stop_words]
    x =  phrase_model[x]



    other_corpus = common_dictionary.doc2bow(x)
    vector = lda_model[other_corpus]
    topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
    a = topic_percs_sorted[0][0]
    b = topic_percs_sorted[1][0]
    c = topic_percs_sorted[2][0]
    a_ = topic_percs_sorted[0][1]
    b_ = topic_percs_sorted[1][1]
    c_ = topic_percs_sorted[2][1]

    if a_ < 0.2:
        st.error('Results have low confidence, consider changing or rewriting the input.')
        st.stop()
    return (a,b,c,a_,b_,c_)


teste = st.text_area("Enter text to classify",height = 200,max_chars = 10000)
url = st.text_input('Or enter URL to be checked')

# url = st.text_area("Or enter URL to be checked",height = 200,max_chars = 1000)


if url and teste:
    st.error('Please fill only one field (URL or text)')
    st.stop()


if not st.button('Processar'):
    st.stop()

if url:
    tipoinput = "URL"
    if "www" not in url:
        url = "www." + url
    if "http" not in url:
        url = "http://"+url
    
    try:    
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(url,context=context)
        webContent = response.read()
        if len(webContent) > 10000:
            teste = webContent[0:10000]
        teste = webContent
    except:
        st.error('URL not valid')
        st.stop()
    
else:
    tipoinput = "given text"
    if len(teste) < 100:   
        st.error('Text provided is too short. Please provide a text with 100 characters or more')
        st.stop()
        

st.spinner()
with st.spinner(text='In progress'):
     a,b,c,a_,b_,c_ = classificalda(teste)
     st.success('Done! ' +"Results for "+ tipoinput +". This text contais topics: A (A_%); B (B_%); C* (C_%). Check bellow the meaning of each topic.".replace("A_",str(int(100*a_))).replace("B_",str(int(100*b_))).replace("C_",str(int(100*c_))).replace("A",str(a+1)).replace("B",str(b+1)).replace("C*",str(c+1)))




d = {'lda1': [a], 'lda2': [b],'lda2': [b],'lda3': [c]}
df = pd.DataFrame(data=d)







# if not classificacao:
#     valor = 0
#     cor = "red"
# else:
#     valor = int(classificacao)
#     if valor>3:
#         cor = "green"
#     else:
#         cor = "red"
    
# fig = go.Figure(go.Indicator(
#     mode = "gauge+number",
#     value = valor,
#     gauge = {"axis": {
#         "range": [0,6]
#       },
#           "bar": {"color" :cor}},
#     title = {'text': "Truthiness"},
#     domain = {'x': (0,1), 'y': (0,1)}
# ))


# st.plotly_chart(fig, use_container_width=True)


# st.write()


f = open("LDA.html" , "r")
data = f.read()

st.components.v1.html(data, width = 1200, height = 900, scrolling = True)


url = 'https://www.bbc.com/future/article/20210315-why-noise-pollution-is-bad-for-your-heart'



