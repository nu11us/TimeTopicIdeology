import sys
import pandas as pd
import gensim
import nltk
import re
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models import LdaMulticore
from pprint import pprint
import csv
import math
import os
import psutil

lemmatizer = WordNetLemmatizer()
stop = set(nltk.corpus.stopwords.words('english'))
def tokenize(text):
    # Cleaning  
    temp = text.replace('\t','').replace('\r','').replace('\n','').replace('`','').replace("'",'').split(' ')
    tokens  = []
    for word in temp:
        word = word.lower()
        if 'http' in word or 'www' in word or '.com' in word or word.count('/')>3:
            tokens.append('URL')
        elif len(word)>2  and word[:2] == 'r/' or word[:3] == '/r/':
            tokens.append('URL')
        elif len(word)>2  and word[:2] == 'u/' or word[:3] == '/u/':
            tokens.append('SCREEN_NAME')
        elif not word.isnumeric() and word != ' ':
            word = re.sub(r'[^\w\s]', '', word) 
            if word not in stop: # Stop word filter
                tokens.append(lemmatizer.lemmatize(word))
    if tokens == []:
        return None
    return tokens


pid = os.getpid()
py = psutil.Process(pid)

dates = ['01','02','03','04','05','06','07','08','09','10','11','12']
left_subs = ['anarchism', 'fullcommunism', 'sandersforpresident', 'progressive', 'socialism']
right_subs = ['anarcho_capitalism', 'conservative', 'cringeanarchy', 'libertarian', 'republican', 'the_donald']
subs = left_subs + right_subs

y = "Medicare for all is the best proposal I have heard. I love these caucuses. Lets go Bernie!"
x = "MAGA I LoVe trump jesus christ in heaven praise reagan"
x = tokenize(x)
z = True
models = {}
dcts = {}
a = []
for d in dates:
    for s in subs:
        if z:
            z = False
            model = LdaMulticore.load(f'models/2016-{d}-{s}-model', mmap='r')
            dct = Dictionary.load(f'models/2016-{d}-{s}-dict', mmap='r')

            models[f'{d}{s}'] = model
            dcts[f'{d}{s}'] = dct
            
            m0 = model
            d0 = dct
        else:
            model = LdaMulticore.load(f'models/2016-{d}-{s}-model', mmap='r')
            dct = Dictionary.load(f'models/2016-{d}-{s}-dict', mmap='r')

            models[f'{d}{s}'] = model
            dcts[f'{d}{s}'] = dct
            x_bow = dct.doc2bow([x])
            inf = model.inference([dct.doc2bow(x)])[0][0]
            
            #a.append([inf.argmax(axis=0),f'{d}{s}'])
            #a.append((model.log_perplexity([dct.doc2bow(x)]), f'{d}{s}'))
#a = sorted(a)
#for i in a:
#    print(i)