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

x = "Capitalism is evil."
x = tokenize(x)

#models = []
#for d in dates:
#    for s in subs:

#        models.append(LdaMulticore.load(f'models/2016-{d}-{s}-model'))
#        rev = models[-1].id2word
#        rev_d = {b: a for a,b in rev.items()}
#        x_bow = [rev_d[i] for i in x if i in rev_d]
#        pprint(models[-1].show_topics())
#        pprint(models[-1].top_topics([x_bow]))

model = LdaMulticore.load('models/2016-11-political_revolution-model', mmap='r')
dct = Dictionary.load('models/2016-11-political_revolution-dict', mmap='r')
print(dct.id2token[111])
print(model.id2word[111])


#x_val = model.get_document_topics([x_bow])
#pprint(top_topics)
#print(x_val)
#for i in x_val:
#    print(i)
