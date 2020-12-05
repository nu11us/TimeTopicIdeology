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

dates = ['01','02','03','04','05','06','07','08','09','10','11','12']
left_subs = ['anarchism', 'fullcommunism', 'progressive', 'sandersforpresident', 'socialism']
right_subs = ['anarcho_capitalism', 'conservative', 'cringeanarchy', 'libertarian', 'republican', 'the_donald']
subs = left_subs + right_subs

models = {}
dicts = {}

for d in dates:
    m = f'subdata/2016-{d}.tsv'
    for s in subs:
        if d in ['08','09','10','11','12'] and s == 'sandersforpresident':
            st = f'2016-{d}-political_revolution'
        else:
            st = f'2016-{d}-{s}'
        models[st] = LdaMulticore.load('models/'+st+'-model', mmap='r')
        dicts[st] = Dictionary.load('models/'+st+'-dict', mmap='r')

tags = [i for i in model.keys()]
print(tags)