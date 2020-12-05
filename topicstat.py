import sys
import pandas as pd
import gensim
import nltk
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models import LdaMulticore
from pprint import pprint
import csv
import math
import os
import psutil
from collections import defaultdict

dates = ['01','02','03','04','05','06','07','08','09','10','11','12']
left_subs = ['anarchism', 'fullcommunism', 'progressive', 'sandersforpresident', 'socialism']
right_subs = ['anarcho_capitalism', 'conservative', 'cringeanarchy', 'libertarian', 'republican', 'the_donald']
subs = left_subs + right_subs

models = {}
dicts = {}
sx = defaultdict(int)

for d in dates:
    for s in subs:
        if d in ['08','09','10','11','12'] and s == 'sandersforpresident':
            st = f'data/2016-{d}-political_revolution.tsv'
            a = -1
            with open(st) as z:
                for line in z:
                    a += 1
            sx['political_revolution'] += a

        else:
            st = f'data/2016-{d}-{s}.tsv'
            a = -1
            with open(st) as z:
                for line in z:
                    a += 1
            sx[s] += a
for i in sx:
    print(i, sx[i])
"""
tags = [i for i in models.keys()]
lst = []
for t1 in tags:
    for t2 in tags:
        if t1 != t2:
            m1 = models[t1]
            m2 = models[t2]
            dif = m1.diff(m2,distance='jaccard', num_words=1000, annotation=False)
            scalar = sum([sum(i) for i in dif[0]])
            lst.append([scalar, f'{t1} {t2}'])
    
    x = sorted(lst)
    for i in x[:10]:
        print(i)
    for i in x[-10:]:
        print(i)
    print()
    print()
"""