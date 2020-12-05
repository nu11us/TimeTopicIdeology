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
class Dataset:
    def __init__(self,path):
        self.path = path
        self.df = pd.read_csv(
            path,
            header = 0,
            sep = '\t',
            names =["id","subreddit","created_utc","author","link_id","parent_id","score","body"],
            usecols=['body'],
            encoding='utf-8',
            quoting=csv.QUOTE_NONE,
            error_bad_lines=False
        )
        self.stop = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
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
                if word not in self.stop: # Stop word filter
                    tokens.append(self.lemmatizer.lemmatize(word))
        if tokens == []:
            return None
        return tokens

    def clean(self):
        self.df['body'] = self.df['body'].apply(self.tokenize)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def ext_clean(self, text):
        return self.tokenize(text)

if __name__ == '__main__':
    f = ['anarcho_capitalism','conservative','libertarian','republican','the_donald','cringeanarchy']
    for ff in f:
        xx = [
            f'data/2016-01-{ff}.tsv',
            f'data/2016-02-{ff}.tsv',
            f'data/2016-03-{ff}.tsv',
            f'data/2016-04-{ff}.tsv',
            f'data/2016-05-{ff}.tsv',
            f'data/2016-06-{ff}.tsv',
            f'data/2016-07-{ff}.tsv',
            f'data/2016-08-{ff}.tsv',
            f'data/2016-09-{ff}.tsv',
            f'data/2016-10-{ff}.tsv',
            f'data/2016-11-{ff}.tsv',
            f'data/2016-12-{ff}.tsv'       
        ]

        for arg in xx:
            print(arg)
            d = Dataset(arg)
            d.clean()
            docs = d.df['body']
            #bigrams = Phrases(docs, min_count=10)
            #for idx in range(len(docs)):
            #    for token in bigrams[docs[idx]]:
            #        if '_' in token:
            #            docs[idx].append(token)
            dct = Dictionary(docs)
            dct.filter_extremes(no_below=10, no_above=0.75)
            bow = [dct.doc2bow(d) for d in docs]

            t0 = dct[0]
            id2word = dct.id2token

            new_name = 'models' + str(arg).replace('data','').replace('.tsv','')+'-dict'
            dct.save(new_name, sep_limit=134217728)
            del dct
            del d
