import sys
import pandas as pd
import gensim
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models import LdaMulticore
from pprint import pprint

class Dataset:
    def __init__(self,path):
        self.path = path
        self.df = pd.read_csv(
            path,
            header = 0,
            sep = '\t',
            names =["id","subreddit","created_utc","author","link_id","parent_id","score","body"],
            usecols=['body']
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


if __name__ == '__main__':
    d = Dataset(sys.argv[1])
    d.clean()
    docs = d.df['body']
    bigrams = Phrases(docs, min_count=0)
    for idx in range(len(docs)):
        for token in bigrams[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)
    dct = Dictionary(docs)
    dct.filter_extremes(no_below=10, no_above=0.75)
    bow = [dct.doc2bow(d) for d in docs]


    t0 = dct[0]
    id2word = dct.id2token
    model = LdaMulticore(
        workers=4,
        corpus=bow,
        id2word=id2word,
        chunksize=10000,
        iterations=50,
        num_topics=5,
        passes=20,
        eval_every=None
    )
    top_topics = model.top_topics(bow)

    pprint(top_topics)
