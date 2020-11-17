
import pandas as pd
import datetime
from collections import defaultdict

data = open('politics.tsv')

names =["id","subreddit","created_utc","author","link_id","parent_id","score","body"]
header = True
dct = defaultdict(int)
for line in data:
    if header:
       header = False 
    else:
        s = line.rstrip('\n').split('\t')
        time = s[2] 
        subreddit = s[1]
        d = datetime.date.fromtimestamp(int(time))
        dct[d.strftime("%Y-%m")+"-"+subreddit] += 1
for elem in dct:
    print(elem, dct[elem])