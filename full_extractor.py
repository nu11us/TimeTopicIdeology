
import pandas as pd
import datetime
from collections import defaultdict
import sys

a = open('misc/data')
for linex in a:
    target = linex.split(' ')[0]
    y = linex.split(' ')[1].rstrip('\n')
    data = open('politics.tsv')
    f = open('data/'+target+'.tsv','w+')
    names =["id","subreddit","created_utc","author","link_id","parent_id","score","body"]
    f.write('\t'.join(names)+'\n')
    header = True
    dct = defaultdict(int)
    x = 0
    save = None
    for line in data:
        if header:
            header = False 
        else:
            s = line.rstrip('\n').split('\t')
            time = s[2] 
            d = datetime.date.fromtimestamp(int(time))
            n0 = d.strftime('%Y-%m')
            if n0 in target:
                subreddit = s[1]
                name = n0+"-"+subreddit
                if name == target:
                    save = d.month
                    f.write(line)
                    x+=1
                    #print(target,x)
            if x == int(y):
                break
    print(target)
    data.close()
    f.close()