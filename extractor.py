
import pandas as pd
import datetime
from collections import defaultdict
import sys

if __name__ == '__main__':
    data = open('politics.tsv')
    target = sys.argv[1]
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
            subreddit = s[1]
            d = datetime.date.fromtimestamp(int(time))
            name = d.strftime("%Y-%m")+"-"+subreddit
            if name == target:
                save = d.month
                f.write(line)
                x += 1
            if save:
                if d.month == save + 2:
                    break 
                
            