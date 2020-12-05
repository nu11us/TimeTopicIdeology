d = open('data')
e = open('data.csv','w')
for i in d:
    s=i.split('-')[0]+','+i.split('-')[1]+','+i.split('-')[2]
    s = s.rstrip('\n')
    s=s.replace(' ',',')
    e.write(s+'\n')
e.close()
d.close()