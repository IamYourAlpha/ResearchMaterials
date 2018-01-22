import pandas as pd
import numpy as np

data = pd.read_csv('with_cluster_group.csv')
cats = pd.read_csv('Catagories.csv')
data['LargeCl'] = 'na'
data['MiddleCl'] = 'na'
f = ['categoryId', 'largeCl']
c = np.array(cats)
print c
for cat in c:
    for problem in data['cat']:
        if(cat[0] == problem):
            data.loc[ data['cat']==cat[0], 'LargeCl'] = cat[1]
            data.loc[ data['cat']==cat[0], 'MiddleCl'] = cat[2]
            
    #for problem in data['cat']:
     #   if( problem == cat ):
     #       temp['lc']
data.to_csv('with_cluster_group.csv', encoding='utf-8')

            
        
