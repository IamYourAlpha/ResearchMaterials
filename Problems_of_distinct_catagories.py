import pandas as pd
import json
import urllib2
import requests
import numpy as np
import csv
import json 

data = pd.read_csv('Catagories.csv')
cat_list = data['categoryId']
#cat_list = np.array([cat_list])
#cat_list = np.sort(cat_list)

#url = "http://judgeapi.u-aizu.ac.jp/problems/categories/1"
str1 = "http://judgeapi.u-aizu.ac.jp/problems/categories/"


list_of_problems = dict()
tab = pd.read_csv('dumy.csv')
print tab['id'].head()
tab['cat'] = -1 
for cat in cat_list:
    url1 = str1 + str(cat)
    data = requests.get(url1).json()
    problem_id_list = list()
    numOfProblems = data['numberOfProblems']
    for i in range(numOfProblems):
        for j in tab['id']:
         #problem_id_list.append(data['problems'][i]['id'])
            if( j == data['problems'][i]['id'] ):
                tab.loc[ (tab['id']==data['problems'][i]['id']) , 'cat'] = cat
                print "match"
                #tab.loc[ j  == data['problems'][i]['id'], 'cat'] = cat 
                break
        #print data['problems'][i]['id']
    #list_of_problems[str(cat)] = problem_id_list
print tab['cat']
tab.to_csv('CompleteProblemSet_with_catagories.csv', encoding='utf-8')
#data = requests.get(url).json()
#problem_id_list = list()
#numOfProblems = data['numberOfProblems']
#for i in range(numOfProblems):
#    problem_id_list.append(data['problems'][i]['id'])

#print problem_id_list[0]
