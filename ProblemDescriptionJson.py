import pandas as pd
import json
import urllib2
import requests
#url = "http://judgedat.u-aizu.ac.jp/rating/problems/GRL_5_C/statistics"
problems = pd.read_csv('Problems_csv.csv')
problem_list = problems['id']
str1 = "http://judgedat.u-aizu.ac.jp/rating/problems/"
str2 = "/statistics"
ac = list()
tle = list()
mle = list()
wa = list()
rte = list()

for problem in problem_list:
    url = str1 + problem + str2
    vertict = requests.get(url).json()
    ac.append(vertict['statusStatistics'][0]['Accepted'])
    tle.append(vertict['statusStatistics'][0]['Time Limit Exceeded'])
    mle.append(vertict['statusStatistics'][0]['Memory Limit Exceeded'])
    wa.append(vertict['statusStatistics'][0]['Wrong Answer'])
    rte.append(vertict['statusStatistics'][0]['Runtime Error'])
problems['AC'] = ac
problems['TLE'] = tle
problems['MLE'] = mle
problems['WA'] = wa
problems['RTE'] = rte

#url = str1 + problem_list[0] +  str2
#import requests
#data = requests.get(url).json()
#d =  data['statusStatistics'][0]['Accepted']

#print d

problems.to_csv('CompleteProblemSet2.csv', encoding='utf-8')
