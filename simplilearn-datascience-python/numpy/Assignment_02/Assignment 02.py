# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:31:35 2019

@author: C740129
"""

import numpy as np

countries = np.array(['GBR','CHN','RUS','US','KOR','JPN','GER'])
gold = np.array([29,38,24,46,13,7,11])
silver = np.array([17,28,25,28,8,14,11])
bronze = np.array([19,22,32,29,7,17,14])

country_with_max_gold = countries[gold.argmax()]

print(country_with_max_gold)

print(countries[gold > 20])

for i in range(len(gold)):
    if gold[i]>20:
        print("Countries with more then 20 medels {}".format(countries[i]))
        
        
for i in range(len(countries)):
    total_medel = gold[i]+silver[i]+bronze[i]
    print("Country : {} Gold : {} Total : {}".format(countries[i],gold[i],total_medel))