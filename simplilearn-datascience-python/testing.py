# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:00:15 2019

@author: C740129
"""

import pandas as pd
olympic_data_dict = {'London':{2012,205},'Beijing':{2008,204}}
df_olympic_data = pd.DataFrame(olympic_data_dict)

print(df_olympic_data)