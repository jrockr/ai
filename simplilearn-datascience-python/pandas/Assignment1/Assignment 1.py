# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:46:05 2019

@author: C740129
"""

import pandas as pd

df_faa_dataset = pd.read_csv('C:\\Users\\C740129\\OneDrive - Standard Bank\\certification\\Ai Engineer\\DS\\pandas\\Assignment1\\faa_ai_prelim\\faa_ai_prelim.csv')

print(df_faa_dataset)

df_faa_dataset.shape

print(df_faa_dataset.head())