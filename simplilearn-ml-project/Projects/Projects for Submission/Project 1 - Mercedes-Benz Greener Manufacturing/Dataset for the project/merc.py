#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:37:09 2020

@author: joydeep
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#load test and train data 

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
threshold = 0

train_variance = train_df.var()
index_var = train_df.var() == threshold
index_var=index_var[index_var==True]
index_var_list = index_var.index.tolist()
new_train_df = train_df.drop(axis=1, labels=index_var_list)
new_test_df = test_df.drop(axis=1, labels=index_var_list)

plt.figure(figsize=(8,6))
plt.scatter(range(new_train_df.shape[0]), np.sort(new_train_df.y.values))
plt.xlabel('index')
plt.ylabel('y')
plt.show() 

# Check for null and unique values for test and train sets
missing_train_df = new_train_df.isnull().sum(axis=0).reset_index()
missing_train_df.columns = ['column_name', 'missing_count']
missing_train_df = missing_train_df.loc[missing_train_df['missing_count']>0]
print("missing Count : ",missing_train_df)


# Apply label encoder
X = new_train_df.iloc[:,2:]
y = new_train_df.iloc[:,1]

le = LabelEncoder()
X['X0'] = le.fit_transform(X['X0'] )
X['X1'] = le.fit_transform(X['X1'] )
X['X2'] = le.fit_transform(X['X2'] )
X['X3'] = le.fit_transform(X['X3'] )
X['X4'] = le.fit_transform(X['X4'] )
X['X5'] = le.fit_transform(X['X5'] )
X['X6'] = le.fit_transform(X['X6'] )
X['X8'] = le.fit_transform(X['X8'] )

# Perform dimensionality reduction.
X_normalized = StandardScaler().fit_transform(X)
X_normalized=pd.DataFrame(X_normalized,columns=X.columns)

pca = PCA()
x_pca = pca.fit_transform(X_normalized)
x_pca = pd.DataFrame(x_pca)
#print("x_pca : %",x_pca.head())
pca_variance = pca.explained_variance_ratio_

# plot the important features
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2,random_state=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'silent': 1,
    'learning_rate':0.01,
    'verbose':True
}
dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=r2_score, maximize=True)

# plot the important features #
fig, ax = plt.subplots(figsize=(10,20))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# Predict your test_df values using xgboost
new_test_df = new_test_df.iloc[:,1:]
new_test_df['X0'] = le.fit_transform(new_test_df['X0'] )
new_test_df['X1'] = le.fit_transform(new_test_df['X1'] )
new_test_df['X2'] = le.fit_transform(new_test_df['X2'] )
new_test_df['X3'] = le.fit_transform(new_test_df['X3'] )
new_test_df['X4'] = le.fit_transform(new_test_df['X4'] )
new_test_df['X5'] = le.fit_transform(new_test_df['X5'] )
new_test_df['X6'] = le.fit_transform(new_test_df['X6'] )
new_test_df['X8'] = le.fit_transform(new_test_df['X8'] )

test_df_normalized = StandardScaler().fit_transform(new_test_df)
test_df_normalized=pd.DataFrame(test_df_normalized,columns=new_test_df.columns)

d_test = xgb.DMatrix(X_test)
predict = model.predict(d_test)
print("Predicted : ", predict)