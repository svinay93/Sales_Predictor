
# coding: utf-8

# In[41]:


import os
import gc
from sklearn import linear_model
import pandas as pd
import numpy as np
import time


t_one = time.time()
data_training = pd.read_csv( 'sales_train_v2.csv')
data_test = pd.read_csv( 'test.csv')
print(data_training)

#remove the columns item price and date 
data_training.drop(['date','item_price'], axis=1, inplace=True)
#***************************************

# month based aggregation
aggreg_train = data_training.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg(sum) 
aggreg_train.columns = ['date_block_num','shop_id','item_id','item_cnt_month']
del data_training
gc.collect()
#*****************************************

# only last 10 months
aggreg_train = aggreg_train[aggreg_train.date_block_num>23]
#*********************************************

# normalize actuals to range of 0 to 20
aggreg_train['item_cnt_month'] = np.maximum(0, np.minimum(20, aggreg_train['item_cnt_month']))
#********************************************

# pivot table
aggreg_train['date_block_num'] = 33 - aggreg_train['date_block_num'] # change it, so that it is 0 to 9
train_p = aggreg_train.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', aggfunc='sum', fill_value=0)
del aggreg_train
gc.collect()
data = np.array(train_p.values, dtype=np.int32)
#*****************************************************

# Create linear regression and an object to reference it
regr = linear_model.LinearRegression()
#****************************************************

# Training
x = data[:,1:10]
y = data[:,0]
#*****************************************************

#fitting
regr.fit(x, y)
#***************************************************

#prediction step
pred = regr.predict(x)
#******************************************************

# The ROOT MEAN SQUARED ERROR
print('MSRE: %.2f'%np.sqrt(((y-pred)*(y-pred)).mean()))
#*******************************************************

# Output file creation
p = regr.predict(data[:,0:9])
train_p['pred'] = p
train_p.drop('item_cnt_month',axis=1, inplace=True)
train_p.reset_index(level=['item_id', 'shop_id'], inplace=True)
s_df = pd.merge(data_test, train_p, how='left', on=['item_id', 'shop_id'])
#**************************************************************


# to get sales by shop as percentage of average sales for all shops
ss = train_p.drop('item_id', axis=1).groupby('shop_id', as_index=False).agg(sum)
ss['pred'] = ss['pred'] / ss['pred'].mean() # 54 shops. range: 0.03 to 3.9
#****************************************************************

# get sales by item as percentage of average sales for all items
si = train_p.drop('shop_id', axis=1).groupby('item_id', as_index=False).agg(sum)
si['pred'] = si['pred'] / si['pred'].mean() # 11249 items. range: 0.03 to 90
s_df = pd.merge(s_df, ss, how='left', on='shop_id')
s_df = pd.merge(s_df, si, how='left', on='item_id')
s_df.columns = ['ID', 'shop_id','item_id', 'item_cnt_month', 'shop', 'item']
v = s_df['item_cnt_month'].mean()
s_df['pred2'] = v * s_df['shop'] * s_df['item'] * 0.225 # cut in X for new items
s_df['item_cnt_month'].fillna(s_df['pred2'], inplace=True)
#***************************************************************

# Missing data dealt with by averaging
s_df['pred3'] = v * s_df['shop'] * 0.225 # cut in X for new items
s_df['item_cnt_month'].fillna(s_df['pred3'], inplace=True)
#*************************************************************


s_df.drop(['shop_id','item_id','shop','item','pred2','pred3'], axis=1, inplace=True)
s_df['item_cnt_month'] = np.maximum(0, np.minimum(20, s_df['item_cnt_month']))
s_df.to_csv('submission.csv', index=False)
#*****************************************END**************************************

