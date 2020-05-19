#%%
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 
from mlxtend.preprocessing import TransactionEncoder
import scipy as sp
from scipy import stats

raw_data_path = "/mnt/d/文档/grd/course/数据挖掘/assignments/04/wine-reviews/winemag-data_first150k.csv"
raw_data = pd.read_csv(raw_data_path)
raw_data.head()


# %%
raw_data.columns

# %%
#简单修复数据
#首先发现第一列没有表头名称
raw_data = raw_data.rename(columns={"Unnamed: 0": "index"})
raw_data.columns

#%%
#去掉城市，省份, 设计为空的行
raw_data.dropna(axis=0,subset=["country","province","designation"],inplace=True)
#去掉region_1,region_2同时为空的
raw_data.dropna(axis=0,how="all",subset=["region_1","region_2"],inplace=True)
#对region_2中为空的，用region_1的来填充
reg2_na_index = pd.isna(raw_data.loc[:,"region_2"])
raw_data.loc[reg2_na_index,"region_2"] = raw_data.loc[reg2_na_index,"region_1"]
# any(pd.isna(raw_data.loc[:,"region_2"]))

#对price中为空的，用众数来填充
price_nd = raw_data.loc[:,"price"].to_numpy(np.float32)
mode, count = stats.mode(price_nd,nan_policy="omit")
price_na_index = pd.isna(raw_data.loc[:,"price"])
raw_data.loc[price_na_index,"price"] = mode[0]

nan_l = []
for column in raw_data.columns:
    if any(pd.isna(raw_data.loc[:,column])):
        nan_l.append(column)
print(nan_l)

#%%
#根据country, points, price来分析
sub_data = raw_data.loc[:,["country","points","price"]]
sub_data.head()


#%%
#数据处理
tmp = sub_data.to_numpy(str)
X = tmp.tolist()
print(X[:10])

#%%
#编码
te = TransactionEncoder()
te_ary = te.fit(X).transform(X)
encode_df = pd.DataFrame(te_ary, columns=te.columns_)
encode_df.head()

#%%
#使用apriori

freq_items = apriori(encode_df,min_support=0.05,use_colnames=True)
freq_items.head()

#%%
#关联规则
rules = association_rules(freq_items, metric ="lift", min_threshold = 1)
rules

#%%