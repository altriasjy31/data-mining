#%%
import pandas as pd
import numpy as np
from mlxtend import preprocessing as pp
from mlxtend import frequent_patterns as fp

vgData_path = "D:\\文档\\grd\\course\\数据挖掘\\assignments\\06\\284_618_bundle_archive\\vgsales.csv"
vgData = pd.read_csv(vgData_path)

vgData = vgData.iloc[:,1:]
vgData

#%%
#对Name, Platform, Year, Genre, Publisher进行关联规则挖掘
vgSub = vgData.iloc[:,0:5]
vgSub.head()

#%%
vgs_ll = vgSub.to_numpy().astype("str").tolist()
vgs_ll[:10]

#%%
te = pp.TransactionEncoder()
te_nd = te.fit(vgs_ll).transform(vgs_ll)
te_df = pd.DataFrame(te_nd, columns=te.columns_)
te_df.head()

#%%
frequent_itemset = fp.apriori(te_df, min_support=0.003, use_colnames=True)
frequent_itemset.loc[:5]
frequent_itemset.loc[-6:]

#%%
#lift
as_rules = fp.association_rules(frequent_itemset,metric="lift",min_threshold=0.1)
as_rules.loc[:5]
as_rules.loc[-6:]


# %%
from sklearn import svm
from sklearn import model_selection as md
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import itertools as it

#这是接下来要预测的sales的名称
sales_l = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]

#每次选中一列后就把剩余列作为参数
#但是要把对应文字转为整数
change_l = ["Name","Platform","Genre","Publisher"]
for c in change_l:
    vg_s = vgData.loc[:,c]
    g = it.groupby(vg_s.sort_values())
    l = [v for v, _ in g]
    for i, v in enumerate(l):
        vgData.loc[vgData.loc[:,c]==v,c] = i
vgData

#%%
#使用两折交叉验证
#这里以Global sales为例
kf = md.KFold(n_splits=2,shuffle=True)

selected = 9

mark = [True] * 10
mark[selected] = False

for train_index, test_index in kf.split(vgData.loc[:,mark],vgData.iloc[:,selected]):
    vg_train = vgData.loc[train_index,mark]
    y_train = vgData.iloc[train_index,selected]
    vg_test = vgData.iloc[test_index,mark]
    y_test = vgData.iloc[test_index,selected]

vg_train.fillna(0,inplace=True)
#%%
y_train.fillna(0,inplace=True)
#%%
y_test.fillna(0,inplace=True)
vg_test.fillna(0,inplace=True)
#%%
X_train = vg_train.to_numpy(np.float32)
X_test = vg_test.to_numpy(np.float32)
y = y_train.to_numpy(np.float32)
#%%
regr = make_pipeline(StandardScaler(),svm.SVR(C=1.0,epsilon=0.2))
model = regr.fit(X_train,y)
y_prd = model.predict(X_test)
#%%
y_prd
#%%
y_test

#%%
