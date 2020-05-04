#!/usr/bin/evn python
# coding: utf-8

# In[ ]:
#处理数据集visit-patterns-by-census-block-group
#建立一个统一的类
#分析：
#cbg这个文件中，visitor_home_cbgs，visitor_work_cbgs，related_same_day_brand
#这几项缺存在缺失值
import numpy as np
import pandas as pd
import re
import scipy as scp
from scipy import stats
from matplotlib import pyplot as plt
import functools as func
from operator import itemgetter, attrgetter
import os

class Process(object):
    def __init__(self, csv_path):
        self.csv_path = csv_path
    def read_csv(self, file_path=None):
        if file_path is not None:
            self.csv_path = file_path
        csb_df = pd.read_csv(self.csv_path)
        return csb_df
    # 用于处理data frame里的数据
    # 对于attributes给出的属性，输出对应的频数
    #分析visitor_work_cbgs属性
    #visitor_work_cbgs属性，相当于一种指向关系
    #例如 10479572002 {"010479567011":67,"010479567021":60}
    #表示10479572002指向10479567011和1047957021，权值分别为67和60
    #计算每个顶点的出度，以及以该顶点为起始点的边的平均权重
    def prepare_visitor_work_cbgs(self, visitor_work_cbgs, resultPath, new_ndarray=False):
        if not new_ndarray:
            return
        n = visitor_work_cbgs.shape[0]

        data = np.zeros((2,n),np.float32)
        s_dict_l = [self.str2dict(visitor_work_cbgs[i]) for i in range(n)]
        # s0 = s_dict_l[0]
        data[0] = np.array([
            len(s_dict_l[i]) for i in range(n)
        ])

        def getValueAve(key2value):
            if key2value == {}:
                return -1
            value_nd = np.array(list(key2value.values()))
            return np.average(value_nd)
        
        data[1] = np.array([
            getValueAve(s_dict_l[i]) for i in range(n)
        ])
        
        np.save(open(resultPath,"wb"),data)

    
    #把str转为dict来分析
    #逗号作为分隔符
    def str2dict(self, str):
        d = {}
        m1 = r"{(?P<content>\S*)}"
        
        content = re.match(m1, str).group("content")
        if content == "":
            return d
        p_l = content.split(",")

        def getKey_Value(str):
            m2 = r"\"(?P<key>[0-9]+)\":(?P<value>[0-9]+)"
            r = re.search(m2, str)
            key, value = r.group("key"), int(r.group("value"))
            return (key,value)

        key2value_l = list(
            map(
                lambda s: getKey_Value(s),
                p_l
            )
        )
        key2value = dict(key2value_l)
        return key2value

    #五数概括
    #出度为0，平均值为-1的是缺失值
    #默认分析行向量
    #quantile是list
    def quantile(self, array, quantile, axis=1):
        try:
            a = np.load(open(array,"rb"))
        except:
            a = array
        a_q = np.quantile(a,quantile,axis,keepdims=True)
        return a_q
    
    #出度为0，平均值为-1的是缺失值
    def countNan(self, array):
        try:
            a = np.load(open(array,"rb"))
        except:
            a = array
        
        index_p = np.where(a[0] == 0)[0]
        c = np.shape(index_p)[0]
        return c
    
    #绘制盒图
    #以行为单位
    def plotBox(self, array,labels):
        try:
            a = np.load(open(array,"rb"))
        except:
            a = array
        a_l = a.tolist()
        fig0, ax0 = plt.subplots()
        ax0.set_title("Outdegree and Weight average")
        ax0.boxplot(a_l,labels=labels)
        return fig0,ax0
    
    #绘制直方图
    def plotHist(self, array, fig=None, ax=None, config=None):
        a = array.copy()
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        if config is not None:
            ax.hist(a,**config)
        else:
            ax.hist(a)
        return fig, ax
    
    def getMaxFreqency(self, series):
        n = series.shape[0]
        s_dict_l = [self.str2dict(series[i]) for i in range(n)]

        def plusStrategy(d1, d2):
            if d1 == {}:
                return d2
            
            for k in d2:
                if d1.get(k) is None:
                    d1[k] = 0
                d1[k] += d2[k]
            return d1
        
        s_dict = func.reduce(
            lambda d1, d2: self.mergeDictBy(d1,d2,plusStrategy),
            s_dict_l
        )

        s_p_sorted = sorted(s_dict.items(),key=itemgetter(1),reverse=True)
        return s_p_sorted


    
    def mergeDictBy(self, dict1, dict2, op):
        return op(dict1,dict2)



def analysis_it():
    csv_path = "cbg_patterns.csv"
    visitor_work_cbgs_path = "data/ndarray/visitor_work_cbgs.npy"
    if not os.path.exists(visitor_work_cbgs_path):
        dir1 = os.path.dirname(visitor_work_cbgs_path)
        if not os.path.exists(dir1):
            dir2 = os.path.dirname(dir1)
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            os.mkdir(dir1)

    P = Process(csv_path)
    df = P.read_csv()
    # print(np.array(df["census_block_group"][:10],int))
    s = df["visitor_work_cbgs"]
    newNdarray = True
    P.prepare_visitor_work_cbgs(s,visitor_work_cbgs_path,newNdarray)

    #计算五数概括
    quantile = [0,0.25,0.5,0.75,1]
    print("五数概括:")
    s_quantile = P.quantile(visitor_work_cbgs_path,quantile,1)
    print(s_quantile)
    #计算缺失值个数
    s_count_nan = P.countNan(visitor_work_cbgs_path)
    print("缺失值个数:")
    print(s_count_nan)

#-------------------------------------------------------------------------------
    #数据可视化

    #绘制盒图
    labels = ["Outdegree", "Weight Average"]
    fig0, ax0 = P.plotBox(visitor_work_cbgs_path,labels)
    # plt.show()

    #绘制直方图
    #先绘制出度的直方图
    config = {"label": "Outdegree of cbg"}
    s_nd = np.load(open(visitor_work_cbgs_path,"rb"))
    fig, axs = plt.subplots(1,2)
    ax1 = axs[0]
    ax2 = axs[1]
    fig1, ax1 = P.plotHist(s_nd[0],fig, ax1,config=config)
    
    config = {"label": "Weight average"}
    #把缺失值用0来补充
    s_nd[s_nd < 0] = 0
    fig2, ax2 = P.plotHist(s_nd[1],fig,ax2,config)
    # ax1.set_title("Outdegree and Weight Average")
    # plt.show()
    
    #剔除缺失值
    #即剔除出度为0，平均值为-1的部分
    s_nd1 = s_nd[0][s_nd[0] > 0]
    s_nd2 = s_nd[1][s_nd[1] > 0]
    
    new_visitor_work_cbgs_path = "data/ndarray/filter_visitor_work_cbgs.npy"
    if not os.path.exists(new_visitor_work_cbgs_path):
        dir1 = os.path.dirname(new_visitor_work_cbgs_path)
        if not os.path.exists(dir1):
            dir2 = os.path.dirname(dir1)
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            os.mkdir(dir1)

    s_nd = np.vstack([s_nd1,s_nd2])
    np.save(open(new_visitor_work_cbgs_path,"wb"),s_nd)

    #取频率最高的值用于修复缺失值
    #因此需要计算众数
    #这里需要重新计算
    #根据每个顶点给出的加权出度的情况
    #选择加权最高的
    new_visitor_work_cbgs_path = "data/csv/max_visitor_work_cbgs.csv"
    if not os.path.exists(new_visitor_work_cbgs_path):
        dir1 = os.path.dirname(new_visitor_work_cbgs_path)
        if not os.path.exists(dir1):
            dir2 = os.path.dirname(dir1)
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            os.mkdir(dir1)

    
    s_p_sorted = P.getMaxFreqency(s)
    value = "'{0}':{1}".format(*s_p_sorted[0])
    value = "{" + value + "}"
    # s[s == {}] = value
    df.loc[s == "{}","visitor_work_cbgs"] = value
    # df["visitor_work_cbgs"][df["visitor_work_cbgs"] == "{}"] = value
    df.to_csv(open(new_visitor_work_cbgs_path,"w",newline=""),index=False)

    #通过相似的属性来修正缺失值
    #visitor_home_cbgs与visitor_work_cbgs是相互对应的
    #可以通过这个属性来修复

    #visitor_work_cbgs可以认为是一种带权值的有向图
    #因此，存在缺失值的顶点，不存在出度，但可能存在入度，因此，通过其他指向该点的点
    #这些点可以认为是存在相似的点，故，使用这些点的平均值，用于修复缺失值


if __name__ == "__main__":
    analysis_it()
    

# %%
#分析winemag-data_first150k
import numpy as np
import pandas as pd
import re
import scipy as scp
from scipy import stats
from matplotlib import pyplot as plt
import functools as func
from operator import itemgetter, attrgetter
import os

class Process(object):
    def __init__(self, csv_path):
        self.csv_path = csv_path
    
    def read_csv(self, file_path=None):
        if file_path is not None:
            self.csv_path = file_path
        
        df = pd.read_csv(self.csv_path)
        return df

    #五数概括
    #分析某一列
    #series表示一列
    #quantile是list
    def quantile(self, series, quantile):
        # #找出最低值后给NaN的列赋予最低值-1
        # nan_value = series.min()-1
        # n = series.shape[0]
        # series.loc[pd.isna(series)] = nan_value

        #把NaN值填充为None方便处理
        series = series.fillna(method="ffill")

        array = series.to_numpy()
        return np.quantile(array,quantile)
    
    #这里的DataFrame是经过刚才那样简单处理的
    #目的是为了方便绘制Box Plot
    #后续的处理要重新读取csv文件
    def boxPlot(self, data_frame,labels):
        data_frame = data_frame.fillna(method="ffill",axis=0)
        mat = data_frame.to_numpy()
        k, n = mat.shape
        mat_l = mat.tolist()

        fig, ax = plt.subplots(1, k)
        ax1,ax2,ax3 = ax
        ax1.boxplot(mat_l[0],labels=[labels[0]])
        ax2.boxplot(mat_l[1],labels=[labels[1]])
        ax3.boxplot(mat_l[2],labels=[labels[2]])

        return ax

    
    
def analysis_it():
    csv_path = "winemag-data_first150k.csv"
    P = Process(csv_path)
    df = P.read_csv()
    # print(df[:20])
    # print(pd.isna(df.loc[1000:1100,"designation"]))

    #简单观察缺失值的情况后
    #提取存在缺失值的列
    df_ex_nan = df.loc[:, [any(pd.isna(df[x])) for x in df.columns]]
    print(df_ex_nan.columns)
    #输出发现"Index(['country', 'designation', 'price', 'province', 'region_1', 'region_2']存在缺失值
    #同时Column 1, points, price是数值属性
    #第一列未命名，这里称之为Column1
    # print(df.columns)
    #实际上为

    c1 = df.iloc[:,0]
    points = df.loc[:,"points"]
    price = df.loc[:, "price"]

    quantile_l = [0,0.25,0.5,0.75,1]
    print(P.quantile(c1,quantile_l))
    print(P.quantile(points,quantile_l))
    print(P.quantile(price,quantile_l))

    # print(c1[:10])

    #-------------------------------------------------------------
    #数据可视化
    df_nu = pd.DataFrame([c1,points,price])
    labels = ["Column 1", "points", "price"]
    # P.boxPlot(df_nu,labels)

    # plt.show()
    #--------------------------------------------------------------
    #修复缺失值
    #重新读取数据
    df = P.read_csv()
    df_ex_nan = df.loc[:, [any(pd.isna(df[x]) + (df[x] == "")) for x in df.columns]]
    # print(df_ex_nan.columns)
    #"Index(['country', 'designation', 'price', 'province', 'region_1', 'region_2']
    #这里是存在缺失值的列
    #country，desi
    
    #price是数值，可以通过众数来修复
    mode_price = stats.mode(df_ex_nan.loc[:,"price"].to_numpy())
    mode_ndarray = mode_price.mode
    n_mode = np.shape(mode_ndarray)[0]
    random_mode = np.random.randint(n_mode,size=1)[0]
    mode_p = mode_ndarray[random_mode]
    df.loc[pd.isna(df.loc[:,"price"]), "price"] = mode_p
    # print(mode_price.mode)
    print(any(pd.isna(df.loc[:,"price"])))

    #region_1和region_2之间存在关系，可以相互修复
    #province属于country，可以填写属于该country即可
    #designation和country直接去除



if __name__ == "__main__":
    analysis_it()


# %%
