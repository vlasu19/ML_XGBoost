import os

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (20,20)
'''
特征变量名称对应
'''
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

'''
设置图像横轴为百分比
'''
def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'


'''
数据预处理
'''
# x = ['年末总人口（户籍统计）', '地区生产总值', '第二产业增加值', '工业总产值', '房屋建筑竣工面积', '公路里程']
features_name = ['population', 'dist_production', 'increase_2', 'increase_3' ,'industry_prod', 'house', 'road']
ceate_feature_map(features_name)  # 设置数据label

'''
引入数据文件夹
'''
Folder_Path = r'D:\科研\CART树\数据'
os.chdir(Folder_Path)
file_list = os.listdir()

for item in file_list:
    print("==========正在处理{}==========".format(item))
    
    data = pd.read_excel(r"D:\科研\CART树\数据\{}".format(item.title()), usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    data_name = pd.read_excel(r"D:\科研\CART树\数据\{}".format(item.title()),usecols=[1])
    data.rename(columns = {'x0':'年末总人口（户籍统计）'}, inplace=True)
    data = np.array(data)
    # 删除 年末全部就业人数
    data = np.delete(data, np.s_[1], axis=0)

    train = data[0:7, :]

    '''
    STD
    '''
    label_std = data[7, :]  # 取STD为label

    X_train, X_test, y_train, y_test = train_test_split(train.T, label_std,test_size=0.2, random_state=12)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
    model.fit(X_train, y_train)

    ans = model.predict(X_test)
    print(ans)


    # x = ['年末总人口（户籍统计）', '地区生产总值', '第二产业增加值', '工业总产值', '房屋建筑竣工面积', '公路里程']
    ax1 = plt.subplot(2,2,1)
    plot_importance(model, fmap='../xgb.fmap', title="{}, Y=STD".format(item.title()), max_num_features=7,ax=ax1)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    '''
    MEAN
    '''
    label_std = data[8, :]  # 取MEAN为label
    X_train, X_test, y_train, y_test = train_test_split(train.T, label_std,test_size=0.2, random_state=12)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
    model.fit(X_train, y_train)

    ans = model.predict(X_test)
    print(ans)

    ax2 = plt.subplot(2,2,2)
    plot_importance(model, fmap='../xgb.fmap', title="{}, Y=MEAN".format(item.title()), max_num_features=7, ax=ax2)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))


    '''
    SUM
    '''
    label_std = data[9, :]  # 取MEAN为label
    X_train, X_test, y_train, y_test = train_test_split(train.T, label_std,test_size=0.2, random_state=12)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
    model.fit(X_train, y_train)

    ans = model.predict(X_test)
    print(ans)

    ax3 = plt.subplot(2,2,3)
    plot_importance(model, fmap='../xgb.fmap', title="{}, Y=SUM".format(item.title()), max_num_features=7, ax=ax3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.savefig(r"D:\科研\CART树\图像\{}.png".format(item.title()))
    plt.clf()
    plt.close()