# -*-coding:utf-8-*-

'''
Create on 2015.07.28
数据集划分与特征提取
@author: 佟俊宇
@version: 1.0

'''


import csv
import time
import MySQLdb as mysql
import math
import threading
import profile
import pandas as pd
import multiprocessing as multipro
import numpy as np
from sklearn import  linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier as GB_DT
from sklearn.ensemble import GradientBoostingRegressor as GB_RT
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from datetime import datetime
from datetime import timedelta
import types
import gc
import copy
from collections import Counter
base_dir='D:\\java_workstation\\workspace\\data\\'
'''
    连接数据库
    @param host:数据库位置一般为本地数据库
    @param user:用户名
    @param passwd:密码
    @param database:数据库名
    @param port:端口
    @return: connect 数据库连接对象 
    @return: cur    数据库游标
'''

def connect_database(host,user,passwd,database,port):
    try:
        connect = mysql.connect(host,user,passwd,database,port)
        cur = connect.cursor()
    except mysql.Error,e:
        print 'connect Error %d:%s'%(e.args[0],e.args[1])
    print 'connect database successufly'
        
    return connect,cur
'''
    断开数据连接
    @param connect:数据库连接对象
    @param cur:  数据库游标
'''
def close_database(connect,cur):
    try:
        connect.close()
        cur.close()
    except mysql.Error,e:
        print 'close database Error%d:%s'%(e.args[0],e.args[1])

'''
    创建表
    @param tablename:表名
    @param args:表列名及其类型  
    @param cur:游标
'''
def create_table(cur,tablename,*column):
    if not column.__len__:
        print 'There is no column param'
        return
    table_list = cur.execute('show tables')
    table = list(cur.fetchall())[0]
    table = list(table)
    #print table
    if table_list and tablename in table:
        next = raw_input('Do you want to drop exist table?Y/N')
        if next == 'N' or next == 'n':
            print 'No table create'
            return
        else:
            cur.execute('drop table %s'%tablename)
    try:
        tag = True
        for item in column:
            if tag:
                print item
                cur.execute('create table %s (%s %s)'%(tablename,item[0],item[1]))
                tag = False
            else:
                print item
                cur.execute('alter table %s add %s %s'%(tablename,item[0],item[1]))
    except mysql.Error,e:
        print 'create table error%d:%s'%(e.args[0],e.args[1])
        return
    print 'Create table %s success'%tablename


'''
导入数据到数据库
@param cur:数据库游标
@param filename:导入文件名
@param tablename:表名
'''
def load_data_to_database(connect,cur,filename,tablename):
    load_data_mysql = '''
    LOAD DATA INFILE '%s'
    INTO TABLE %s
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY "\r\n"
    IGNORE 1 LINES
    '''%(filename,tablename)
    #print load_data_mysql
    try:
        cur.execute(load_data_mysql)
    except mysql.Error,e:
        print 'load data into table failed %d:%s'%(e.args[0],e.args[1])
        return
    connect.commit()
    print 'load data into table successufly'
       
            
        



'''
    分割数据集
    @param filename:文件名称
    @param seperate_date:分割日期 
'''
#分割训练集和测试集
def split_data(filename,seperate_date):
    source_file = open(filename,'rb')
    out_file1 = open('train.csv','wb')
    out_file2 = open('tag.csv','wb')
    data = csv.reader(source_file)
    train = csv.writer(out_file1)
    test = csv.writer(out_file2)
    seperate_time_array = time.strptime(seperate_date,"%Y-%m-%d %H")
    seperate_time = time.mktime(seperate_time_array)
    for line in data:
        #忽略第一行
        if data.line_num == 1:
            continue
        current_time = line[5]
        current_time_array = time.strptime(current_time,"%Y-%m-%d %H")
        current_time = time.mktime(current_time_array)
        if current_time < seperate_time:
            train.writerow(line)
               
        else:
            test.writerow(line)
    source_file.close()
    out_file1.close()
    out_file2.close()
    print 'splite file successufly'



'''
    生成验证集
    @param filename:输入文件名
    @param outfile:输出文件名
    @param connect: 数据库连接对象: 
    @param cur:数据库游标对象 
    
'''       
def generate_validate_set(filename,tag_data,ratio):
    feature_data = pd.read_csv(filename)
    validate_data = tag_data[tag_data.behavior_type == 4]
    validate_data = validate_data[['user_id','item_id','behavior_type']]
    validate_data = validate_data.drop_duplicates()
    #print len(validate_data)    
    #选取最近一周有过点击交互的
    result = pd.merge(feature_data[['user_id','item_id']],validate_data,on=['user_id','item_id'],how='left')
    del validate_data
    #gc.collect()
    #填充缺失值为0
    result = result.fillna(0)
    result['behavior_type'] = result['behavior_type']/4
    feature_data['behavior_type'] = result['behavior_type']
    #feature_data = feature_data.set_index(['user_id','item_id'])
    del result
    #gc.collect()
    #规范化数据在0-1之间
    #f = lambda x:x/(x.max()-x.min())
    #feature_data = feature_data.apply(f)
    #选取正负数据集，比例为1:10
    positive_set = feature_data[feature_data.behavior_type > 0]
    positive_len = len(positive_set)
    print "positive_set len %d"%positive_len
    negative_sub_set = pd.DataFrame(columns=feature_data.columns)
    negative_set = feature_data[feature_data.behavior_type == 0]
    length = len(negative_set)
    
    count = 0
    index_set = set()
    #negative_sub_set = negative_sub_set.append(positive_set)
    #negative_sub_set = negative_sub_set.append(positive_set)
    #print 'debug 1  negative_sub_set len :%d'%len(negative_sub_set)
    for i in xrange(ratio*positive_len):
        index = 0
        while True:
            index = np.random.random_integers(0,length-1)
            if index not in index_set:
                index_set.add(index)
                break
        negative_sub_set = negative_sub_set.append(negative_set.iloc[index])
    #negative_sub_set = negative_sub_set.append(positive_set)
    #print 'debug 2  negative_sub_set len :%d'%len(negative_sub_set)
    print 'generate_validate_set successulf'
    return negative_sub_set.append(positive_set)
    
 
def generate_test_set(feature_data): 
    result = feature_data.set_index(['user_id','item_id'])
    #规范化数据在0-1之间
    f = lambda x:x/(x.max()-x.min())
    #result = result.apply(preprocessing.scale)
    return result    
    
def timeprocess(time_tuple,current_time):
    current_time_array = time.strptime(current_time,"%Y-%m-%d %H")
    current_time = time.mktime(current_time_array)
    time_sum = 0
    near_time = 10000
    for time_item in time_tuple:
        time_array = time.strptime(time_item,"%Y-%m-%d %H")
        last_time = time.mktime(time_array)
        time_diff = (current_time - last_time)/3600.0
        if time_diff < near_time:
            near_time = time_diff
        time_sum += time_diff
    return (time_sum/len(time_tuple),near_time)
        
        


            
'''
提取用户特征,用户交互统计特征和交互转化率
@param filename:输入文件名
@param outfile:输出文件名
@param time_point:分割时间点
   
'''
def generate_user_feature1(group_data,outfile,time_point=None):   
    file = csv.writer(open(outfile,'wb'))
    featurename = ['user_id','user_click_count','user_collect_count','user_cart_count','user_buy_count',\
                   'user_click_buy_rate','user_collect_buy_rate','user_cart_buy_rate',\
                   'u_last1day_click_buy_rate','u_last1day_collect_buy_rate','u_last1day_cart_buy_rate',\
                   'u_last3day_click_buy_rate','u_last3day_collect_buy_rate','u_last3day_cart_buy_rate',\
                   'u_last7day_click_buy_rate','u_last7day_collect_buy_rate','u_last7day_cart_buy_rate']
    file.writerow(featurename)
    feature = []
    for user,temp_data in group_data:
        feature.append(user)
        #抽取次数特征
        for i  in xrange(4):
            count = len(temp_data[(temp_data.behavior_type == i+1)])
            feature.append(count)
        #抽取比例特征
        for behavior in [1,2,3]:
            if feature[4] == 0:
                feature.append(0)
            else:
                if feature[behavior] == 0:
                    feature.append(0)
                else:
                    feature.append(feature[4]/(feature[behavior]+0.0))
                    
        for day in [1,3,7]:
            
            for behavior in [1,2,3]:
                item_count = len(temp_data[(temp_data.behavior_type == behavior) & (temp_data.time < 24 * day)])
                item_buy_count = len(temp_data[(temp_data.behavior_type == 4) & (temp_data.time < 24 * day)])
                
                if item_buy_count == 0:
                    feature.append(0)
                else:
                    if item_count == 0:
                        feature.append(0)
                    else:
                        feature.append(item_buy_count/(item_count+0.0))
        
        file.writerow(feature)
        #print feature
        feature = []
    print "generate_feature1 end"
def generate_user_feature2(group_data,outfile,time_point):
    #预测时间点前一天
    point = time_point
    file = csv.writer(open(outfile,'wb'))
    featurename = ['user_id','user_least_click_hour','user_least_collect_hour','user_least_cart_hour','user_least_buy_hour']
    file.writerow(featurename)
    feature = []
    for user,data in group_data:
        feature.append(user)
        for i in range(1,5):
            temp_data = data[(data.behavior_type == i)]
            
            if len(temp_data) != 0:
                hour = int(temp_data['time'].min())
            else:
                hour = 720
            feature.append(hour)
        
        file.writerow(feature)
        #print feature
        feature = []
    print 'generate_user_feature2 end'


def generate_user_feature3(group_data,outfile,time_point):
    #预测点前五天
    #stamp = datetime.strptime(time_point,'%Y-%m-%d %H')
     
    file = csv.writer(open(outfile,'wb'))
    featurename = ['user_id','user_last_1_day_click_count','user_last_1_day_collect_count','user_last_1_day_cart_count','user_last_1_day_buy_count',\
                   'user_click_brand_count_for_1_day','user_collect_brand_count_for_1_day','user_cart_brand_count_for_1_day','user_buy_brand_count_for_1_day',\
                   'user_last_3_day_click_count','user_last_3_day_collect_count','user_last_3_day_cart_count','user_last_3_day_buy_count',\
                   'user_last_5_day_click_count','user_last_5_day_collect_count','user_last_5_day_cart_count','user_last_5_day_buy_count',\
                   ]
    file.writerow(featurename)
    feature = []
    for user,data in group_data:
        feature.append(user)
        for i in [1,3,5]:
            #point = (stamp - timedelta(i)).strftime('%Y-%m-%d %H')
            point = 24 * i
            temp_data = data[data.time < point]
            for j in xrange(4):
                count = len(temp_data[(temp_data.behavior_type == j+1)])
                feature.append(count)
            if i == 1:
                for n in range(1,5):
                    count = len(temp_data[(temp_data.behavior_type == n )]['item_id'].unique())
                    feature.append(count)
                    
        file.writerow(feature)
        #print feature
        feature = []
    print 'generate_user_featuer3 end'
 
'''
    用户商品转化率
''' 
def generate_user_feature4(group_data,outfile,time_point):
    #预测时间点前一天
    point = time_point
    file = csv.writer(open(outfile,'wb'))
    featurename = ['user_id','u_click_item_buy_rate','u_collect_item_buy_rate','u_cart_item_buy_rate',\
                   'u_1day_click_item_buy_rate','u_1day_collect_item_buy_rate','u_1day_cart_item_buy_rate',\
                   'u_3day_click_item_buy_rate','u_3day_collect_item_buy_rate','u_3day_cart_item_buy_rate',\
                   'u_7day_click_item_buy_rate','u_7day_collect_item_buy_rate','u_7day_cart_item_buy_rate',\
                   ]
    file.writerow(featurename)
    feature = []
    for user,data in group_data:
        feature.append(user)
        for date in [28,1,3,7]:
            
            for behavior in [1,2,3]:
                
                item_buy = data[(data.behavior_type == 4) & (data.time < 24 * date)][['item_id']].drop_duplicates()
                
                item_behavior = data[(data.behavior_type == behavior) & (data.time < 24 * date)][['item_id']].drop_duplicates()
                
                if behavior == 1:
                    if len(item_buy) == 0:
                        feature.append(0)
                    else:
                        if len(item_behavior) == 0:
                            feature.append(1)
                        else:
                            feature.append(len(item_buy)/(len(item_behavior)+0.0))
                            
                else:
                    if len(item_behavior) == 0:
                        feature.append(0)
                    else:
                        temp = pd.merge(item_behavior,item_buy,on=['item_id'],how='inner')
                        if len(temp) == 0:
                            feature.append(0)
                        else:
                            feature.append(len(item_buy)/(len(temp)+0.0))
        #print feature
        file.writerow(feature)
        feature = []
        
    print 'generate user feature 4 end!'            
        
def generate_user_feature5(group_data,outfile,time_point):
    point = time_point
    file = csv.writer(open(outfile,'wb'))
    featurename = ['user_id','uf_clickitems_10','uf_clickcate_10','uf_collectitems_10','uf_collectcate_10','uf_cartitems_10','uf_cartcate_10',\
                   'uf_buyitems_10','uf_buycate_10','uf_click_cate_buy_rate_10','uf_click_item_buy_rate_10','uf_click_item_cickitems_7',\
                   'uf_clickcate_7','uf_collectitems_7','uf_collectcate_7','uf_cartitems_7','uf_cartcate_7','uf_buyitems_7','uf_buycate_7',\
                   'uf_click_cate_buy_rate_7','uf_click_item_buy_rate_7','uf_click_item_cickitems_5','uf_clickcate_5','uf_collectitems_5',\
                   'uf_collectcate_5','uf_cartitems_5','uf_cartcate_5','uf_buyitems_5','uf_buycate_5','uf_click_cate_buy_rate_5','uf_click_item_buy_rate_5',\
                   'uf_click_item_cickitems_3','uf_clickcate_3','uf_collectitems_3','uf_collectcate_3','uf_cartitems_3','uf_cartcate_3',\
                   'uf_buyitems_3','uf_buycate_3','uf_click_cate_buy_rate_3','uf_click_item_buy_rate_3',\
                   'uf_click_item_cickitems_2','uf_clickcate_2','uf_collectitems_2','uf_collectcate_2','uf_cartitems_2','uf_cartcate_2',\
                   'uf_buyitems_2','uf_buycate_2','uf_click_cate_buy_rate_2','uf_click_item_buy_rate_2',\
                   'uf_click_item_cickitems_1','uf_clickcate_5','uf_collectitems_1','uf_collectcate_1','uf_cartitems_1','uf_cartcate_1',\
                   'uf_buyitems_1','uf_buycate_1','uf_click_cate_buy_rate_1','uf_click_item_buy_rate_1'
                   ]
    file.writerow(featurename)
    feature = []
    for user,data in group_data:
        feature.append(user)
        for day in [10,7,5,3,2,1]:
            temp_data = data[data.time < (day * 24 + 1)]
            for behavior in [1,2,3,4]:
                item_count = len(temp_data[temp_data.behavior_type == behavior]['item_id'].drop_duplicates())
                categ_count = len(temp_data[temp_data.behavior_type == behavior]['item_category'].drop_duplicates())
                feature.append(item_count)
                feature.append(categ_count)
            if feature[2] == 0:
                click_cate_buy_rate = 0 
            else:
                click_cate_buy_rate = float(feature[8])/feature[2] 
            if feature[1] == 0:
                click_item_buy_rate = 0
            else:
                click_item_buy_rate = float(feature[7])/feature[1]
            feature.append(click_cate_buy_rate)
            feature.append(click_item_buy_rate) 
        
        file.writerow(feature)
        feature = []
    print " generate user feature 5 end"
 
def generate_user_feature6(group_data,outfile,time_point):
    #point = time_point
    file = csv.writer(open(outfile,'wb'))  
    featurename = ['user_id','uf_active_31','uf_buy_active_31','uf_active_buy_rate_31',\
                   'uf_active_10','uf_buy_active_10','uf_active_buy_rate_10',\
                   'uf_active_7','uf_buy_active_7','uf_active_buy_rate_7',\
                   'uf_active_3','uf_buy_active_3','uf_active_buy_rate_3',\
                   'uf_active_2','uf_buy_active_2','uf_active_buy_rate_2',\
                   'uf_active_1','uf_buy_active_1','uf_active_buy_rate_1'
                   ]        
    file.writerow(featurename)
    feature = []
    func = lambda x:int(x/24)
    for user,data in group_data:
        feature.append(user)
        for day in [31,10,7,3,2,1]:
            temp_data = data[data.time < (24*day+1)]
            time_list = temp_data['time'].apply(func)
            active_days = len(time_list.drop_duplicates())
            #print 'day %d'%day
            #print 'active days %d'%active_days
            buy_data = temp_data[temp_data.behavior_type == 4]
            time_list = buy_data['time'].apply(func)
            active_buy_days = len(time_list.drop_duplicates())
            if active_days == 0:
                active_buy_rate = 0.0
            else:
                active_buy_rate = float(active_buy_days)/active_days
            feature.extend([active_days/(day+0.0),active_buy_days/(day+0.0),active_buy_rate])
        #print feature
        file.writerow(feature)
        feature = []
    print "generate user feature 6 end"

'''
提取用户位置信息特征
'''


def generate_user_geo_feature(pre_geo, data):
    #筛选出包含位置信息的记录
    geo_data = data[data.geo_1 > 0]
    gb = geo_data.groupby(pre_geo)
    # 对每个地区别提取特征、每个用户的特征除以本地区的该特征的平均值
    func = lambda x: 1 / (1+np.e**(x.mean()-x))
    result_data = pd.DataFrame(columns=data.columns)

    temp_file_name = base_dir + 'temp.csv'
    # 存放统计数据文件名
    '''
    text_filename = base_dir + pre_geo + '.csv'
    file_object = csv.writer(open(text_filename, 'wb'))
    line = ['geo', 'user_numbers', 'uf_active_31', 'uf_buy_active_31', 'uf_active_buy_rate_31',
            'uf_active_10', 'uf_buy_active_10', 'uf_active_buy_rate_10',
            'uf_active_7', 'uf_buy_active_7', 'uf_active_buy_rate_7',
            'uf_active_3', 'uf_buy_active_3', 'uf_active_buy_rate_3',
            'uf_active_2', 'uf_buy_active_2', 'uf_active_buy_rate_2',
            'uf_active_1', 'uf_buy_active_1', 'uf_active_buy_rate_1']
    file_object.writerow(line)
    '''

    for geo, group_data in gb:
        # 如果该地区的用户数小于100则不对该地区做区域性处理
        if len(group_data['user_id'].drop_duplicates()) < 100:
            continue

        line = list()
        line.append(geo)
        user_gb = group_data.groupby('user_id')
        # 生成该地区的用户特征
        generate_user_feature6(user_gb, temp_file_name, '')
        feature_data = pd.read_csv(temp_file_name)
        # 读取统计信息
        line.append(len(feature_data))
        for column in feature_data.columns:
            if column == 'user_id':
                continue
            else:
                line.append(feature_data[column].mean())
        # file_object.writerow(line)
        modify_data = feature_data.apply(func)
        result_data = result_data.append(modify_data)
        # print result_data.describe()
        pass
    print 'user geo number %d\n' % len(result_data)
    # 对不包含位置信息的用户交互记录做特殊处理
    # 使用全局数据计算不包含位置信息用户的特征
    user_gb = data.groupby('user_id')
    generate_user_feature6(user_gb, temp_file_name, '')
    feature_data = pd.read_csv(temp_file_name)
    # ###########################统计全部用户的位置信息##################################
    line = list()
    line.append('all_user')
    line.append(len(feature_data))
    for column in feature_data.columns:
        if column == 'user_id':
            continue
        else:
            line.append(feature_data[column].mean())
    # file_object.writerow(line)
    # ###############################################################################
    no_geo_feature = feature_data.apply(func)
    result_data = result_data.append(no_geo_feature[~no_geo_feature.user_id.isin(result_data['user_id'])])
    print 'all user number %d \n' % len(result_data)

    return result_data
    pass


'''
根据用户位置信息生成位置信息特征
'''


def generate_geo_feature(input_filename, train_or_test):
    # 读入包含位置信息的文件
    data = pd.read_csv(input_filename)
    print 'user number %d' % len(data['user_id'].drop_duplicates())
    if train_or_test == 'train':
        data = data[data.time > 24]
        data['time'] = data['time'].apply(lambda x: x - 24)
    else:
        data = data[data.time > 0]

    # 根据不同长度的前缀分割数据
    for geo in ['geo_2']:
        # 用户位置信息特征输出文件名
        user_geo_feature_name = base_dir + '\\train_test_set\\%s_user_geo_feature%s.csv' \
                                           % (train_or_test, geo)
        user_feature_data = generate_user_geo_feature(geo, data)
        print 'user geo feature output filename %s' % user_geo_feature_name
        # user_feature_data.to_csv(user_geo_feature_name, index=False)
        pass
    pass
        
            
            
            
'''
    @param time_string: 日期字符串
    @return: time_list: [year,month,day] 
''' 
def time_to_strlist(time_string):
    time_list = time_string.split(" ")
    time_str = time_list[0]
    time_list = time_str.split("-")
    return time_list
     
        
def generate_user_item_feature1(group_data,data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name =['user_id','item_id','buy_count','least_buy_hour_count',\
                   'cart_count','least_cart_day_count','colect_count'\
                   ,'least_collect_day_count','click_count','least_click_day_count']
    time_count = 0
    feature = []
    file.writerow(feature_name)
    for ui,temp_data in group_data:
        feature.extend(list(ui))
        for i in xrange(4,0,-1):
            temp_result = temp_data[(temp_data.behavior_type == i)]
            count = len(temp_result)
            #用户点击、收藏、购买次数
            feature.append(count)
            if len(temp_result):
                feature.append(int(temp_result['time'].min()))
            else :
                feature.append(720)       
        file.writerow(feature)
        #print feature
        feature =[]
    print 'process1 end'   
        
def generate_user_item_feature2(group_data,data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['user_id','item_id','operator_day_count']
    feature = []
    file.writerow(feature_name)
    for ui ,result in group_data:
        feature.extend(list(ui))
        result = result['time']/24
        result = [int(time) for time in result]
        result = set(result)
        day_count = len(result)
        feature.append(day_count)
        #print feature
        file.writerow(feature)
        feature = []
    print 'process2 end'

def generate_user_item_feature3(group_data,data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['user_id','item_id','user_buy_brand_rate','user_click_brand_rate',\
                   'click_buy_rate','click_2_count','buy_2_count','brand_is_bought_rate',\
                   'brand_is_clicked_rate','click_delete_user_avge_rate'\
                   ,'buy_delete_user_avg_rate']
    feature = []
    file.writerow(feature_name)
    #根据用户分组和根据商品分组
    user_group = data.groupby('user_id')
    item_group = data.groupby('item_id')
    for ui,gdata in group_data:
        feature.extend(list(ui))
        #获取用户u的所有交互记录
        user_temp_data = user_group.get_group(ui[0])
        #用户u对商品i的购买次数
        user_item_buy_count = len(gdata[gdata.behavior_type == 4])
        #用户u的购买次数
        user_buy_count = len(user_temp_data[user_temp_data.behavior_type == 4])
        #用户U对商品i的点击次数
        user_item_click_count = len(gdata[(gdata.behavior_type == 1)])
        #用户u的点击总数
        user_click_count = len(user_temp_data[user_temp_data.behavior_type == 1])
        #用户u的对商品i购买率
        user_buy_brand_rate = user_item_buy_count /(user_buy_count + 1.0)
        #用户u对商品i点击率
        user_click_brand_rate = user_item_click_count/(user_click_count + 1.0)
        #用户u对商品i的点击购买率
        click_buy_rate = user_item_buy_count/(user_item_click_count + 1.0)
        
        click_2_count = math.sqrt(user_item_click_count)
        
        buy_2_count = math.sqrt(user_item_buy_count)
        
        feature.append(user_buy_brand_rate)
        feature.append(user_click_brand_rate)
        feature.append(click_buy_rate)
        feature.append(click_2_count)
        feature.append(buy_2_count)
        
        item_temp_data = item_group.get_group(ui[1])
        item_buy_count = len(item_temp_data[(item_temp_data.behavior_type == 4)])
        brand_is_bought_rate = user_item_buy_count / (item_buy_count + 1.0)
     
        item_click_count = len(item_temp_data[(item_temp_data.behavior_type == 1)])
        brand_is_clicked_rate = user_item_click_count / (item_click_count + 1.0)
        feature.append(brand_is_bought_rate)
        feature.append(brand_is_clicked_rate)
     
        user_item_count = len(user_temp_data[(user_temp_data.behavior_type == 1)]['item_id'].unique())
        #print user_item_count
        click_delete_user_avge_rate = math.fabs((user_item_click_count - (user_click_count) / \
                                                 (user_item_count+1.0)))/((user_click_count+1.0) / (user_item_count+1.0))
  
        user_item_buy_band_count = len(user_temp_data[(user_temp_data.behavior_type == 4)]['item_id'].unique())
        buy_delete_user_avge_rate = math.fabs((user_item_buy_count -\
                                                (user_buy_count)/(user_item_buy_band_count + 1.0))\
                                              ) / ((user_buy_count + 1.0) / (user_item_buy_band_count + 1.0))
        feature.append(click_delete_user_avge_rate)
        feature.append(buy_delete_user_avge_rate)
        #print feature
        file.writerow(feature)
        feature = []
    print 'process3 end'    
def generate_user_item_feature4(group_data,data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['user_id','item_id','last_1_day_click_count','last_1_day_collect_count','last_1_day_cart_count','last_1_day_buy_count',\
                    'last_2_day_click_count','last_2_day_collect_count','last_2_day_cart_count','last_2_day_buy_count',\
                    'last_3_day_click_count','last_3_day_collect_count','last_3_day_cart_count','last_3_day_buy_count',\
                    '3-day_click_count','3-day_collect_count','3-day_cart_count','3-day_buy_count',\
                    '7-day_click_count','7-day_collect_count','7-day_cart_count','7-day_buy_count',\
                    'time_feature1','time_feature2','time_feature3','time_feature4','time_feature5','time_feature6',\
                    'time_feature7','time_feature8','time_feature9','time_feature10','time_feature11','time_feature12',\
                    'time_feature13','time_feature14','time_feature15','time_feature16','time_feature17','time_feature18',\
                    'time_feature19','time_feature20','time_feature21','time_feature22','time_feature23','time_feature24',\
                    'time_feature25','time_feature26','time_feature27','time_feature28','time_feature29','time_feature30',\
                    'time_feature31','time_feature32','time_feature33','time_feature34','time_feature35','time_feature36',\
                    'time_feature37','time_feature38','time_feature39','time_feature40','time_feature41','time_feature42',\
                    'time_feature43','time_feature44','time_feature45','time_feature46','time_feature47','time_feature48',\
                    ]
    feature = []
    file.writerow(feature_name)
    for ui,data in group_data:
        feature.extend(list(ui))
        #第一天、第二天、第三天
        for day in [1,2,3]:
            start_time = (day-1)*24
            end_time = day*24
            click_count = len(data[(data.behavior_type == 1) &(start_time < data.time) &(data.time < end_time)])
            collect_count = len(data[(data.behavior_type == 2) &(start_time < data.time) &(data.time < end_time)])
            cart_count = len(data[(data.behavior_type == 3) &(start_time < data.time) &(data.time < end_time)])
            buy_count = len(data[(data.behavior_type == 4) &(start_time < data.time) &(data.time < end_time)])
            feature.append(click_count)
            feature.append(collect_count)
            feature.append(cart_count)
            feature.append(buy_count)
        #三天、一周
        for day in [3,7]:
            click_count = len(data[(data.behavior_type == 1) & (data.time < day*24)])
            collect_count = len(data[(data.behavior_type == 2) & (data.time < day*24)])
            cart_count = len(data[(data.behavior_type == 3) & (data.time < day*24)])
            buy_count = len(data[(data.behavior_type == 4) & (data.time < day*24)])
            feature.append(click_count)
            feature.append(collect_count)
            feature.append(cart_count)
            feature.append(buy_count)
        one_day_data = data[data.time < 25]
        for i in range(1,25):
            for behavior in range(1,5):
                count = len(one_day_data[(one_day_data.behavior_type == behavior) & (one_day_data.time < i)])
                feature.append(count)
        file.writerow(feature)
        #print feature
        feature = []
    print 'process4 end'

def getfeature(ten_data):
    feature = []
    if len(ten_data) == 0:
        return [0,0,0,0,0,0,0.0]
    ten_click = len(ten_data[ten_data.behavior_type == 1])
    ten_collect = len(ten_data[ten_data.behavior_type == 2])
    ten_cart = len(ten_data[ten_data.behavior_type == 3])
    ten_buy = len(ten_data[ten_data.behavior_type == 4])
    result = ten_data['time']/24
    result = [int(time) for time in result]
    result = set(result)
    ten_active_day_count = len(result)
   
    ten_buy_data = ten_data[ten_data.behavior_type == 4]
    if len(ten_buy_data) > 0:
        result = ten_buy_data['time']/24
        result = [int(time) for time in result]
        result = set(result)
        ten_buy_days = len(result)
    else:
        ten_buy_days = 0
    feature.append(ten_click)
    feature.append(ten_collect)
    feature.append(ten_cart)
    feature.append(ten_buy)
    feature.append(ten_active_day_count)
    feature.append(ten_buy_days)
    feature.append(float(ten_buy_days)/ten_active_day_count)
    return feature
                    
def  generate_user_item_feature5(group_data,data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['user_id','item_id','uif_gapday','uif_gaphour','uif_opearte_hour','uif_click_10','uif_collect_10',\
                    'uif_cart_10','uif_buy_10','uif_active_day_10','uif_buy_days_10','uif_buydays_division_activedays_10',\
                    'uif_click_5','uif_collect_5','uif_cart_5','uif_buy_5','uif_active_days_5','uif_buy_days_5',\
                    'uif_buydays_dividsion_activedays_5','uif_click_3','uif_collect_3','uif_cart_3','uif_buy_3',\
                    'uif_active_days_3','uif_buy_days_3','uif_buydays_dividsion_activedays_3',\
                    'uif_click_2','uif_collect_2','uif_cart_2','uif_buy_2','uif_active_days_2','uif_buy_days_2',\
                    'uif_buydays_dividsion_activedays_2','uif_maxclick_hour_1','uif_max_collect_hour_1',\
                    'uif_max_cart_hour_1','uif_max_buy_hour_1','uif_start_hour_1','uif_end_hour_1','uif_behavior_hour_1','uif_click_division_behaviorhour_1']
    feature = []
    file.writerow(feature_name)
    #根据用户分组和根据商品分组
    user_group = data.groupby('user_id')
    item_group = data.groupby('item_id')
    for ui,gdata in group_data:  
        feature.extend(list(ui))
        
        min_hour = gdata['time'].min()
        min_day = int(gdata['time'].min()/24)
        max_hour = gdata['time'].max()
        feature.append(min_day)
        feature.append(min_hour)
        feature.append(max_hour-min_hour)
        
        ten_data = gdata[gdata.time < 241]
        ten_feature = getfeature(ten_data)
        feature.extend(ten_feature)
        
        five_data = ten_data[ten_data.time < 121]
        five_feature = getfeature(five_data)
        feature.extend(five_feature)
        
        three_data = five_data[five_data.time < 73]
        three_feature = getfeature(three_data)
        feature.extend(three_feature)
        
        two_data = three_data[three_data.time < 49]
        two_feature = getfeature(two_data)
        feature.extend(two_feature)
        
        one_data = two_data[two_data.time < 25]
        if len(one_data) == 0:
            feature.extend([0,0,0,0,0,0,0,0.0])
        else:
            for i in range(1,5):
                behavior_time = one_data[one_data.behavior_type == i]['time']
                if len(behavior_time) == 0:
                    feature.append(0)
                    continue
                max_behavior_hour = Counter(behavior_time).most_common()
                feature.append(24-max_behavior_hour[0][0])
            begin = 24 - one_data['time'].max()
            end = 24 - one_data['time'].min()
            feature.append(begin)
            feature.append(end)
            feature.append(end - begin)
            click_count = len(one_data[one_data.behavior_type == 1])
            if end - begin == 0:
                feature.append(float(click_count))
            else:
                feature.append(float(click_count)/click_count)
            
        file.writerow(feature)
        feature = []
    print 'process5 end'
 
           
'''
提取商品特征
@param connect:数据库连接对象
@param cur:数据库游标  
@param time_point:当前的时间 
@param outfilename:输出文件名 
'''           
def generate_item_feature(data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['item_id','all_click_count','ave_click_count','all_collect_count','ave_collect_count','all_cart_count',\
                    'ave_cart_count','all_buy_count','ave_buy_count','click_buy_rate','collect_buy_rate','cart_buy_rate',\
                    '1_day_click_count','1_day_collect_count',\
                    '1_day_cart_count','1_day_buy_count','1_day_user_click_count','1_day_user_collect_count','1_day_user_cart_count',\
                    '1_day_user_buy_count','3_day_click_count','3_day_collect_count','3_day_cart_count','3_day_buy_count',\
                    '3_day_user_click_count','3_day_user_collect_count','3_day_user_cart_count','3_day_user_buy_count',\
                    '5_day_click_count','5_day_collect_count','5_day_cart_count','5_day_buy_count',\
                    '5_day_user_click_count','5_day_user_collect_count','5_day_user_cart_count','5_day_user_buy_count',\
                    '7_day_click_count','7_day_collect_count','7_day_cart_count','7_day_buy_count',\
                    '7_day_user_click_count','7_day_user_collect_count','7_day_user_cart_count','7_day_user_buy_count',\
                    'hot_level']
    file.writerow(feature_name)
    feature = []
    group_data = data.groupby('item_id')
    for item , item_temp_data in group_data:
        #添加商品id
        feature.append(item)
        #对item交互的用户数
        user_count = len(item_temp_data['user_id'].drop_duplicates())
        
        #抽取不同行为的操作次数
        for i  in range(4):
            count = len(item_temp_data[(item_temp_data.behavior_type == i+1)])
            #交互次数
            feature.append(count)
            #平均每个用户的交互次数
            feature.append(count/(user_count+0.1))
        #抽取比例特征
        feature.append(feature[7]/(feature[1]+1.0))
        feature.append(feature[7]/(feature[3]+1.0))
        feature.append(feature[7]/(feature[5]+1.0))
        
        #提取时间层次特征
        for j in range(1,8,2):
            #stamp = datetime.strptime(time_point,'%Y-%m-%d %H')
            #point = (stamp - timedelta(j)).strftime('%Y-%m-%d %H')
            #temp_data = item_temp_data[item_temp_data.time > point]
            temp_data = item_temp_data[item_temp_data.time/24 < j]
            for i in [1,2,3,4]:
                different_behavior_data = temp_data[(temp_data.behavior_type == i)]
                count = len(different_behavior_data)
                feature.append(count)
                user_count = len(different_behavior_data['user_id'].drop_duplicates())
                feature.append(user_count)
                
            
        #提取商品热度特征
        hot_level = feature[1]*0.1 + feature[3] + feature[5] + feature[7]*5
        feature.append(hot_level)
        file.writerow(feature)
        #print feature
        feature = []    
    #print feature
    print 'Generate item feature end!'

'''
商品流行度特征
'''    
def generate_item_feature2(data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['item_id','first_day_click_pro','first_day_collect_pro','first_day_cart_pro','first_day_buy_pro',\
                    'second_day_click_pro','second_day_collect_pro','second_day_cart_pro','second_day_buy_pro',\
                    'third_day_click_pro','third_day_collect_pro','third_day_cart_pro','third_day_buy_pro',\
                    'three_day_click_pro','three_day_collect_pro','three_day_cart_pro','three_day_buy_pro'
                    ]
    file.writerow(feature_name)
    feature = []
    group_data = data.groupby('item_id')
    for item , item_temp_data in group_data:
        feature.append(item)
        click_count = len(item_temp_data[item_temp_data.behavior_type==1])/30
        collect_count = len(item_temp_data[item_temp_data.behavior_type==2])/30
        cart_count = len(item_temp_data[item_temp_data.behavior_type==3])/30
        buy_count = len(item_temp_data[item_temp_data.behavior_type==4])/30
        
        behavior_type_count = [click_count,collect_count,cart_count,buy_count]
        
        three_day_data = item_temp_data[item_temp_data.time < 73]
        
        third_day_data = three_day_data[three_day_data.time > 47]
        second_day_data = three_day_data[(three_day_data.time > 24) & (three_day_data.time < 48)]
        first_day_data = three_day_data[three_day_data.time < 25]
        
        for i in range(1,5):
            pro_feature = len(first_day_data[first_day_data.behavior_type == i])/(behavior_type_count[i-1]+0.1)
            feature.append(pro_feature)
        
        for i in range(1,5):
            pro_feature = len(second_day_data[second_day_data.behavior_type == i])/(behavior_type_count[i-1]+0.1)
            feature.append(pro_feature)
            
        for i in range(1,5):
            pro_feature = len(third_day_data[third_day_data.behavior_type == i])/(behavior_type_count[i-1]+0.1)
            feature.append(pro_feature)
        for i in range(1,5):
            pro_feature = len(three_day_data[three_day_data.behavior_type == i])/(behavior_type_count[i-1]+0.1)
            feature.append(pro_feature)
        
        file.writerow(feature)
        #print feature
        feature = []   
    print 'Generate item feature end!'   
    
#商品购买的用户转化率特征   
def generate_item_feature3(data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['item_id','i_user_click_buy_rate','i_last1day_user_click_buy_rate','i_last3day_user_click_buy_rate','i_last7day_user_click_buy_rate',\
                    'i_user_collect_buy_rate','i_last1day_user_collect_buy_rate','i_last3day_user_collect_buy_rate','i_last7day_user_collect_buy_rate',\
                    'i_user_cart_buy_rate','i_last1day_user_cart_buy_rate','i_last3day_user_cart_buy_rate','i_last7day_user_collect_buy_rate'
                    ]
    file.writerow(feature_name)
    feature = []
    group_data = data.groupby('item_id')
    for item , item_temp_data in group_data:
        feature.append(item)
        for i in [1,2,3]:#click collect cart
            for t in [28,1,3,7]:#前n天
                user_count = len(item_temp_data[(item_temp_data.behavior_type == i) & (item_temp_data.time < 24 * t)]['user_id'].drop_duplicates())
                user_buy_count = len(item_temp_data[(item_temp_data.behavior_type == 4) & (item_temp_data.time < 24 * t) ]['user_id'].drop_duplicates())
                if user_buy_count == 0:
                    feature.append(0)
                else:
                    if user_count == 0:
                        feature.append(0)
                    else:
                        feature.append(user_buy_count/(user_count+0.0))
        #print feature
        file.writerow(feature)
        feature = []
                
    print 'Generate item feature3 end!'         
            


'''
    data:输入数据
    outfilename:输出文件名
    time_point:时间节点
'''
def generate_user_category_feature(data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['usre_id','item_category','uc_1day_click_count','uc_1day_collect_count','uc_1day_cart_count','uc_1day_buy_count',\
                    'uc_1day_click_item_count','uc_1day_collect_item_count','uc_1day_cart_item_count','uc_1day_buy_item_count',\
                    'uc_3day_click_count','uc_3day_collect_count','uc_3day_cart_count','uc_3day_buy_count',\
                    'uc_3day_click_item_count','uc_3day_collect_item_count','uc_3day_cart_item_count','uc_3day_buy_item_count',\
                    ]
    file.writerow(feature_name)
    feature = []
    uc_group = data.groupby(['user_id','item_category'])
    for uc,group_data in uc_group:
        feature.a
    pass
'''
    data:输入数据集
    outfilename：输出文件名
    time_point：时间节点
'''
def generate_category_feature(data,outfilename,time_point):
    file = csv.writer(open(outfilename,'wb'))
    feature_name = ['c_click_count','c_cart_count','c_buy_count','c_click_buy_rate','c_cart_buy_rate',\
                    'c_user_count','c_user_click_count','c_user_cart_count','c_user_buy_count','c_user_click_buy_rate',\
                    'c_user_cart_buy_rate','c_last3day_user_click_buy_rate','c_last7day_user_click_buy_rate','item_id','i_c_click_count_rate','i_c_cart_count_rate','i_c_buy_count_rate','i_c_clcik_buy_rate_rate',\
                    'i_c_cart_buy_rate_rate','i_c_user_count_rate','i_c_user_click_count_rate','i_c_user_cart_count_rate',\
                    'i_c_user_buy_count_rate','i_c_user_click_buy_rate_rate','i_c_user_cart_buy_rate_rate',\
                    ]
    file.writerow(feature_name)
    feature = []
    group_data = data.groupby('item_category')
    for category,group in group_data:
        #行为统计特征
        c_click_count = len(group[group.behavior_type==1])
        feature.append(c_click_count)
        c_cart_count = len(group[group.behavior_type == 3])
        feature.append(c_cart_count)
        c_buy_count = len(group[group.behavior_type == 4])
        feature.append(c_buy_count)
        #行为转化率特征
        feature.append((c_buy_count+0.0)/(c_click_count+1))
        feature.append((c_buy_count+0.0)/(c_cart_count+1))
        #用户统计特征
        c_user_count = len(group['user_id'].drop_duplicates())
        feature.append(c_user_count)
        c_user_click_count = len(group[group.behavior_type==1]['user_id'].drop_duplicates())
        feature.append(c_user_click_count)
        c_user_cart_count = len(group[group.behavior_type == 3]['user_id'].drop_duplicates())
        feature.append(c_user_cart_count)
        c_user_buy_count = len(group[group.behavior_type == 4]['user_id'].drop_duplicates())
        feature.append(c_user_buy_count)
        
        #用户转化率特征
        feature.append((c_user_buy_count+0.0)/(c_user_click_count+1))
        feature.append((c_user_buy_count+0.0)/(c_user_cart_count+1))
        #最近三天用户数统计
        c_user_3_day_count = len(group[group.time<72]['user_id'].drop_duplicates())
        c_user_3_day_buy_count = len(group[(group.time<72) & (group.behavior_type == 4)]['user_id'].drop_duplicates())
        if c_user_3_day_count == 0:
            feature.append(0)
        else:
            feature.append(c_user_3_day_buy_count/(c_user_3_day_count+0.0))
        #最近七天用户数统计
        c_user_7_day_count = len(group[group.time<168]['user_id'].drop_duplicates())
        c_user_7_day_buy_count = len(group[(group.time<168) & (group.behavior_type == 4)]['user_id'].drop_duplicates())
        if c_user_7_day_count == 0:
            feature.append(0)
        else:
            feature.append(c_user_7_day_buy_count/(c_user_7_day_count+0.0))
        item_group = group.groupby('item_id')
       
        for item , temp_data in item_group:
            temp_feature = feature[:]
            temp_feature.append(item)
            #商品的行为统计数
            i_click_count = len(temp_data[temp_data.behavior_type==1])
            i_cart_count = len(temp_data[temp_data.behavior_type==3])
            i_buy_count = len(temp_data[temp_data.behavior_type==4])
            temp_feature.append(i_click_count/(feature[0]+1.0))
            temp_feature.append(i_cart_count/(feature[1]+1.0))
            temp_feature.append(i_buy_count/(feature[2]+1.0))
            #商品转化率
            i_click_buy_rate = (i_buy_count/(i_click_count+1.0))
            i_cart_buy_rate = (i_buy_count/(i_cart_count+1.0))
            temp_feature.append(i_click_buy_rate/(feature[3]+0.00001))
            temp_feature.append(i_cart_buy_rate/(feature[4]+0.00001))
            #商品用户统计
            i_user_count = len(temp_data['user_id'].drop_duplicates())
            i_user_click_count = len(temp_data[temp_data.behavior_type == 1]['user_id'].drop_duplicates())
            i_user_cart_count = len(temp_data[temp_data.behavior_type == 3]['user_id'].drop_duplicates())
            i_user_buy_count = len(temp_data[temp_data.behavior_type == 4]['user_id'].drop_duplicates())
            #商品用户转化率
            i_user_click_buy_rate = i_user_buy_count/(i_user_click_count+1.0)
            i_user_cart_buy_rate = i_user_buy_count/(i_user_cart_count+1.0)
            temp_feature.append(i_user_count/(feature[5]+1.0))
            temp_feature.append(i_user_click_count/(feature[6]+1.0))
            temp_feature.append(i_user_cart_count/(feature[7]+1.0))
            temp_feature.append(i_user_buy_count/(feature[8]+1.0))
            temp_feature.append(i_user_click_buy_rate/(feature[9]+0.0001))
            temp_feature.append(i_user_cart_buy_rate/(feature[10]+0.0001))
            file.writerow(temp_feature)
            #print temp_feature
            temp_feature = []
        feature = []   
    print 'Generate category feature end!'        
class MyThread(threading.Thread):
    def __init__(self,func,args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
    def run(self):
        apply(self.func,self.args)
            
def create_user_item_feature(data,filename,time_point):
    group_data = data.groupby(['user_id','item_id'])
    for i in xrange(4,5):
        func_name =eval('generate_user_item_feature%d'%(i+1))
        outfilename = filename+'%d.csv'%(i+1)
        func_name(group_data,data,outfilename,time_point)
    
 
#创建用户特征 
def create_user_feature(data,filename,point):
    time_point = point
    group_data = data.groupby('user_id')
    for i in [5]:
        outfilename = filename+'%d.csv'%(i+1)
        print outfilename
        func_name =eval('generate_user_feature%d'%(i+1))
        func_name(group_data,outfilename,time_point)
        
def user_item_feature_merge(filename):
    column_list=[['user_id','item_id','first_operator_day_count','last_operator_day_count'],\
                 ['user_id','item_id','user_buy_brand_rate','user_click_brand_rate',\
                   'click_buy_rate','click_2_count','buy_2_count','brand_is_bought_rate',\
                   'brand_is_clicked_rate','click_delete_user_avge_rate'\
                   ,'buy_delete_user_avg_rate'],['user_id','item_id','last_5_day_click_count','last_5_day_collect_count'\
                   ,'last_5_day_cart_count','last_5_day_buy_count']]
    user_item_feature1_data = pd.read_csv(filename+'1.csv')
    result=user_item_feature1_data
    for i in xrange(2,5):
        temp_user_item_feature_data = pd.read_csv(filename+'%d.csv'%i,header=None,names=column_list[i-2])
        result = pd.merge(result,temp_user_item_feature_data,on=['user_id','item_id'],how='inner')
    return result
    
 
def generate_data_set(full_data,tag_data,test_or_train_flag):
    print 'begin generate data set '
    item_data = pd.read_csv(base_dir + "tianchi_mobile_recommend_train_item.csv")[['item_id']].drop_duplicates()
    if test_or_train_flag == 'train': 
        #训练集:20141120-20141217
        time_point = '2014-12-17 00'
        buy_item = tag_data[tag_data.behavior_type == 4][['item_id']].drop_duplicates()
        item = item_data.append(buy_item).drop_duplicates()
      
        data = pd.merge(full_data,item,on=['item_id'],how='inner')[full_data.columns]
    
        #构建用户特征
        outfile_name = base_dir + 'train_test_set\\train_set_user_feature'
        print 'begin generate train set user feature'
        create_user_feature(full_data,outfile_name,time_point)
        
        #构建商品特征
        outfile_name = base_dir + 'train_test_set\\train_set_item_feature1.csv'
        #generate_item_feature(data,outfile_name,time_point)
        outfile_name = base_dir + 'train_test_set\\train_set_item_feature2.csv'
        #generate_item_feature2(data,outfile_name,time_point)
        outfile_name = base_dir + 'train_test_set\\train_set_item_feature3.csv'
        #generate_item_feature3(data,outfile_name,time_point)
        
        
        
        #构建用户与商品特征
        outfile_name = base_dir + "train_test_set\\train_set_user_item_feature"
        #create_user_item_feature(data,outfile_name,time_point)
        
        #合并用户与商品特征集
        #feature_data = user_item_feature_merge('D://java_workspace//tuijian//src//third_train_set_user_item_feature')
        #feature_data.to_csv('D://java_workspace//tuijian//src//third_train_user_item_feature.csv',index = False)
    
        #构建商品类别特征
        outfile_name = base_dir + '\\train_set_category_feature.csv'
        #generate_category_feature(data,outfile_name,time_point)
        
    elif test_or_train_flag == 'test':
        #测试集:20141118-20141217
        test_data = pd.merge(full_data,item_data,on=['item_id'],how='inner')[full_data.columns]
        time_point = '2014-12-18 00'
        #构建用户特征
        outfile_name = base_dir + 'train_test_set\\predict_set_user_feature'
        create_user_feature(full_data,outfile_name,time_point)
        
        #构建商品特征
        #outfile_name = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_item_feature1.csv'
        #generate_item_feature(test_data,outfile_name,time_point)
        #outfile_name = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_item_feature2.csv'
        #generate_item_feature2(test_data,outfile_name,time_point)
        outfile_name = base_dir + 'train_test_set\\predict_set_item_feature3.csv'
        #generate_item_feature3(test_data,outfile_name,time_point)
        
        #构建用户与商品特征
        outfile_name = base_dir + "train_test_set\\predict_set_user_item_feature"
        #create_user_item_feature(test_data,outfile_name,time_point)
        
        #合并用户与商品特征集
        #feature_data = user_item_feature_merge('D://java_workspace//tuijian//src//predict_train_set_user_item_feature')
        #feature_data.to_csv('D://java_workspace//tuijian//src//predict_train_user_item_feature.csv',index = False)
        
        #构建商品类别特征
        outfile_name = base_dir + 'train_test_set\\predict_set_category_feature.csv'
        #print outfile_name
        #generate_category_feature(test_data,outfile_name,time_point)
  
    
def logic_reg(X,Y,test_data,c,columns_list,prob=False):
    test_data = test_data.set_index(["user_id","item_id"])
    randstate = np.random.RandomState(np.random.randint(1000))
    clf = linear_model.LogisticRegression(C=c,penalty='l2',max_iter=1000)#,random_state=randstate)
    #clf = linear_model.RandomizedLogisticRegression(C=c)
    clf.fit(X, Y) 
    #print clf.get_support()
    
    if prob:
        prob_result = clf.predict_proba(test_data[columns_list])
        result = prob_result[:,1]
    else:
        result = clf.predict(test_data[columns_list])
    return result

def redige_reg(X,Y,test_data,a,columns_list):
    test_data = test_data.set_index(["user_id","item_id"])
    clf = linear_model.Ridge(alpha = a)
    clf.fit(X,Y)
    print clf.coef_
    result = clf.predict(test_data[columns_list])
    return result
def Lasso(X,Y,test_data,a,columns_list):
    test_data = test_data.set_index(["user_id","item_id"])
    #clf = linear_model.Lasso(alpha = a)
    clf = linear_model.RandomizedLasso()
    clf.fit(X,Y)
    print clf.scores_
    return 
    print clf.coef_
    result = clf.predict(test_data[columns_list])
    return result
    
def addboost(X,Y,test_data,columns_list,rate,estimators=100,prob=False):
    test_data = test_data.set_index(["user_id","item_id"])
    if prob:
        clf = AdaBoostRegressor(n_estimators=estimators,learning_rate = rate)
        clf.fit(X,Y)
        result = clf.predict(test_data[columns_list])
    else:
        clf = AdaBoostClassifier(n_estimators  = estimators,learning_rate=rate)
        clf.fit(X,Y)
        result = clf.predict(test_data[columns_list])
    return result

def randomforest_C(X,Y,test_data,columns_list,estimators=20,prob=False):
    test_data = test_data.set_index(["user_id","item_id"])
    randstate = np.random.RandomState(np.random.randint(1000))
 
    if prob:
        rlf = RandomForestRegressor(n_estimators = 100,max_depth=2)#, max_features = len(columns_list),max_depth = 50)#random_state=randstate)#n_estimators=estimators)#,random_state=randstate)
        rlf.fit(X,Y)
        print rlf.feature_importances_
        prob_result = rlf.predict(test_data[columns_list])
        result = prob_result[:]
    else:
        clf = RandomForestClassifier()#random_state=randstate)#n_estimators=estimators)#,random_state=randstate)
        clf.fit(X,Y)
        result = clf.predict(test_data[columns_list])
    return result
def GBRT(X,Y,test_data,columns_list,estimators=100,prob=False):
    test_data = test_data.set_index(["user_id","item_id"])
    randstate = np.random.RandomState(np.random.randint(1000))
    
    if prob:
        rlf = GB_RT(n_estimators=50,learning_rate=0.05,max_depth=2,alpha=0.2)#n_estimators = estimators,learning_rate=0.1,max_depth=3,loss='ls')#,random_state=randstate)
        rlf.fit(X,Y)
        prob_result = rlf.predict(test_data[columns_list])
        result = prob_result[:]
    else:
        clf = GB_DT(n_estimators=estimators)#,random_state=randstate)
        clf.fit(X,Y)
        result = clf.predict(test_data[columns_list])
    return result

def feature_select_2(train_file_name,columns_list,feature_num,model_name):
    file_name = "D:\\java_workspace\\tuijian\\src\\train_test_set\\"+train_file_name[0]
    print "训练集文件%s"%(file_name)
    X,Y = read_X_Y(file_name,columns_list)
    
    feature_select = SelectKBest(f_regression,k=10)
    
    if model_name == 'LR':
        clf = linear_model.LogisticRegression(C=1.0,penalty='l2',max_iter=1000)
    
    anova_model = Pipeline([('FeatureSelect',feature_select),('model',clf)])
    
    if model_name == 'LR':
        anova_model.set_params(FeatureSelect__k=feature_num,model__C=1.0).fit(X,Y)
    
    select_f = anova_model.named_steps['FeatureSelect'].get_support()
    feature_list = []
    n = 0
    for f in select_f:
        if f:
            feature_list.append(columns_list[n])
        n += 1
            
    return feature_list

    
def normalize_data(filename,tag_file,ratio):
    tag = pd.read_csv(tag_file)
    train_set = generate_validate_set(filename,tag,ratio)
    del tag
    return train_set
    #gc.collect()
    #LR_train_set.to_csv(outfilename,index=False)
    
def read_X_Y(filename,select_columns):
    LR_train_set = pd.read_csv(filename)
    LR_train_set = LR_train_set.set_index(["user_id","item_id"])
    columns = set(LR_train_set.columns)
    column_list = list(columns.difference(set(["behavior_type"])))
    X = LR_train_set[select_columns]
    Y = LR_train_set['behavior_type']
    return X,Y

def preprocess_scal(file_list,output,tag_file,merge_columns):
    result_data = pd.read_csv(file_list[0])
    columns = set(result_data.columns)
    index_column = list(columns.intersection(set(['user_id','item_id','behavior_type'])))
    temp_data = result_data.set_index(index_column)
    temp_data = temp_data[temp_data.columns].apply(preprocessing.scale)
    result_data = temp_data.reset_index(index_column)
    for filename in file_list[1:]:
        print filename
        temp_data = pd.read_csv(filename)
        print 'debug 2 feature file lines %d'%len(temp_data)
        
        columns = set(temp_data.columns)
        index_column = list(columns.intersection(set(['user_id','item_id'])))
        temp_data = temp_data.set_index(index_column)
        temp_data = temp_data.apply(preprocessing.scale)
        temp_data = temp_data.dropna(axis=1,how='any')
        temp_data = temp_data.reset_index(index_column)
        result_columns = set(result_data.columns)
        columns_list = list(columns.intersection(result_columns).intersection(merge_columns))
        print columns_list
        result_data = pd.merge(result_data,temp_data,on=columns_list,how='inner')
        #result_data = result_data.join(temp_data,on=columns_list,lsuffix="_x",rsuffix="_y",how='inner')
        print 'debug 3 merge data result data len %d'%len(result_data)
    result_data.to_csv(output,index=False)
    

    

#数据集划分与预处理      
def datapreprocessing():
    file_list = [['D:\\java_workspace\\tuijian\\src\\train_tag_data.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature1.csv',\
                       #'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature4.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature5.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature1.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature4.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature5.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature1.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_category_feature.csv'],
                      ['D://java_workspace//tuijian//src//train_test_set//predict_set_user_item_feature1.csv',\
                       #'D://java_workspace//tuijian//src//train_test_set//predict_set_user_item_feature2.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_item_feature3.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_item_feature4.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_item_feature5.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_feature1.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_feature2.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_feature3.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_feature4.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_user_feature5.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_item_feature1.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_item_feature2.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_item_feature3.csv',\
                       'D://java_workspace//tuijian//src//train_test_set//predict_set_category_feature.csv']]
    out_put_file_list = ['D://java_workspace//tuijian//src//train_test_set//1_2_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_4_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_6_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_8_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_10_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_12_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//1_14_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_16_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_18_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_20_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_22_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_24_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_26_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_28_train_set.csv',\
                         #'D://java_workspace//tuijian//src//train_test_set//1_30_train_set.csv',\
                         'D://java_workspace//tuijian//src//train_test_set//predict_set.csv']
    loop = 0;
    ratio = [2,4,6,8,10,12,14]#,16,18,20,22,24,26,28,30]
    merge_columns = ['user_id','item_id']
    #训练集数据预处理
    
    for r in ratio:
        temp_data = normalize_data(file_list[0][1],file_list[0][0],r)
        merger_feature_data(temp_data,file_list[0][2:],merge_columns,out_put_file_list[loop])
        loop += 1
        del temp_data
        gc.collect()
    
    merger_feature_data(pd.read_csv(file_list[1][0]),file_list[1][1:],merge_columns,out_put_file_list[-1])
    print 'preprocess done'
 
def merger_feature_data(feature_data,feature_data_file_list,merge_columns,outfile_name):
    result_data = feature_data
    print 'debug 1 merger_feature_data input data len %d'%len(result_data)
    #规范化数据在0-1之间
    #f = lambda x:(x-x.min())/(x.max()-x.min())
    f = lambda x:x/(x.max()-x.min())
    #f = preprocessing.scale
    columns = set(result_data.columns)
    index_column = list(columns.intersection(set(['user_id','item_id','behavior_type'])))
    temp_data = result_data.set_index(index_column)
    temp_data = temp_data[temp_data.columns].apply(f)
    result_data = temp_data.reset_index(index_column)
    for filename in feature_data_file_list:
        print filename
        temp_data = pd.read_csv(filename)
        print 'debug 2 feature file lines %d'%len(temp_data)
        
        columns = set(temp_data.columns)
        index_column = list(columns.intersection(set(['user_id','item_id'])))
        temp_data = temp_data.set_index(index_column)
        temp_data = temp_data.apply(f)
        temp_data = temp_data.dropna(axis=1,how='any')
        temp_data = temp_data.reset_index(index_column)
        
        result_columns = set(result_data.columns)
        columns_list = list(columns.intersection(result_columns).intersection(merge_columns))
        print columns_list
        result_data = pd.merge(result_data,temp_data,on=columns_list,how='inner')
        #result_data = result_data.join(temp_data,on=columns_list,lsuffix="_x",rsuffix="_y",how='inner')
        print 'debug 3 merge data result data len %d'%len(result_data)
        
    
    result_data.to_csv(outfile_name,index=False)

def update_data_set(feature_data_file,feature_data_file_list,merge_columns,outfile_name):
    result_data = pd.read_csv(feature_data_file)
    print 'debug 1 merger_feature_data input data len %d'%len(result_data)
    #规范化数据在0-1之间
    #f = lambda x:(x-x.min())/(x.max()-x.min())
    f = lambda x:x/(x.max()-x.min())
    #f = lambda x:(x-x.mean())/x.std()
    #f = lambda x: x
    for filename in feature_data_file_list:
        print filename
        temp_data = pd.read_csv(filename)
        print 'debug 2 feature file lines %d'%len(temp_data)
        columns = set(temp_data.columns)
        index_column = list(columns.intersection(set(['user_id','item_id'])))
        temp_data = temp_data.set_index(index_column)
        temp_data = temp_data.apply(f)
        temp_data = temp_data.dropna(axis=1,how='any')
        temp_data = temp_data.reset_index(index_column)
        result_columns = set(result_data.columns)
        columns_list = list(columns.intersection(result_columns).intersection(merge_columns))
        print columns_list
        result_data = pd.merge(result_data,temp_data,on=columns_list,how='inner')
        #result_data = result_data.join(temp_data,on=columns_list,lsuffix="_x",rsuffix="_y",how='inner')
        print 'debug 3 merge data result data len %d'%len(result_data)
    result_data.to_csv(outfile_name,index=False)
    pass
def describe_predict_result(test_set,reference_set,predict_result,probility,column_list):
    #预测
    user_item_list = test_set[['user_id','item_id']]
    user_item_list.insert(2,'behavior_type',predict_result)
    user_item_list.to_csv('D:\\java_workspace\\tuijian\\src\\train_test_set\\analysys_2.csv',index = False)
    print user_item_list.describe()
    print "------------------------------------------------------------------------------------\n"
    analysys_data = pd.merge(user_item_list,reference_set,on=['user_id','item_id'],how='inner')
    analysys_data.to_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\analysys.csv",index = False)
    print analysys_data.describe()
    #print analysys_data
    #print user_item_list
    if probility:
        #predict_set = user_item_list[(user_item_list.behavior_type <=1.4) & (user_item_list.behavior_type >= 0.35)]redige
        #predict_set = user_item_list[(user_item_list.behavior_type <=1) & (user_item_list.behavior_type >= 0.886)]
        #predict_set = user_item_list[(user_item_list.behavior_type < 0.8) & (user_item_list.behavior_type > 0.65)]
        user_item_list = user_item_list.sort_values('behavior_type',0,False)
        # GBDT predict_set = user_item_list.head(1000)
        predict_set = user_item_list.head(1400)
    else: 
        predict_set = user_item_list[user_item_list.behavior_type==1]
    #去除10小时内已经有过购买行为的用户
    user_data_feature = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_user_feature2.csv")
    user_buy_list = user_data_feature[user_data_feature.user_least_buy_hour < 10][['user_id']]
   
    
    temp_data = pd.merge(test_set,predict_set,on=['user_id','item_id'],how='inner')
    user_item_list = temp_data[(temp_data.buy_count ==0) & (~temp_data.user_id.isin(user_buy_list['user_id']))][["user_id","item_id"]].drop_duplicates()
    print '预测集总量     %d'%len(user_item_list)
    
    temp_data.to_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\analysys_1.csv",index=False)
   
    analysys_data = analysys_data.drop_duplicates()
    print '测试集中可能所有被购买的数据个数%d'%len(analysys_data)               
    precision_set = pd.merge(user_item_list,reference_set, on=['user_id','item_id'], how='inner')
    print '成功预测数%d'%len(precision_set)
    precision = (len(precision_set)+0.0)/(len(user_item_list)+0.00001)
    recall = (len(precision_set)+0.0)/len(reference_set.drop_duplicates())
    F1_score = (2*precision*recall)/(precision+recall+0.00001) 
    print "准确率%f"%precision
    print "召回率%f"%recall
    print "F1值%f"%F1_score
    return F1_score,analysys_data["behavior_type_x"].mean()
    
def one_model_train_predict(file_list,model,select_column):
    #读取验证集
    test_tag_data = pd.read_csv('D:\\java_workspace\\tuijian\\src\\predict_tag_data.csv')
    test_set = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set.csv")
    item_data = pd.read_csv("D:\\java_workspace\\tuijian\\src\\tianchi_mobile_recommend_train_item.csv")[['item_id']].drop_duplicates()
    #生成验证集
    test_tag = pd.merge(test_tag_data[test_tag_data.behavior_type==4],item_data,on=['item_id'],how='inner')[test_tag_data.columns[:5]]
    test_tag = test_tag[test_tag.user_id.isin(test_set['user_id'])]
    test_tag = test_tag[test_tag.item_id.isin(test_set['item_id'])]
    seven_day_data = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_user_item_feature1.csv")
    seven_day_data = seven_day_data[seven_day_data.least_click_day_count < 72][['user_id','item_id']]
    test_set = pd.merge(test_set,seven_day_data,on=['user_id','item_id'],how = 'inner')
    print "测试集数据总量%d"%len(test_set)
    #test_tag = test_tag.drop_duplicates()
    reference_set = test_tag#[['user_id','item_id']]
    print '实际购买总数%d'%len(reference_set)
    for file in file_list:
        print "\n\n"
        file_name = "D:\\java_workspace\\tuijian\\src\\train_test_set\\"+file
        print "训练集文件%s"%(file_name)
        X,Y = read_X_Y(file_name,select_column)
        probility = True
        if model == "LR":
            for C in [1.0]:#,0.5,0.1]:#,0.06,0.05,0.04,0.03,0.02,0.01]:
                #训练
                print "\nparam C %f"%C
                predict_result = logic_reg(X,Y,test_set,C,select_column,prob=probility)
                #print predict_result   
                f1,pro_mean=describe_predict_result(test_set,reference_set,predict_result,probility,select_column)
                return f1,pro_mean
            continue
        elif model == "RF":
            probility = True
            predict_result = randomforest_C(X,Y,test_set,select_column,prob=probility)
        elif model =="GDBT":
            probility=False
            predict_result = GBRT(X,Y,test_set,select_column,prob=probility,estimators=500)
        elif model == "Ridge":
            probility = True
            for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]:
                print "\nparam A %f"%a 
                predict_result = redige_reg(X,Y,test_set,a,select_column)
                describe_predict_result(test_set,reference_set,predict_result,probility,select_column)
            continue
        elif model == "Lasso":
            probility = True
            for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]:
                print "\nparam A %f"%a 
                predict_result = Lasso(X,Y,test_set,a,select_column)
                describe_predict_result(test_set,reference_set,predict_result,probility,select_column)
            continue 
        elif model == "adboost":
            probility = True
            for rate in [0.05,0.01]:
                print "\n rate %f"%rate
                predict_result =  addboost(X,Y,test_set,select_column,rate,estimators=100,prob=probility)  
                describe_predict_result(test_set,reference_set,predict_result,probility,select_column)
            continue
        else:
            print "error model"
        describe_predict_result(test_set,reference_set,predict_result,probility,select_column)
                  
        
        '''
        user_item_list = temp_data[temp_data.behavior_type == 1][["user_id","item_id"]].drop_duplicates()
        print '预测集总量%d'%len(user_item_list)
        analysys_data = pd.merge(user_item_list,reference_set,on=['user_id','item_id'],how='inner')
        #print analysys_data.describe()
        #analysys_data = analysys_data.drop_duplicates()
        print '测试集中可能所有被购买的数据个数%d'%len(analysys_data)       
        precision_set = pd.merge(predict_set,reference_set, on=['user_id','item_id'], how='inner')
        print '成功预测数%d'%len(precision_set)
        precision = (len(precision_set)+0.0)/(len(predict_set)+1)
        recall = (len(precision_set)+0.0)/len(reference_set)
        F1_score = (2*precision*recall)/(precision+recall+0.00001)
        print "准确率%f"%precision
        print "召回率%f"%recall
        print "F1值%f"%F1_score
        #规则：购买过的商品几乎不会被重复购买
        '''
        '''
            predict_set = temp_data[temp_data.buy_count == 0][['user_id','item_id']]
            print len(predict_set)
            user_item_list = predict_set
            GBRT_user_item_list = user_item_list
            normal_test = normal_test_data.reset_index(['user_id','item_id'])
            #print predict_set
            rf_data = pd.merge(normal_test,predict_set,on=['user_id','item_id'],how = 'inner')
            rf_data = rf_data.set_index(['user_id','item_id'])
            #print rf_data
            estimators = 100
            probility = True
            predict_set = randomforest_C(X_R,Y_R,rf_data,estimators,probility)
            user_item_list['behavior_type'] = predict_set
            if probility == False:
                predict_set = user_item_list[user_item_list.behavior_type == 1]
            else:
                predict_set = user_item_list[(user_item_list.behavior_type >= 0.64)&(user_item_list.behavior_type <= 1)]
            print len(predict_set)
            #user_item_list.to_csv(base_dir+'train_test_set\\result.csv')

            time_data = pd.read_csv(base_dir+'train_test_set\\one_week_train_set_user_item_feature2.csv',header=None,names=['user_id','item_id',\
                                                                                               'first_operator_day_count',\
                                                                                               'last_operator_day_count'])
            predict_set = pd.merge(predict_set,time_data,on=['user_id','item_id'],how='left')
            #print predict_set
            #predict_set =predict_set[(predict_set.first_operator_day_count< 5 ) & (predict_set.last_operator_day_count < 4)]
            #print len(predict_set)
            user_item_list = predict_set
            Gbdt_data = pd.merge(normal_test,predict_set[['user_id','item_id']],on=['user_id','item_id'],how ='inner')
            Gbdt_data = Gbdt_data.set_index(['user_id','item_id'])
            estimators = 100;
            #user_item_list = normal_test[['user_id','item_id']]
            #Gbdt_data = normal_test.set_index(['user_id','item_id'])
            predict_set = GBRT(X_G,Y_G,rf_data,estimators,prob=probility)
            GBRT_user_item_list['behavior_type'] = predict_set
            user_item_list = GBRT_user_item_list
            #user_item_list['behavior_type'] = predict_set
            if probility == False:
                predict_set = user_item_list[user_item_list.behavior_type == 1]
            else:
                predict_set = user_item_list[(user_item_list.behavior_type >= 0.64)&(user_item_list.behavior_type <= 1)]
                #print user_item_list
            print len(predict_set)
            user_item_list.to_csv(base_dir+'train_test_set\\result.csv')
            print user_item_list.describe()    
        '''
 
 
def model_train_predict(file_list,column_list):
    #读取验证集
    test_tag_data = pd.read_csv('D:\\java_workspace\\tuijian\\src\\predict_tag_data.csv')
    test_set = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set.csv")
    item_data = pd.read_csv("D:\\java_workspace\\tuijian\\src\\tianchi_mobile_recommend_train_item.csv")[['item_id']].drop_duplicates()
    #生成验证集
    test_tag = pd.merge(test_tag_data[test_tag_data.behavior_type==4],item_data,on=['item_id'],how='inner')[test_tag_data.columns[:2]]
    seven_day_data = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_user_item_feature1.csv")
    seven_day_data = seven_day_data[seven_day_data.least_click_day_count < 170][['user_id','item_id']]
    test_set = pd.merge(test_set,seven_day_data,on=['user_id','item_id'],how = 'inner')
    print "测试集数据总量%d"%len(test_set)
    #test_tag = test_tag.drop_duplicates()
    reference_set = test_tag[['user_id','item_id']]
    print '实际购买总数%d'%len(reference_set)
    X_LR,Y_LR = read_X_Y("D:\\java_workspace\\tuijian\\src\\train_test_set\\"+file_list[3],column_list)
    X_RF,Y_RF = read_X_Y("D:\\java_workspace\\tuijian\\src\\train_test_set\\"+file_list[-2],column_list)
    X_GBDT,Y_GBDT = read_X_Y("D:\\java_workspace\\tuijian\\src\\train_test_set\\"+file_list[-1],column_list)
   
    
    
    for C in [2.0,1.0,0.5]:#,0.1,0.06,0.05,0.04,0.03,0.02]:
        #训练
        probility = True
        #test_set = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set.csv")
        #test_set = pd.merge(test_set,seven_day_data,on=['user_id','item_id'],how = 'inner')
        print "\nparam C %f"%C
        predict_result = logic_reg(X_LR,Y_LR,test_set,C,column_list,prob=probility)
        user_item_list = test_set[['user_id','item_id']]
        user_item_list.insert(2,'behavior_type',predict_result)
        #print user_item_list
        if probility:
            predict_set = user_item_list.head(100000)
        else: 
            predict_set = user_item_list[user_item_list.behavior_type==1]
        
        
        print "LR predict_set num %d"%len(predict_set)
        LR_predict = pd.merge(predict_set,reference_set,on=['user_id','item_id'],how='inner')
        print "LR right predict num %d"%len(LR_predict)
        '''
        #test_set = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_test_set\\set_1\\predict_set.csv")
        RF_data = pd.merge(test_set,predict_set,on=['user_id','item_id'],how='inner')[test_set.columns]
       
        probility =True
        RF_result = randomforest_C(X_RF,Y_RF,RF_data,column_list,estimators=100,prob=probility)
        
        describe_predict_result(RF_data,reference_set,RF_result,probility)
    
    
        user_item_list = RF_data[['user_id','item_id']]
        user_item_list.insert(2,"behavior_type",RF_result)
        
        if probility:
            predict_set = user_item_list.head(100000)
        else: 
            predict_set = user_item_list[user_item_list.behavior_type==1]
        print "RF predict_set num %d"%len(predict_set) 
        RF_predict = pd.merge(predict_set,reference_set,on=['user_id','item_id'],how='inner')
        print "RF right predict num %d"%len(RF_predict)
        
        '''
        probility =False
        GBDT_data = pd.merge(test_set,predict_set,on=['user_id','item_id'],how='inner')[test_set.columns]
        GBDT_result = GBRT(X_GBDT,Y_GBDT,GBDT_data,column_list,estimators=100,prob=probility)
        
        describe_predict_result(GBDT_data,reference_set,GBDT_result,probility,column_list)
        

'''
特征选择：1、去掉取值变化小的特征，如特征中某个特征值占95%以上的特征
'''
def feature_select_1(feature_data):
    column_list = feature_data.columns
    index_column = list(set(column_list).intersection(set(['user_id','item_id','behavior_type'])))
    select_columns =[]#= index_column
    process_data = feature_data.set_index(index_column)
    column_list = process_data.columns
    for column in column_list:
      
        rank_list = process_data[column].value_counts()
        data_num = rank_list.sum()
       
        most_num = rank_list.iloc[0]
        percent =  most_num/(data_num+0.0)
        if percent < 0.95:
        #if True:
            select_columns.append(column)
    return select_columns

            

def feature_select():
    file_list = [       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature1.csv',\
                       #'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature4.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature1.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature1.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature2.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_item_feature3.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_category_feature.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature5.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature4.csv',\
                       'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature5.csv'
                       ]
    result_columns = []
    for file_name in file_list:
        file_object = pd.read_csv(file_name)
        select_columns = feature_select_1(file_object)
        result_columns.extend(select_columns)
    return result_columns
 
'''
每次剔除一个特征比较f1值是否变化：
如果f1值减小，保留这个特征
如果f1值增大，剔除这个特征
'''

def feature_select_3(train_file_name,feature_list,columns_list,model):
    
    pre_f1,pre_mean = one_model_train_predict(train_file_name,model,feature_list)
    f1 = 0.0
    pro_mean = 0.0 
    select_column = feature_list
    for iter in range(5,len(columns_list)):
        remove_feature = columns_list[iter]
        if remove_feature in select_column:
            continue
        select_column.append(remove_feature)
        
        f1 ,pro_mean = one_model_train_predict(train_file_name,model,select_column)
        
        if f1 >= pre_f1 :#and pro_mean >= pre_mean:
            pre_f1 = f1
            pre_mean = pro_mean
            print "add %s f1 socre %f"%(remove_feature,f1)
        else:
            select_column.remove(remove_feature)
    
    if f1 > pre_f1:
        print "f1 socre %f"%(f1)
    else:
        print "f1 socre : %f"%pre_f1
    return select_column 
        
        
    pass
    
    
                 
            
'''
    地理位置信息使用，对每组用户分别建模
'''    
def geo_model(user_group_file_list):
    train_user_item_feature = pd.read_csv(base_dir +  "user_item_feature111.csv")
    predict_user_item_feaure = pd.read_csv(base_dir+"train_test_set\\predict_set_user_item_feature1.csv")
    tag_data = pd.read_csv(base_dir+"train_tag_data.csv")
    ratio = 1
    filename = base_dir + "group_1.csv"
    user_group_list = []
    for user_group_file in user_group_file_list:
        file_object = pd.read_csv(base_dir+user_group_file)
        group = list(file_object['user_id'])
        user_group_list.append(group)
    
    for user_group in user_group_list[1:]:
        one_group_train_set = train_user_item_feature[train_user_item_feature.user_id.isin(user_group)]
        one_group_train_set.to_csv(filename,index=False)
        print len(one_group_train_set)
        break
    
    generate_validate_set(filename,tag_data,ratio)
   
   


         
if __name__=='__main__': 
    #原始数据集
    #full_data = pd.read_csv('tianchi_mobile_recommend_train_user.csv')  
    #标志集
    #test_tag_data = full_data[(full_data.time > '2014-12-17 23')]
    #test_tag_data.to_csv('D:\\java_workspace\\tuijian\\predict_tag_data.csv',index = False)
    
    #构建训练集
    #原始数据集
    #train_data = pd.read_csv(base_dir + 'train_user_item2014-12-17 00.csv')
    #标志集
    #train_tag_data = pd.read_csv(base_dir + 'train_tag_data.csv')
    
    #构建测试集
    #test_data = pd.read_csv(base_dir + 'train_user_item2014-12-18 00.csv')
    #test_tag_data = pd.read_csv(base_dir + 'predict_tag_data.csv')
    
    #构建训练集的用户特征、商品特征、用户与商品特征、商品类别特征
    #generate_data_set(train_data,train_tag_data,'train')
    
    #构建测试集的用户特征、商品特征、用户与商品特征、商品类型特征
    #generate_data_set(test_data,test_tag_data,'test')
    
    #数据预处理
    #datapreprocessing()
    generate_geo_feature(base_dir+'train_test_set\\user_item_geo_data_all.csv', 'train')


    ''' 
    train_data_file = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\set_1\\1_2_train_set.csv'
    test_data_file = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\set_1\\predict_set.csv'
    train_feature_data_file_list = ['D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature4.csv',\
                              'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_feature5.csv',\
                              'D:\\java_workspace\\tuijian\\src\\train_test_set\\train_set_user_item_feature5.csv'
                              ]
    test_feature_data_file_list = ['D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_user_feature4.csv',\
                              'D:\\java_workspace\\tuijian\\src\\train_test_set\\preidct_set_user_feature5.csv',\
                              'D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set_user_item_feature5.csv'
                              ]
    train_outfile_name = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\1_2_train_set.csv'
    test_outfile_name = 'D:\\java_workspace\\tuijian\\src\\train_test_set\\predict_set.csv'
    update_data_set(train_data_file,train_feature_data_file_list,['user_id','item_id'],train_outfile_name)
    #update_data_set(test_data_file,test_feature_data_file_list,['user_id','item_id'],test_outfile_name)
    '''


    #特征选择
    select_column = []
    #select_column = feature_select()
    
    print  'column len %d'%len(select_column)
    
    #LR feature set
    '''
    LR_feature = select_column[:5] + select_column[7:8] + select_column[10:11] +select_column[12:13] + \
     select_column[15:16] + select_column[17:22] + select_column[24:25] +select_column[27:28] + select_column[30:32] +select_column[34:35] +\
     select_column[36:37] + select_column[39:41] + select_column[44:45] + select_column[49:50] + select_column[51:52] +select_column[53:54] + \
     select_column[55:57] + select_column[61:62] + select_column[64:65] + select_column[69:70] + select_column[70:71] + select_column[77:79] 
    '''
    LR_feature = ['cart_count', 'least_cart_day_count', 'click_count', 'least_click_day_count', 'user_click_brand_rate',\
                   'click_delete_user_avge_rate', '7-day_click_count', 'user_collect_count', 'user_click_buy_rate', 'user_cart_buy_rate',\
                    'u_last1day_click_buy_rate', 'u_last1day_collect_buy_rate', 'u_last1day_cart_buy_rate', 'u_last3day_click_buy_rate',\
                     'u_last7day_click_buy_rate', 'user_least_click_hour', 'user_least_buy_hour', 'user_last_1_day_click_count', \
                     'user_last_1_day_buy_count', 'user_collect_brand_count_for_1_day', 'user_last_3_day_click_count', \
                     'user_last_3_day_collect_count', 'user_last_5_day_collect_count', 'all_collect_count', 'all_cart_count',\
                      '1_day_click_count', '3_day_click_count', '3_day_collect_count', 'hot_level', 'third_day_click_pro', 'c_click_buy_rate', \
                      'c_cart_buy_rate', 'c_last3day_user_click_buy_rate', 'c_last7day_user_click_buy_rate', '3-day_click_count',\
                       'user_collect_buy_rate', 'u_last3day_collect_buy_rate', 'u_last3day_cart_buy_rate', 'u_last7day_collect_buy_rate',\
                        'user_last_1_day_collect_count', 'user_buy_brand_count_for_1_day', 'user_last_5_day_click_count', 'all_click_count', \
                        '5_day_click_count', 'uif_gapday', 'uif_click_10', 'u_cart_item_buy_rate', 'u_7day_click_item_buy_rate',\
                         'u_7day_cart_item_buy_rate', 'uf_collectitems_10', 'uf_click_cate_buy_rate_10', 'uf_click_item_buy_rate_10', \
                         'uf_collectitems_7', 'uf_click_item_buy_rate_7', 'uf_collectitems_5', 'uf_click_item_buy_rate_5', 'uf_collectitems_3', \
                         'uf_click_item_buy_rate_3', 'uf_click_item_buy_rate_2', 'uf_buycate_1', 'uf_click_item_buy_rate_1', 'click_2_count', \
                         'u_last7day_cart_buy_rate', 'ave_cart_count', '7_day_click_count', 'uif_active_day_10', 'u_1day_click_item_buy_rate',\
                          'u_3day_click_item_buy_rate', 'u_3day_cart_item_buy_rate', 'uf_click_cate_buy_rate_7']
    #select_column = select_column[:2] + select_column[3:6] + select_column[9:11] + select_column[14:16] + select_column[41:42] + select_column[50:51] + select_column[52:53] + select_column[54:55] + select_column[62:63] + select_column[69:70] + select_column[71:73] + select_column[76:77] + select_column[79:80] + select_column[81:]
    #adboost feature set
    #select_column = select_column[:5]# + select_column[7:9] #+ select_column[13:14] #+ select_column[12:13]
    '''
    #GBDT feature set
    select_column = select_column[0:7] + select_column[8:11] + select_column[13:18] + select_column[19:20] + select_column[23:26] + \
    select_column[28:29] + select_column[30:31] + select_column[34:35] + select_column[38:39] + select_column[43:44] + select_column[49:51] +\
     select_column[54:61] + select_column[62:65] + select_column[65:66] + select_column[81:83]
    '''
    train_file_list =["1_2_train_set.csv","1_4_train_set.csv","1_6_train_set.csv","1_8_train_set.csv","1_10_train_set.csv",\
                      "1_12_train_set.csv","1_14_train_set.csv","1_16_train_set.csv","1_18_train_set.csv","1_20_train_set.csv",\
                      "1_22_train_set.csv","1_24_train_set.csv","1_26_train_set.csv","1_28_train_set.csv","1_30_train_set.csv"
                      ]
    #select_column = feature_select_3(train_file_list[0:1],LR_feature,select_column,"LR")
    print 'feature num %d\n'%len(LR_feature)
    #select_column = feature_select_2(train_file_list[0:1],select_column,10,'LR')
    #one_model_train_predict(train_file_list[0:1],"LR",LR_feature)
    #model_train_predict(train_file_list[0:],select_column)


