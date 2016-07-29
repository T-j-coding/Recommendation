# -*- coding:utf -8-*-
import pandas as pd
import csv
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
from collections import Counter
import os
import gc
import numpy as np
from scipy.optimize import curve_fit
import math

#工作根目录
base_dir = 'D:\\java_workstation\\workspace\\data\\'

'''
    将原始数据集以天为单位划分
'''


def generate_every_day_data():
    data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')[['item_id']].drop_duplicates()
    data = pd.merge(data, item_data, on=['item_id'], how='inner')
    time_index = pd.to_datetime(data['time'])
    data_time = data.set_index(time_index)
    first_day = datetime.strptime('2014-11-18', '%Y-%m-%d')
    for i in xrange(31):
        current_day = first_day + timedelta(i)
        current_date_str = datetime.strftime(current_day.date(),'%Y-%m-%d')
        current_day_data = data_time[current_date_str]
        current_day_data.to_csv('D:\\java_workspace\\tuijian\\src\\%d.csv' %(i+1), index=False)
'''
    分析一天中购买用户群的特征
'''


def every_user_data_analy():
    #存储每天的用户
    user_list = []
    for i in xrange(31):
        one_day_data = pd.read_csv(base_dir + '%d.csv'%(i+1))
        if i == 30:
            one_day_user_list = one_day_data[one_day_data.behavior_type == 4][['user_id']].drop_duplicates()
        else:
            one_day_user_list = one_day_data[['user_id']].drop_duplicates()
        user_list.append(one_day_user_list)
    print user_list[30]
    user = pd.DataFrame()
    one_day_user_number_list = []
    same_user_list = []
    new_user_number_list = []
    for i in xrange(0,30).__reversed__():
        one_day_user_number_list.append(len(user_list[i]))
        same_user = pd.merge(user_list[30],user_list[i],on='user_id',how='inner')
        same_user_list.append(len(same_user))
        user = user.append(same_user)
        user = user.drop_duplicates()
        new_user_number_list.append(len(user))
    d = {'one_day_user_number': one_day_user_number_list, 'buy_user_number': same_user_list, 'new_user_number': new_user_number_list}
    user_df = pd.DataFrame(data=d)
    print user_df

'''
    将商品数据集的位置信息分析结果存放在文件item_geo_info.txt中
'''


def item_geo_analy():
    filename = base_dir + 'item_geo_info.txt'
    file_writer = csv.writer(open(base_dir + 'item_geo.csv','wb'))
    file_object = open(filename, 'w')
    lines = []
    #读取商品信息
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    #获取没有地理位置信息的商品
    item_no_geo = item_data[item_data.item_geohash.isnull()]
    #获取包含地理位置信息的商品
    item_have_geo = item_data[item_data.item_geohash.notnull()]
    
    gb = item_have_geo.groupby('item_id')
    for item,iter in gb:
        line = [item]
        for i in range(3,4):
            geo_list = (iter['item_geohash'].apply(lambda s: s[:i]))
            geo = Counter(geo_list).most_common()
            line.extend(geo)
        file_writer.writerow(line)
        line = []
        
    #获取两个商品子集中包含的商品类别和商品数量
    no_geo_category = item_no_geo[['item_category']].drop_duplicates()
    lines.append('没有地理位置的商品子集中包含%d个商品类别\n'%len(no_geo_category))
    have_geo_category = item_have_geo[['item_category']].drop_duplicates()
    lines.append('包含地理位置信息商品子集中包含%d个商品类别\n'%len(have_geo_category))
    
    no_geo_item_sum = item_no_geo[['item_id']].drop_duplicates()
    have_geo_item_sum = item_have_geo[['item_id']].drop_duplicates()
    lines.append('没有地理位置信息的商品子集中包含的商品数为：%d\n'%len(no_geo_item_sum))
    lines.append('包含地理位置信息的商品子集中包含的商品数为：%d\n'%len(have_geo_item_sum))
    
    #获取两个商品子集中的商品类别交集和商品id交集
    category_merge = pd.merge(no_geo_category,have_geo_category,\
                              on=['item_category'],how='inner')
    
    item_merge = pd.merge(no_geo_item_sum,have_geo_item_sum,
                          on=['item_id'],how = 'inner')
    lines.append('两个商品子集的商品类别交集个数为:%d\t商品id交集个数为：%d\n'%(len(category_merge)\
                                                        ,len(item_merge)))
    #将信息写入文件
    file_object.writelines(lines)
    file_object.close()
'''
     分析用户数据集中的地理位置信息，将分析数据保存在文件user_geo_data_analyze.txt中
'''


def user_geo_analy(): 
    file_name = base_dir + "user_geo_data_analyze.txt"
    user_geo_file_name = base_dir + "user_geo.csv"
    file_writer = csv.writer(open(user_geo_file_name, 'wb'))
    with open(file_name, 'w') as file_object:
        #读取用户交互数据
        all_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
        #包含位置信息的数据
        geo_data = all_data[all_data.user_geohash.notnull()]
        file_object.write("包含位置信息的交易记录个数为{0}\n".format(len(geo_data)))
        user_group = geo_data[['user_id']].drop_duplicates()
        file_object.write("包含位置信息的用户个数{0}\n".format(len(user_group)))
        #根据用户ID分组处理位置信息
        gb = geo_data.groupby('user_id')
        line = ['user_id', 'geo_1', 'geo_2', 'geo_3']
        file_writer.writerow(line)

        for user, iter in gb:
            line = [user]
            for i in range(1,4):
                geo_list = (iter['user_geohash'].apply(lambda s: s[:i]))
                geo = Counter(geo_list).most_common()
                line.append(geo[0][0])
            file_writer.writerow(line)


'''
用户位置信息分析
'''


def use_geo_analy_2():
    full_data = pd.read_csv(base_dir + "tianchi_mobile_recommend_train_user.csv")
    use_geo_data = full_data[full_data.user_geohash.notnull()].sort_values(by=['time'])
    output = csv.writer(open(base_dir + "train_test_set\\use_geo_analy.csv", 'wb'))
    line_list = ['user_id','item_id','item_category','behavior_type','geo_1','geo_2','geo_3','geo_4','geo_5','geo_6','geo_7',\
                 'time_diff','hour','week_day']
    output.writerow(line_list)
    line_list = []
    last_day = datetime.strptime('2014-12-18 00','%Y-%m-%d %H')
    for line in use_geo_data.values:
        line_list.extend(line[0:2])
        line_list.append(line[4])
        line_list.append(line[2])
        geo_list  = []
        for index in range(1,8):
            geo_list.append(line[3][:index])
        line_list.extend(geo_list)
        current_day = datetime.strptime(line[5],'%Y-%m-%d %H')
        time_diff = last_day - current_day
        line_list.append(time_diff.total_seconds()/3600)
        line_list.append(line[5].split(' ')[1])
        line_list.append(current_day.strftime('%w'))
        output.writerow(line_list)
        line_list = []
        pass
    pass 

'''
分析用户位置信息
'''


def use_geo_analy_3():
    data = pd.read_csv(base_dir + "train_test_set\\use_geo_analy.csv")
    user_group = data.groupby(['user_id']) 
    output = csv.writer(open(base_dir + 'train_test_set\\use_geo_analy_2.csv', 'wb'))
    line_list = ['user_id','geo_1','geo_2','geo_3','geo_4','geo_5','geo_6','geo_7']
    output.writerow(line_list)
    for user, data in user_group:
        line_list.append(user)
        temp_data = data[['geo_1','geo_2','geo_3','geo_4','geo_5','geo_6','geo_7','time_diff']]
        for column_index in range(7):
            column = [temp_data.columns[column_index]] + ['time_diff']
            geo_data = temp_data[column].drop_duplicates()
            most_count = Counter(geo_data[temp_data.columns[column_index]]).most_common()[0][1]
            pro = most_count/(len(geo_data)+0.0)
            line_list.append(pro)
        print line_list
        output.writerow(line_list)
        pass

   

'''
     预测用户位置信息
'''


def predict_user_geo(line,user_group,user_geo_map,threshold):
    all_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    buy_data = all_data[all_data.behavior_type > 1]
    #geo_item = item_data[item_data.item_category.isin(line)]['item_category'].drop_duplicates()
    file_name = open(base_dir + 'user_predict_geo.csv','wb')
    file_object = csv.writer(file_name)
    #不包含位置信息的数据
    no_geo_data = all_data[all_data.user_geohash.isnull() & ~all_data.user_id.isin(user_group)]
    #包含位置信息的数据
    geo_data = all_data[all_data.user_geohash.notnull() | all_data.user_id.isin(user_group)]
    #抽取没有位置信息的用户
    geo_user_list = geo_data['user_id'].drop_duplicates()
    no_geo_user = no_geo_data[~no_geo_data.user_id.isin(geo_user_list)]['user_id'].drop_duplicates()
    no_geo_user_item = no_geo_data[no_geo_data.user_id.isin(no_geo_user)]  
    if len(no_geo_user) == 0:
        return False
    #对不含位置信息的用户预测位置
    gb = no_geo_user_item.groupby('user_id')
    user_geo_data = pd.read_csv(base_dir + 'user_geo.csv')
    if len(user_geo_map) != 0:
        user_geo_data = user_geo_data.append(user_geo_map)
    user_geo = ['user_id','geo_2']
    file_object.writerow(user_geo)
    for user,iter in gb:
        item_list = iter[iter.item_category.isin(line)&iter.behavior_type > 0]['item_id'].drop_duplicates()
        user_list = geo_data[geo_data.item_id.isin(item_list) & (geo_data.behavior_type >1)]['user_id'].drop_duplicates()
        geo_hash_list = user_geo_data[user_geo_data.user_id.isin(user_list)]['geo_2']
        geo_sort = Counter(geo_hash_list).most_common()
        if len(geo_hash_list):
            if geo_sort[0][1] > threshold:
                user_geo.append(user)
                user_geo.append(geo_sort[0][0])
                file_object.writerow(user_geo)
        else:
            item_geo = item_data[item_data.item_id.isin(item_list) & item_data.item_geohash.notnull()]['item_geohash']
            item_geo = item_geo.apply(lambda x : x[:2])
            geo_sort = Counter(item_geo).most_common()
            if len(geo_sort):
                user_geo.append(user)
                user_geo.append(geo_sort[0][0])
                file_object.writerow(user_geo)
            else:
                user_list = buy_data[buy_data.item_id.isin(item_list)]['user_id'].drop_duplicates()
                geo_hash_list = user_geo_data[user_geo_data.user_id.isin(user_list)]['geo_2']
                geo_sort = Counter(geo_hash_list).most_common()
                if len(geo_hash_list):
                    user_geo.append(user)
                    user_geo.append(geo_sort[0][0])
                    file_object.writerow(user_geo)
            
        user_geo = []   
    file_name.close() 
    return True 
        

'''
    填充商品位置信息
'''


def predict_item_geo():
    #读取商品信息
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    item_list = item_data[item_data.item_geohash.isnull()]['item_id'].drop_duplicates()
    #读取交互数据
    all_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    
    
    #读取用户及其位置信息
    user_geo = pd.read_csv(base_dir + 'user_geo.csv')
    
    #过滤出包含需要预测位置的商品的交互数据并且这些数据中的用户都包含自己的位置
    user_item = all_data[all_data.item_id.isin(item_list)]# & all_data.user_id.isin(user_geo.user_id)]
    gb = user_item.groupby('item_id')
    file_object = csv.writer(open(base_dir + "item_user_action_count.csv",'wb'))

    for item,group in gb:
        line = [item]
        user_list = group[group.behavior_type > 1]['user_id'].drop_duplicates()
        line.append(len(user_list))
        geo_list = user_geo[user_geo.user_id.isin(user_list)]['geo_3']
        line.extend(Counter(geo_list).most_common())
        file_object.writerow(line)
        line = []
    

'''
     计算局部性损失
'''


def computer_LG(user_geo_data,data,columns,user_group_dict):
    #根据商品id分组
    item_gb = data.groupby('item_id')
    #商品数量
    item_count = 0
    print "begin"
    parent_item_user_count_dict = {}
    for item,group in item_gb:
        item_count += 1
        #计算父亲节点每个商品的用户集
        user_count = len(group['user_id'].drop_duplicates())
        parent_item_user_count_dict[item] = user_count * (user_count -1) / 2
    
    #分裂子节点
    children_item_user_count_dict = dict()
    if columns != None:
         user_geo_gb = user_geo_data.groupby(columns)
         for geo,temp in user_geo_gb:
             print "geo_hash %s" %geo
             #用户集
             user_group = temp['user_id']
             user_group_dict[geo] = temp
             #用户集的交互记录
             user_item_group = data[data.user_id.isin(user_group)]
             print len(user_item_group)
             one_children_dict = computer_LG(None,user_item_group,None,None)
             for key in one_children_dict:
                 children_item_user_count_dict[key] = children_item_user_count_dict.get(key,0) + one_children_dict[key]
             
         LG = 0
         for key in parent_item_user_count_dict:
             LG += (parent_item_user_count_dict[key] - children_item_user_count_dict[key]) / (parent_item_user_count_dict[key] + 0.1)
         
         LG /= item_count
         
         return LG 
              
             
    else:
        return  parent_item_user_count_dict
             
             
         
        
'''
    对用户进行聚类分析
    对用户根据位置信息进行分组
'''


def user_cluster():
    #输出文件名
    output_file_name = base_dir + "user_group_"
    #包含位置信息的用户集
    user_geo = pd.read_csv(base_dir + 'union_user_geo.csv')
    #所有的交互记录
    user_item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    #不包含位置信息的用户集
    user_no_geo = user_item_data[~user_item_data.user_id.isin(user_geo['user_id'])][['user_id']].drop_duplicates()
    
    all_data = user_item_data[user_item_data.user_id.isin(user_geo['user_id']) & (user_item_data.behavior_type > 1)]
    #对用户根据局部性损失系数LG进行组
    user_group_dict = {}
    LG = computer_LG(user_geo,all_data,'geo_1',user_group_dict)
    user_group_dict['4'] = user_group_dict['4'].append(user_group_dict['6'].append(user_group_dict['o']))
    user_group_dict.pop('6')
    user_group_dict.pop('o')
    
    data = all_data[all_data.user_id.isin(user_group_dict['9']['user_id'])]
    LG = computer_LG(user_group_dict['9'],data,'geo_2',user_group_dict)
    
    user_group_dict['4'] = user_group_dict['4'].append(user_group_dict['t'])
    
    user_group_dict['4'] = user_group_dict['4'].append(user_no_geo)
    
    user_group_dict.pop('t')
    
    user_group_dict['m'] = user_group_dict['m'].append(user_group_dict['mt'])
    user_group_dict.pop('mt')
    
    user_group_dict['90'] = user_group_dict['90'].append(user_group_dict['91'].append(user_group_dict['92']))
    user_group_dict.pop('91') 
    user_group_dict.pop('92')
    
    user_group_dict['9e'] = user_group_dict['9e'].append(user_group_dict['9h'])
    user_group_dict.pop('9h')
    
    user_group_dict['9m'] = user_group_dict['9m'].append(user_group_dict['9n'].append(user_group_dict['9o']).append(user_group_dict['9p']))
    user_group_dict.pop('9n')
    user_group_dict.pop('9o')
    user_group_dict.pop('9p')
    
    user_group_dict['9s'] = user_group_dict['9s'].append(user_group_dict['9t'].append(user_group_dict['9u']).append(user_group_dict['9v']))
    user_group_dict.pop('9t')
    user_group_dict.pop('9u')
    user_group_dict.pop('9v')
    
    for key , item  in user_group_dict.iteritems():
        print key
        item.to_csv(output_file_name+key+'.csv',index=False)
             
        
'''
    分析商品子集中的商品类型与用户交互集中的商品类型
'''


def category_analy():
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    #user_item_data = pd.read_csv('D:\\java_workspace\\tuijian\\src\\tianchi_mobile_recommend_train_user.csv')
    gb = item_data.groupby('item_category')
    line = []
    count = 0
    for category,item in gb:
        item_geo = item[item.item_geohash.notnull()]['item_id'].drop_duplicates()
        item_no_geo = item[item.item_geohash.isnull()]['item_id']
        len1 = len(item_geo)
        len2 = len(item_no_geo)
        if True:#len1 > (len1 + len2) * 0.1:
            line.append(category)
            count += 1
    return line

'''
    生成用户的预测位置
'''


def generate_predict_geo():
    #商品类别分析
    line = category_analy()
    #预测用户位置信息
    for threshold in [10,9,8,7,6,5,4,3,2,1]:
        print threshold
        if os.path.exists(base_dir + 'user_predict_geo.csv'):
            user_geo_map = pd.read_csv(base_dir + 'user_predict_geo.csv')
            user_group = user_geo_map['user_id']
        else:
            user_geo_map = []
            user_group = []
        flag = predict_user_geo(line,user_group,user_geo_map,threshold)
        if flag is False:
            break    

'''
    用户和商品长尾现象
'''


def user_item_long_tail():
    user_data = pd.read_csv(base_dir + "user_feature1.csv")
    user_buy_count = user_data.user_cart_count + user_data.user_buy_count
    user_count = Counter(user_buy_count).most_common()
    user_long_tail_df = pd.DataFrame.from_records(user_count, columns=['user_hot_level','user_count'])
    
    user_long_tail_df.plot(x='user_hot_level',y='user_count',kind='scatter')
    
    #读取商品信息
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    item_list = item_data['item_id'].drop_duplicates()
    #读取交互数据
    all_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    data = all_data[all_data.behavior_type > 2 & all_data.item_id.isin(item_list)]
    item_tail_list = []
    gb = data.groupby('item_id')
    for item,group in gb:
        count = len(group)
        item_tail_list.append(count)
    item_count = Counter(item_tail_list).most_common()
    item_long_tail_df = pd.DataFrame.from_records(item_count,columns=['item_hot_level','item_count'])
    item_long_tail_df.plot(x='item_hot_level',y='item_count',kind='scatter')
    plt.show()
   
#将原始数据中的时间转化为小时并且进行数据分割
#返回分割后文件名


def time_context(time_point):
    # data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_user.csv')
    # test_data = data[data.time < time_point]#[['user_id','item_id','time']]
    # del data
    # gc.collect()
    # func = lambda line: (pd.to_datetime(time_point)-pd.to_datetime(line['time']))
    # test_data['time'] = test_data[['time']].apply(func,1)
    # test_data['time'] = test_data[['time']].apply(lambda line:(line['time'].components.days * 24 + line['time'].components.hours),1)
    file_name = base_dir + "train_user_item" + time_point + ".csv"
    # test_data.to_csv(file_name,index=False)
    return file_name
#


def time_aware_data_analy():
    item_data = pd.read_csv(base_dir + 'tianchi_mobile_recommend_train_item.csv')
    item_data = item_data[['item_id']].drop_duplicates()
    time_point_list = ['2014-12-17 00', '2014-12-18 00']
    file_name_list = []
    for time_point in time_point_list:
        file_name = time_context(time_point)
        file_name_list.append(file_name)

    for behavior in [1, 2, 3, 4]:
        temp_list = list()
        for i in range(2):
            one_day_data_file_name = base_dir + '%s.csv' % (30+i)
            buy_data = pd.read_csv(one_day_data_file_name)
            buy_data = pd.merge(buy_data, item_data, on=['item_id'], how='inner')[buy_data.columns]
            buy_data = buy_data[buy_data.behavior_type == 4][buy_data.columns[:2]].drop_duplicates()
            buy_num = len(buy_data)
            test_data = pd.read_csv(file_name_list[i])
            test_data = test_data[test_data.behavior_type == behavior]
            gb = test_data.groupby('time')
            for t, item in gb:
                count = len(pd.merge(buy_data, item, on=['user_id', 'item_id'], how='inner').drop_duplicates())
                temp_list.append([t, (count+0.0)/buy_num])
        temp_df = pd.DataFrame(temp_list, columns=['time_distance', 'probility'])
        temp_df.to_csv(base_dir + "time_data_%d.csv" % behavior, index=False)
    # return temp_df
'''
拟合时间曲线
'''


def print_line():
    # time_aware_data_analy()
    temp_df = pd.read_csv(base_dir + "time_data_4.csv")

    plt.xlabel("time_distance")
    plt.ylabel("probility")
    plt.scatter(temp_df['time_distance'], temp_df['probility'])
    print 'scatter finished '

    '''
    #temp_df = pd.read_csv("D:\\java_workspace\\tuijian\\src\\time_data_df.csv")
    #gb = temp_df.groupby('time_distance')
    #temp_list = []
    #for t,item in gb:
        #temp_list.append([t,item['probility'].mean()])
        
    #temp_df = pd.DataFrame(temp_list,columns=['time_distance','probility'])
    #plt.scatter(temp_df['time_distance'],temp_df['probility'])
    '''
    func = lambda x, a, b: b*np.power(a, np.log(x)+1)
    # func = lambda x, a, b: 1/(a*x + b)
    # func = lambda x, a, b: np.e**(a * x + b)

    print 'begin curve fit'
    popt, pcov = curve_fit(func, temp_df['time_distance'], temp_df['probility'], maxfev=500)
    print 'curve fit finished'
    print popt
    plt.plot(temp_df['time_distance'], func(temp_df['time_distance'], *popt))
    plt.show()


'''
    合并有地理位置信息的用户和预测地理位置的用户信息
'''
def merge_user_geo():
    #合并用户位置信息
    
    file_name = 'D:\\java_workspace\\tuijian\\src\\union_user_geo.csv'
    geo1 = pd.read_csv('D:\\java_workspace\\tuijian\\src\\user_geo.csv')
    geo2 = pd.read_csv('D:\\java_workspace\\tuijian\\src\\user_predict_geo.csv')
    geo = geo1.append(geo2)
    geo.to_csv(file_name, index=False)
    
'''
分析用户在一个月内的活跃度变化
'''
def use_active_level_analy():
    data_set = pd.read_csv("D:\\java_workspace\\tuijian\\src\\train_user_item2014-12-18 00.csv")
    data_set['time'] = data_set[['time']].apply(lambda line:(int(line['time']/24)),1)
    output_dir = "D:\\java_workspace\\tuijian\\src\\train_test_set\\user_active.csv"
    file = csv.writer(open(output_dir,'wb'))
    #用户在这一个月内，每一天的不同操作类型统计分析
    behavior_type=['click','collect','cart','buy']
    feature_name = ['daytime','user_id','click_count','collect_count','cart_count','buy_count']
    group_data = data_set.groupby(['user_id'])
    file.writerow(feature_name)
    feature = []
    num = 0
    for user_id , temp_data in group_data:
        for day in range(30):
            feature.append(day)
            feature.append(user_id)
            for i in range(4):
                count = len(temp_data[(temp_data.behavior_type == i+1) &(temp_data.time == day)])
                feature.append(count)
            file.writerow(feature)
            feature = []
            pass
        
        num += 1
    pass

'''
对用户存在位置信息但是当前记录不存在的位置进行填充
user_geo_data:原始数据集中所有包含位置信息的记录
hour_diff:当前记录距离12-18的时间间隔（单位是小时）
'''
def extend_geo_list(user_id,user_geo_data,hour_diff):
    user_geo_static_data = pd.read_csv(base_dir + "train_test_set\\use_geo_analy_2.csv")
    user_data = user_geo_data[user_geo_data.user_id == user_id]
    user_static = user_geo_static_data[user_geo_static_data.user_id == user_id]
    if len(user_static[user_static.geo_4 == 1]):
        return user_data.values[0][4:8]
    
    geo_list = []
    data_cur = user_data[user_data.time_diff == hour_diff]
    if len(data_cur) != 0:
        geo_list = data_cur.values[0][4:8]
        return geo_list
    
    data_before = user_data[user_data.time_diff > hour_diff]
    data_after = user_data[user_data.time_diff < hour_diff]
    if len(data_before) > 0:
        hour_diff_before = data_before.values[-1][11] - hour_diff
    else:
        hour_diff_before = 800
    if len(data_after) > 0:
        hour_diff_after = hour_diff - data_after.values[0][11]
    else:
        hour_diff_after = 800
    if hour_diff_before == hour_diff_after:
        return data_before.values[-1][4:8]
    
    elif hour_diff_before > hour_diff_after:
        return data_after.values[0][4:8]
        pass
    else:
        return data_before.values[0][4:8]
        pass
    #return geo_list
    pass

'''
对没有位置信息的用户的位置进行预测
'''
def predict_usergeo(user_id,data,user_geo_data):
    geo_list = []
    #获取用户交互的商品集
    item_group = data[(data.user_id == user_id)]['item_id'].drop_duplicates()
    #获取对商品集有过交互的用户集
    user_item_group = data[(data.item_id.isin(item_group))][['user_id','item_id']].drop_duplicates()
    #获取用户集
    gb = user_item_group.groupby('user_id')
    user_group = []
    max_item = 0
    max_user = 0
    for user, user_data in gb:
        if user == user_id:
            continue
        else:
            user_group.append(user)
        ''' 
        if len(user_data) > max_item:
            max_user = user
            max_item = len(user_data)
        '''
    #user_group.append(max_user)
    #获取用户集中用户的位置信息
    geo_data = user_geo_data[user_geo_data.user_id.isin(user_group)]
    if len(geo_data) == 0:
        return ['None', 'None', 'None', 'None']
    for geo in ['geo_1', 'geo_2', 'geo_3', 'geo_4']:
        geo_list.append(Counter(geo_data[geo]).most_common()[0][0])
    return geo_list
    


'''
将原始数据集中不包含位置信息的记录进行填充并扩展；
将包含位置信息的记录进行扩展
'''


def generate_geo_data():
    full_data = pd.read_csv(base_dir + "tianchi_mobile_recommend_train_user.csv")
    item_data = pd.read_csv(base_dir + "tianchi_mobile_recommend_train_item.csv")[['item_id']].drop_duplicates()
    data = pd.merge(full_data, item_data, on=['item_id'], how='inner')[full_data.columns]
    user_geo_data = pd.read_csv(base_dir + "train_test_set\\use_geo_analy.csv")
    user_have_geo_list = list(user_geo_data['user_id'].drop_duplicates())
    print '有位置信息的用户个数%d' % (len(user_have_geo_list))
    user_no_geo_list = data[~data.user_id.isin(user_have_geo_list)]['user_id'].drop_duplicates()
    #无位置信息的用户位置预测
    user_geo_map = {}
    for id in user_no_geo_list:
        user_geo_map[id] = predict_usergeo(id, data, user_geo_data)
       
    output = csv.writer(open(base_dir + "train_test_set\\user_item_geo_data_all.csv", 'wb'))
    line_list = ['user_id', 'item_id', 'item_category', 'behavior_type', 'geo_1',
                 'geo_2', 'geo_3', 'geo_4', 'time_diff', 'hour', 'week_day']
    output.writerow(line_list)
    line_list = []
    last_day = datetime.strptime('2014-12-18 00', '%Y-%m-%d %H')
    # func = lambda x: (datetime.strftime('2014-12-18 00', '%Y-%m-%d %H')-datetime.strftime(x, '%Y-%m-%d %H')).total_seconds()/3600
    # full_data['time'] = full_data['time'].apply(func)
    for line in full_data.values:
        line_list.extend(line[0:2])
        line_list.append(line[4])
        line_list.append(line[2])
        geo_list = []
        current_day = datetime.strptime(line[5], '%Y-%m-%d %H')
        time_diff = last_day - current_day
        hour_diff = time_diff.total_seconds()/3600
        if line[3] > 0:
            for index in range(1, 5):
                geo_list.append(line[3][:index])
        else:
            if line[0] in user_have_geo_list:
                geo_list = extend_geo_list(line[0], user_geo_data, hour_diff)
            else:
                if line[0] in user_geo_map:
                    geo_list = user_geo_map[line[0]]
                else:
                    geo_list = ['None', 'None', 'None', 'None']
                pass
            pass
        line_list.extend(geo_list)
        line_list.append(hour_diff)
        line_list.append(line[5].split(' ')[1])
        line_list.append(current_day.strftime('%w'))
        output.writerow(line_list)
        line_list = []
        pass
    pass
   

       
if __name__ == '__main__':
    #商品位置信息分析
    #item_geo_analy()
    #用户位置信息分析
    #user_geo_analy()
    #生成用户的预测位置
    #generate_predict_geo()
    #用户和商品的长尾问题
    #user_item_long_tail()
    #时间上下文分析
    print_line()
    #对用户根据地理位置分组分析
    #user_cluster()
    #use_active_level_analy()
    #use_geo_analy_2()
    #use_geo_analy_3()
    #generate_geo_data()
    pass
    
    
    
    
    