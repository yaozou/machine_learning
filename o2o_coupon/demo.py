import numpy as np
import pandas as pd
from datetime import date
import math
"""
    思路：滑块法
    1、特征数据划分                            
                       预测区间                          特征区间
        【dataset3】20160701~20160731 (113640)   【feature3】20160315~20160630  （测试集）
        【dataset2】20160515~20160615 (258446)   【feature2】20160201~20160514  （训练集2）
        【dataset1】20160414~20160514 (138303)   【feature1】20160101~20160413  （训练集1）
    2、8个，从训练集1、训练集2和测试集的预测区间提取： 
      this_month_user_receive_all_coupon_count 这个月用户收取的所有优惠券数目 
      this_month_user_receive_same_coupon_count 这个月用户收到的相同优惠券的数量 
      this_month_user_receive_same_coupon_lastone 这个月优惠券最远接受时间 
      this_month_user_receive_same_coupon_firstone 这个月优惠券最近接受时间 
      this_day_user_receive_all_coupon_count 一天内用户接收到所有优惠券的数量 
      this_day_user_receive_same_coupon_count 一天内用户接收到相同优惠券的数量 
      day_gap_before 用户上一次领取优惠券的时间间隔 
      day_gap_after  (receive the same coupon)用户下一次领取优惠券的时间间隔
      
    3、商户相关特征，9个，从训练集1、训练集2和测试集的特征区间提取： 
        total_sales # 显示每个商品的销售数量
        sales_use_coupon # 显示使用了优惠券消费的商品
        total_coupon # 显示了商品的优惠券的总数量
        coupon_rate = sales_use_coupon/total_sales # 卖出商品中使用优惠券的占比
        merchant_coupon_transfer_rate = sales_use_coupon/total_coupon # 优惠券的使用率

        merchant_mean_distance # 所有使用优惠券消费的用户与商户的距离平均值
        merchant_median_distance # 所有使用优惠券消费的用户与商户的距离中位值
        merchant_min_distance # 所有使用优惠券消费的用户与商户的距离最小值
        merchant_max_distance # 所有使用优惠券消费的用户与商户的距离最大值

    4、优惠券相关特征，8个，从训练集1、训练集2和测试集的预测区间提取： 
        discount_rate # 优惠券折扣率
        discount_man # 显示满了多少钱后开始减
        discount_jian # 显示满减的减少的钱
        is_man_jian # 返回优惠券是否是满减券
        day_of_week # 显示时间是第几周
        day_of_month # 显示时间是几月
        days_distance# 优惠券领取日期和截止日之间的间隔天数
        is_weekend //优惠券领取日期是否属于周末
      
    5、用户相关特征，14个，从训练集1、训练集2和测试集的特征区间提取：
        count_merchant # 用户消费商户数量
        user_avg_distance # 所有使用优惠券消费的商户与用户的平均距离
        user_min_distance # 所有使用优惠券消费的商户与用户的最小距离
        user_max_distance # 所有使用优惠券消费的商户与用户的最大距离
        user_median_distance # 所有使用优惠券消费的商户与用户的中位距离
        buy_use_coupon # 每个用户使用优惠券消费次数
        buy_total # 用户消费次数
        coupon_received # 用户领取优惠券次数
        avg_user_date_datereceived_gap # 用户从领取优惠券到消费的平均时间间隔
        min_user_date_datereceived_gap # 用户从领取优惠券到消费的最小时间间隔
        max_user_date_datereceived_gap # 用户从领取优惠券到消费的最大时间间隔
        user_coupon_transfer_rate = buy_use_coupon/coupon_received # 用户优惠券转化为实际消费比例
        buy_use_coupon_rate = buy_use_coupon/buy_total # 用户使用优惠券消费占总消费的比例
        user_date_datereceived_gap # 接受到优惠券的日期和使用之间的间隔

    6、用户-商户相关特征，9个，从训练样本date,date_received提取特征： 
        user_merchant_buy_total # 用户在商户消费次数
        user_merchant_received # 用户领取商户优惠券次数
        user_merchant_buy_use_coupon # 用户在商户使用优惠券消费次数
        user_merchant_any # 用户在商户的所有消费次数
        user_merchant_buy_common # 用户在商户普通消费次数
        user_merchant_coupon_transfer_rate # 用户对商户的优惠券转化率
        user_merchant_coupon_buy_rate # 用户对商户使用优惠券消费占总消费比例
        user_merchant_rate # 用户对商户消费占总交互比例
        user_merchant_common_buy_rate # 用户对商户普通消费占总消费比例

"""

"""
 1、特征数据划分                            
                       预测区间                          特征区间
        【dataset3】20160701~20160731 (113640)   【feature3】20160315~20160630  （测试集）
        【dataset2】20160515~20160615 (258446)   【feature2】20160201~20160514  （训练集2）
        【dataset1】20160414~20160514 (138303)   【feature1】20160101~20160413  （训练集1）
"""
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv',header=0)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv',header=0)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train = pd.read_csv('data/ccf_online_stage1_train.csv',header=0)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']

dataset3 = off_test.dropna(how='any')
feature3 = off_train[
        ((off_train.date>='20160315')&(off_train.date<='20160630'))|
        ((off_train.date=='null')&(off_train.date_received>='20160315')&
         (off_train.date_received<='20160630'))].dropna(how='any')

dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')].dropna(how='any')
feature2 = off_train[
        (off_train.date>='20160201')&(off_train.date<='20160514')|
        ((off_train.date=='null')&(off_train.date_received>='20160201')&
         (off_train.date_received<='20160514'))].dropna(how='any')

dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')].dropna(how='any')
feature1 = off_train[
        (off_train.date>='20160101')&(off_train.date<='20160413')|
        ((off_train.date=='null')&(off_train.date_received>='20160101')&
         (off_train.date_received<='20160413'))].dropna(how='any')

"""
    2、8个，从训练集1、训练集2和测试集的预测区间提取： 
      this_month_user_receive_all_coupon_count 这个月用户收取的所有优惠券数目 
      this_month_user_receive_same_coupon_count 这个月用户收到的相同优惠券的数量 
      this_month_user_receive_same_coupon_lastone 这个月优惠券最远接受时间 
      this_month_user_receive_same_coupon_firstone 这个月优惠券最近接受时间 
      this_day_user_receive_all_coupon_count 一天内用户接收到所有优惠券的数量 
      this_day_user_receive_same_coupon_count 一天内用户接收到相同优惠券的数量 
      day_gap_before 用户上一次领取优惠券的时间间隔 
      day_gap_after  (receive the same coupon)用户下一次领取优惠券的时间间隔
"""
def is_firstlastone(x):
    if x == 0:
         return 1
    elif x > 0:
        return 0
    else:
        return -1  # those only receive once

def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]),
                                                                                                           int(d[4:6]),
                                                                                                           int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]),
                                                                       int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)

def dealDataset3(dataset3):
    # dataset3
    t = dataset3[['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset3[['user_id', 'coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset3[['user_id', 'coupon_id', 'date_received']]
    t2.date_received = t2.date_received.astype('str')
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset3[['user_id', 'coupon_id', 'date_received']]
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left').dropna(how='any')
    t3.date_received = t3.date_received.astype('float')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    t4 = dataset3[['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    t5 = dataset3[['user_id', 'coupon_id', 'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    t6 = dataset3[['user_id', 'coupon_id', 'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset3[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left').dropna(how='any')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]
    print(t7[:5])

    other_feature3 = pd.merge(t1, t, on='user_id').dropna(how='any')
    other_feature3 = pd.merge(other_feature3, t3, on=['user_id', 'coupon_id']).dropna(how='any')
    other_feature3 = pd.merge(other_feature3, t4, on=['user_id', 'date_received']).dropna(how='any')
    other_feature3 = pd.merge(other_feature3, t5, on=['user_id', 'coupon_id', 'date_received']).dropna(how='any')
    other_feature3 = pd.merge(other_feature3, t7, on=['user_id', 'coupon_id', 'date_received']).dropna(how='any')
    other_feature3.to_csv('data/other_feature3.csv', index=None)
    print(other_feature3.shape)

def dealDataset2(dataset2):
    # for dataset2
    t = dataset2[['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset2[['user_id', 'coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset2[['user_id', 'coupon_id', 'date_received']]
    t2.date_received = t2.date_received.astype('str')
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset2[['user_id', 'coupon_id', 'date_received']]
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received
    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    t4 = dataset2[['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    t5 = dataset2[['user_id', 'coupon_id', 'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    t6 = dataset2[['user_id', 'coupon_id', 'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset2[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    other_feature2 = pd.merge(t1, t, on='user_id')
    other_feature2 = pd.merge(other_feature2, t3, on=['user_id', 'coupon_id'])
    other_feature2 = pd.merge(other_feature2, t4, on=['user_id', 'date_received'])
    other_feature2 = pd.merge(other_feature2, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature2 = pd.merge(other_feature2, t7, on=['user_id', 'coupon_id', 'date_received'])
    other_feature2.to_csv('data/other_feature2.csv', index=None)
    print(other_feature2.shape)

def dealDataset1(dataset1):
    # for dataset1
    t = dataset1[['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset1[['user_id', 'coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset1[['user_id', 'coupon_id', 'date_received']]
    t2.date_received = t2.date_received.astype('str')
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset1[['user_id', 'coupon_id', 'date_received']]
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    t4 = dataset1[['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    t5 = dataset1[['user_id', 'coupon_id', 'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    t6 = dataset1[['user_id', 'coupon_id', 'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset1[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    other_feature1 = pd.merge(t1, t, on='user_id')
    other_feature1 = pd.merge(other_feature1, t3, on=['user_id', 'coupon_id'])
    other_feature1 = pd.merge(other_feature1, t4, on=['user_id', 'date_received'])
    other_feature1 = pd.merge(other_feature1, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature1 = pd.merge(other_feature1, t7, on=['user_id', 'coupon_id', 'date_received'])
    other_feature1.to_csv('data/other_feature1.csv', index=None)
    print(other_feature1.shape)

dealDataset3(dataset3)
dealDataset2(dataset2)
dealDataset1(dataset1)

"""
3、商户相关特征，9个，从训练集1、训练集2和测试集的特征区间提取： 
        total_sales # 显示每个商品的销售数量
        sales_use_coupon # 显示使用了优惠券消费的商品
        total_coupon # 显示了商品的优惠券的总数量
        coupon_rate = sales_use_coupon/total_sales # 卖出商品中使用优惠券的占比
        merchant_coupon_transfer_rate = sales_use_coupon/total_coupon # 优惠券的使用率

        merchant_mean_distance # 所有使用优惠券消费的用户与商户的距离平均值
        merchant_median_distance # 所有使用优惠券消费的用户与商户的距离中位值
        merchant_min_distance # 所有使用优惠券消费的用户与商户的距离最小值
        merchant_max_distance # 所有使用优惠券消费的用户与商户的距离最大值

"""
def merchantRelatedFeature():
    # for dataset3
    merchant3 = feature3[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

    t = merchant3[['merchant_id']]
    t.drop_duplicates(inplace=True)

    t1 = merchant3[merchant3.date != 'null'][['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    t2 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][['merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    t3 = merchant3[merchant3.coupon_id != 'null'][['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    t4 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][['merchant_id', 'distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant3_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t2, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t3, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t5, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t6, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t7, on='merchant_id', how='left')
    merchant3_feature = pd.merge(merchant3_feature, t8, on='merchant_id', how='left')
    merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype(
        'float') / merchant3_feature.total_coupon
    merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype(
        'float') / merchant3_feature.total_sales
    merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    merchant3_feature.to_csv('data/merchant3_feature.csv', index=None)

    # for dataset2
    merchant2 = feature2[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

    t = merchant2[['merchant_id']]
    t.drop_duplicates(inplace=True)

    t1 = merchant2[merchant2.date != 'null'][['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    t2 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    t3 = merchant2[merchant2.coupon_id != 'null'][['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    t4 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id', 'distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant2_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t2, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t3, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t5, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t6, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t7, on='merchant_id', how='left')
    merchant2_feature = pd.merge(merchant2_feature, t8, on='merchant_id', how='left')
    merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype(
        'float') / merchant2_feature.total_coupon
    merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype(
        'float') / merchant2_feature.total_sales
    merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    merchant2_feature.to_csv('data/merchant2_feature.csv', index=None)

    # for dataset1
    merchant1 = feature1[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

    t = merchant1[['merchant_id']]
    t.drop_duplicates(inplace=True)

    t1 = merchant1[merchant1.date != 'null'][['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    t2 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][['merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    t3 = merchant1[merchant1.coupon_id != 'null'][['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    t4 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][['merchant_id', 'distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant1_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t2, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t3, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t5, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t6, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t7, on='merchant_id', how='left')
    merchant1_feature = pd.merge(merchant1_feature, t8, on='merchant_id', how='left')
    merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    merchant1_feature['merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype(
        'float') / merchant1_feature.total_coupon
    merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype(
        'float') / merchant1_feature.total_sales
    merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    merchant1_feature.to_csv('data/merchant1_feature.csv', index=None)

"""
4、优惠券相关特征，8个，从训练集1、训练集2和测试集的预测区间提取： 
        discount_rate # 优惠券折扣率
        discount_man # 显示满了多少钱后开始减
        discount_jian # 显示满减的减少的钱
        is_man_jian # 返回优惠券是否是满减券
        day_of_week # 显示时间是第几周
        day_of_month # 显示时间是几月
        days_distance# 优惠券领取日期和截止日之间的间隔天数
        is_weekend //优惠券领取日期是否属于周末
"""
def calc_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


def get_discount_man(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[0])


def get_discount_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[1])


def is_man_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 0
    else:
        return 1

def couponRelatedFeature (dataset3,dataset2,dataset1):
    # dataset3
    dataset3['day_of_week'] = dataset3.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    dataset3['day_of_month'] = dataset3.date_received.astype('str').apply(lambda x: int(x[6:8]))
    dataset3['days_distance'] = dataset3.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days)
    dataset3['discount_man'] = dataset3.discount_rate.apply(get_discount_man)
    dataset3['discount_jian'] = dataset3.discount_rate.apply(get_discount_jian)
    dataset3['is_man_jian'] = dataset3.discount_rate.apply(is_man_jian)
    dataset3['discount_rate'] = dataset3.discount_rate.apply(calc_discount_rate)
    d = dataset3[['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset3 = pd.merge(dataset3, d, on='coupon_id', how='left')
    dataset3.to_csv('data/coupon3_feature.csv', index=None)
    # dataset2
    dataset2['day_of_week'] = dataset2.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    dataset2['day_of_month'] = dataset2.date_received.astype('str').apply(lambda x: int(x[6:8]))
    dataset2['days_distance'] = dataset2.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 5, 14)).days)
    dataset2['discount_man'] = dataset2.discount_rate.apply(get_discount_man)
    dataset2['discount_jian'] = dataset2.discount_rate.apply(get_discount_jian)
    dataset2['is_man_jian'] = dataset2.discount_rate.apply(is_man_jian)
    dataset2['discount_rate'] = dataset2.discount_rate.apply(calc_discount_rate)
    d = dataset2[['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset2 = pd.merge(dataset2, d, on='coupon_id', how='left')
    dataset2.to_csv('data/coupon2_feature.csv', index=None)
    # dataset1
    dataset1['day_of_week'] = dataset1.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    dataset1['day_of_month'] = dataset1.date_received.astype('str').apply(lambda x: int(x[6:8]))
    dataset1['days_distance'] = dataset1.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 4, 13)).days)
    dataset1['discount_man'] = dataset1.discount_rate.apply(get_discount_man)
    dataset1['discount_jian'] = dataset1.discount_rate.apply(get_discount_jian)
    dataset1['is_man_jian'] = dataset1.discount_rate.apply(is_man_jian)
    dataset1['discount_rate'] = dataset1.discount_rate.apply(calc_discount_rate)
    d = dataset1[['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset1 = pd.merge(dataset1, d, on='coupon_id', how='left')
    dataset1.to_csv('data/coupon1_feature.csv', index=None)

"""
5、用户相关特征，14个，从训练集1、训练集2和测试集的特征区间提取：
        count_merchant # 用户消费商户数量
        user_avg_distance # 所有使用优惠券消费的商户与用户的平均距离
        user_min_distance # 所有使用优惠券消费的商户与用户的最小距离
        user_max_distance # 所有使用优惠券消费的商户与用户的最大距离
        user_median_distance # 所有使用优惠券消费的商户与用户的中位距离
        buy_use_coupon # 每个用户使用优惠券消费次数
        buy_total # 用户消费次数
        coupon_received # 用户领取优惠券次数
        avg_user_date_datereceived_gap # 用户从领取优惠券到消费的平均时间间隔
        min_user_date_datereceived_gap # 用户从领取优惠券到消费的最小时间间隔
        max_user_date_datereceived_gap # 用户从领取优惠券到消费的最大时间间隔
        user_coupon_transfer_rate = buy_use_coupon/coupon_received # 用户优惠券转化为实际消费比例
        buy_use_coupon_rate = buy_use_coupon/buy_total # 用户使用优惠券消费占总消费的比例
        user_date_datereceived_gap # 接受到优惠券的日期和使用之间的间隔


"""
def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                        int(s[1][6:8]))).days
def userRelatedFeature():
    # for dataset3
    user3 = feature3[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    t = user3[['user_id']]
    t.drop_duplicates(inplace=True)

    t1 = user3[user3.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    t2 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    t7 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    t8 = user3[user3.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    t9 = user3[user3.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    t10 = user3[(user3.date_received != 'null') & (user3.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user3_feature = pd.merge(t, t1, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t3, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t4, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t5, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t6, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t7, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t8, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t9, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t11, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t12, on='user_id', how='left')
    user3_feature = pd.merge(user3_feature, t13, on='user_id', how='left')
    user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan, 0)
    user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan, 0)
    user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype(
        'float') / user3_feature.buy_total.astype(
        'float')
    user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype(
        'float') / user3_feature.coupon_received.astype('float')
    user3_feature.buy_total = user3_feature.buy_total.replace(np.nan, 0)
    user3_feature.coupon_received = user3_feature.coupon_received.replace(np.nan, 0)
    user3_feature.to_csv('data/user3_feature.csv', index=None)

    # for dataset2
    user2 = feature2[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    t = user2[['user_id']]
    t.drop_duplicates(inplace=True)

    t1 = user2[user2.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    t2 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    t7 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    t8 = user2[user2.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    t9 = user2[user2.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    t10 = user2[(user2.date_received != 'null') & (user2.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user2_feature = pd.merge(t, t1, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t8, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t12, on='user_id', how='left')
    user2_feature = pd.merge(user2_feature, t13, on='user_id', how='left')
    user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan, 0)
    user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan, 0)
    user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype(
        'float') / user2_feature.buy_total.astype(
        'float')
    user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype(
        'float') / user2_feature.coupon_received.astype('float')
    user2_feature.buy_total = user2_feature.buy_total.replace(np.nan, 0)
    user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan, 0)
    user2_feature.to_csv('data/user2_feature.csv', index=None)

    # for dataset1
    user1 = feature1[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    t = user1[['user_id']]
    t.drop_duplicates(inplace=True)

    t1 = user1[user1.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    t2 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    t7 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    t8 = user1[user1.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    t9 = user1[user1.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    t10 = user1[(user1.date_received != 'null') & (user1.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user1_feature = pd.merge(t, t1, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t3, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t4, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t5, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t6, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t7, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t8, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t9, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t11, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t12, on='user_id', how='left')
    user1_feature = pd.merge(user1_feature, t13, on='user_id', how='left')
    user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan, 0)
    user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan, 0)
    user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype(
        'float') / user1_feature.buy_total.astype(
        'float')
    user1_feature['user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype(
        'float') / user1_feature.coupon_received.astype('float')
    user1_feature.buy_total = user1_feature.buy_total.replace(np.nan, 0)
    user1_feature.coupon_received = user1_feature.coupon_received.replace(np.nan, 0)
    user1_feature.to_csv('data/user1_feature.csv', index=None)

"""
    6、用户-商户相关特征，9个，从训练样本date,date_received提取特征： 
        user_merchant_buy_total # 用户在商户消费次数
        user_merchant_received # 用户领取商户优惠券次数
        user_merchant_buy_use_coupon # 用户在商户使用优惠券消费次数
        user_merchant_any # 用户在商户的所有消费次数
        user_merchant_buy_common # 用户在商户普通消费次数
        user_merchant_coupon_transfer_rate # 用户对商户的优惠券转化率
        user_merchant_coupon_buy_rate # 用户对商户使用优惠券消费占总消费比例
        user_merchant_rate # 用户对商户消费占总交互比例
        user_merchant_common_buy_rate # 用户对商户普通消费占总消费比例
"""
def userMerchantRelatedFeature():
    # for dataset3
    all_user_merchant = feature3[['user_id', 'merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    t = feature3[['user_id', 'merchant_id', 'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    t1 = feature3[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    t2 = feature3[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = feature3[['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    t4 = feature3[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant3 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant3 = pd.merge(user_merchant3, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_received.astype('float')
    user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')
    user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype(
        'float') / user_merchant3.user_merchant_any.astype('float')
    user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype(
        'float') / user_merchant3.user_merchant_buy_total.astype('float')
    user_merchant3.to_csv('data/user_merchant3.csv', index=None)

    # for dataset2
    all_user_merchant = feature2[['user_id', 'merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    t = feature2[['user_id', 'merchant_id', 'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    t1 = feature2[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    t2 = feature2[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = feature2[['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    t4 = feature2[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant2 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant2 = pd.merge(user_merchant2, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant2 = pd.merge(user_merchant2, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant2 = pd.merge(user_merchant2, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant2 = pd.merge(user_merchant2, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant2.user_merchant_received.astype('float')
    user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant2.user_merchant_buy_total.astype('float')
    user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype(
        'float') / user_merchant2.user_merchant_any.astype('float')
    user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype(
        'float') / user_merchant2.user_merchant_buy_total.astype('float')
    user_merchant2.to_csv('data/user_merchant2.csv', index=None)

    # for dataset2
    all_user_merchant = feature1[['user_id', 'merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    t = feature1[['user_id', 'merchant_id', 'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    t1 = feature1[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    t2 = feature1[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = feature1[['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    t4 = feature1[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant1 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant1 = pd.merge(user_merchant1, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant1 = pd.merge(user_merchant1, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant1 = pd.merge(user_merchant1, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant1 = pd.merge(user_merchant1, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant1.user_merchant_received.astype('float')
    user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant1.user_merchant_buy_total.astype('float')
    user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype(
        'float') / user_merchant1.user_merchant_any.astype('float')
    user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype(
        'float') / user_merchant1.user_merchant_buy_total.astype('float')
    user_merchant1.to_csv('data/user_merchant1.csv', index=None)

couponRelatedFeature (dataset3,dataset2,dataset1)
merchantRelatedFeature()
userRelatedFeature()
userMerchantRelatedFeature()

##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0] == 'null':
        return 0
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                      int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1


coupon3 = pd.read_csv('data/coupon3_feature.csv')
merchant3 = pd.read_csv('data/merchant3_feature.csv')
user3 = pd.read_csv('data/user3_feature.csv')
user_merchant3 = pd.read_csv('data/user_merchant3.csv')
other_feature3 = pd.read_csv('data/other_feature3.csv')
dataset3 = pd.merge(coupon3, merchant3, on='merchant_id', how='left')
dataset3 = pd.merge(dataset3, user3, on='user_id', how='left')
dataset3 = pd.merge(dataset3, user_merchant3, on=['user_id', 'merchant_id'], how='left')
dataset3 = pd.merge(dataset3, other_feature3, on=['user_id', 'coupon_id', 'date_received'], how='left')
dataset3.drop_duplicates(inplace=True)
print(dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan, 0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan, 0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan, 0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3, weekday_dummies], axis=1)
dataset3.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1, inplace=True)
dataset3 = dataset3.replace('null', np.nan)
dataset3.to_csv('data/dataset3.csv', index=None)

coupon2 = pd.read_csv('data/coupon2_feature.csv')
merchant2 = pd.read_csv('data/merchant2_feature.csv')
user2 = pd.read_csv('data/user2_feature.csv')
user_merchant2 = pd.read_csv('data/user_merchant2.csv')
other_feature2 = pd.read_csv('data/other_feature2.csv')
dataset2 = pd.merge(coupon2, merchant2, on='merchant_id', how='left')
dataset2 = pd.merge(dataset2, user2, on='user_id', how='left')
dataset2 = pd.merge(dataset2, user_merchant2, on=['user_id', 'merchant_id'], how='left')
dataset2 = pd.merge(dataset2, other_feature2, on=['user_id', 'coupon_id', 'date_received'], how='left')
dataset2.drop_duplicates(inplace=True)
print(dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan, 0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan, 0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan, 0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2, weekday_dummies], axis=1)
dataset2['label'] = dataset2.date.astype('str') + ':' + dataset2.date_received.astype('str')
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
              inplace=True)
dataset2 = dataset2.replace('null', np.nan)
dataset2.to_csv('data/dataset2.csv', index=None)

coupon1 = pd.read_csv('data/coupon1_feature.csv')
merchant1 = pd.read_csv('data/merchant1_feature.csv')
user1 = pd.read_csv('data/user1_feature.csv')
user_merchant1 = pd.read_csv('data/user_merchant1.csv')
other_feature1 = pd.read_csv('data/other_feature1.csv')
dataset1 = pd.merge(coupon1, merchant1, on='merchant_id', how='left')
dataset1 = pd.merge(dataset1, user1, on='user_id', how='left')
dataset1 = pd.merge(dataset1, user_merchant1, on=['user_id', 'merchant_id'], how='left')
dataset1 = pd.merge(dataset1, other_feature1, on=['user_id', 'coupon_id', 'date_received'], how='left')
dataset1.drop_duplicates(inplace=True)
print(dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan, 0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan, 0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan, 0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1, weekday_dummies], axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' + dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
              inplace=True)
dataset1 = dataset1.replace('null', np.nan)
dataset1.to_csv('data/dataset1.csv', index=None)
