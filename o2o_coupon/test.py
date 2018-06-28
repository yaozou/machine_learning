import pandas as pd
import numpy as np
from datetime import date

off_test = pd.read_csv('test_revised.csv',header=0)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
dataset3 = off_test.dropna(how='any')

def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]),
                                                                                                           int(d[4:6]),
                                                                                                           int(d[
                                                                                                               6:8]))).days
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

def test(s):
    print(type(s))
    return type(s) == float

t6 = dataset3[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t6.rename(columns={'date_received': 'dates'}, inplace=True)

t7 = dataset3[['user_id', 'coupon_id', 'date_received']]
t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-'+t6.dates
print(t7[:5])
test = t7[test(t7['date_received_date'])]
print(test[:5])
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]
print(t7[:5])
