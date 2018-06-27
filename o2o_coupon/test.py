import pandas as pd
import numpy as np

ipl_data1 = {'user_id': [1002625, 100330, 1003825,1008100,1010689],
         'coupon_id': [4806, 8635, 13870,4727,7965],
         'max_date_received': [20160726,20160703,20160731,20160721,20160718],
         'min_date_received':[20160712,20160702,20160716,20160711,20160712]}
t2 = pd.DataFrame(ipl_data1)
print (t2)

ipl_data2 = {'user_id': [4129537, 6949378, 2166529,2166529],
         'coupon_id': [9983, 3429, 6928,1808],
         'date_received': [20160712,20160706,20160727,20160727]}
t3 = pd.DataFrame(ipl_data2)
print (t3)
#填充缺失值
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left').fillna(0)
#去掉缺失行
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left').dropna(how='any')
print (t3)