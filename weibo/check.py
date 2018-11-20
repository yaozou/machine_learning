import pandas as pd

"""
检测数据
"""
class check_Data(object):

    def read_Data(self,filepath):
        data = pd.read_table(filepath,header=None)
        return data

    def data_explore(self,data):
        print(data.describe())


if __name__ == '__main__':
    cd = check_Data()
    data = cd.read_Data('Data/weibo_train_data.txt')
    data.columns = ['用户ID', '微博ID', '发送时间', '转发', '评论', '赞', '内容']
    cd.data_explore(data)
