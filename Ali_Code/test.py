import pandas as pd
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x-self.X) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            count = np.bincount(self.y[index])
            result.append(count.argmax())
        return result

    def predict2(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            count = np.bincount(self.y[index], weights=1/dis[index])
            result.append(count.argmax())
        return result

def test_main():
    data = pd.read_csv(r'data/iris.csv')
    # 将类别文本映射成数值类型
    data['Species'] = data['Species'].map({'virginica': 0, 'setosa': 1, 'versicolor': 2})
    # 删除不需要的一列
    data.drop('Id', axis=1, inplace=True)
    # 查看重复数据
    data.duplicated().any()
    # 删除重复数据
    data.drop_duplicates(inplace=True)

    # 提取出每个类别的花数据
    t0 = data[data['Species'] == 0]
    t1 = data[data['Species'] == 1]
    t2 = data[data['Species'] == 2]

    # 对每个类别的数据进行打乱
    t0 = t0.sample(len(t0), random_state=0)
    t1 = t1.sample(len(t1), random_state=0)
    t2 = t2.sample(len(t2), random_state=0)

    # 构造训练集和测试集
    train_X = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
    test_X = pd.concat([t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)
    # 创建knn对象，进行训练
    knn = KNN(k=3)
    # 进行训练
    knn.fit(train_X, train_y)
    # 进行测试
    result = knn.predict(test_X)

    print(np.sum(result == test_y)/len(result))
