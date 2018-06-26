import numpy as np

""""
    分类
"""

from sklearn.datasets import  load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target
print(dataset.DESCR)
n_samples,n_features = X.shape

attribute_means = X.mean(axis=0)
assert attribute_means.shape == (n_features,)
X_d = np.array(X >= attribute_means,dtype="int")

# 训练、测试数据的划分
from sklearn.cross_validation import train_test_split
random_state = 14
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
print("There are {} training samples".format(y_train.shape))
print("There are {} testing samples".format(y_test.shape))

from collections import defaultdict
from operator import itemgetter

def train(X,y_true,feature):
    n_samples,n_features = X.shape
    assert 0 <= feature < n_features
    values = set(X[:,feature])
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class,error =train_feature_value(X,y_true,feature,current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors,total_error

def train_feature_value(X,y_true,feature,value):
    class_counts = defaultdict(int)