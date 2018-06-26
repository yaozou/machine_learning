import numpy as np

"""
    亲和性分类
"""

"----处理数据集----"
dataset_filename = "affinity_dataset.txt"
X = np.load(dataset_filename)
"获取矩阵的形状 （行列）"
n_samples,n_features = X.shape
print("This dataset has {0} samples and {1} features".format(n_samples,n_features))

print(X[:5])

features = ["bread","milk","cheese","apples","bananas"]

num_apple_purchases = 0 # 苹果购买人数
for sample in X:
    if sample[3] == 1:
        num_apple_purchases += 1
print("{0} people bought Apples".format(num_apple_purchases))

rule_valid = 0 # 购买了苹果又购买了香蕉的人数
rule_invalid = 0
for sample in X:
    if sample[3] == 1:
        if sample[4] == 1:
            rule_valid += 1
        else:
            rule_invalid += 1


support = rule_valid # 支持度
confidence = rule_valid / num_apple_purchases #置信度
print("The support is {0} and the confidence is {1:.3f}.".format(support, confidence))
print("As a percentage, that is {0:.1f}%.".format(100 * confidence))

from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences = defaultdict(int)

#
def print_rule(premise,conclusion,support,confidence,features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")
# 获取到购买了X又购买了Y的置信度和支持度
for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion: continue
            if sample[conclusion] == 1:valid_rules[(premise,conclusion)] += 1
            else: invalid_rules[(premise,conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    confidence[(premise,confidence)] = valid_rules[(premise,confidence)]/num_occurences[premise]

for premise,conclusion in confidence:
    print_rule(premise,conclusion,support,confidence,features)

premise = 1
conclusion = 3
print_rule(premise,conclusion,support,confidence,features)

# 支持度排序
from pprint import pprint
pprint(list(support.items()))

from operator import itemgetter
sorted_support = sorted(support.items(),key=itemgetter(1),reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise,conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

# 置信度排序
sorted_confidence = sorted(confidence.items(),key=itemgetter(1),reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise,conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)

# ------------------------------- #
