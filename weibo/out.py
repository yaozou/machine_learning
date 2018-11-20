import pandas as pd

result = pd.read_csv("Data/result.csv", encoding='ANSI')
print(result.columns)
result.columns = ['顺序索引', 'mid', 'uid', 'forward_count', 'comment_count', 'like_count']

text = open("Data/out.txt", 'w')
i = 0
len_result = len(result)
for index, item in result.iterrows():
    print((i * 1.0) / len_result)
    mid = str(item['mid'])
    uid = str(item['uid'])
    forward_count = str(item['forward_count'])
    comment_count = str(item['comment_count'])
    like_count = str(item['like_count'])
    out_put = uid + '\t' + mid + '\t' + forward_count + ',' + comment_count + ',' + like_count
    print(out_put, file=text)
text.close()