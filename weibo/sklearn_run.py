import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

if __name__ == '__main__':
    train_df = pd.read_csv('Data/train_dataset_updated.csv', encoding='ANSI')
    predict_df = pd.read_csv('Data/predict_dataset_updated.csv', encoding='ANSI')

    target = ['转发', '评论', '赞']
    dropped_train_dataset = ['微博ID', '用户ID', '发送时间', '转发', '评论', '赞', '内容', 'Cache']
    dropped_predict_datastet = ['微博ID', '用户ID', '发送时间', '内容', 'Cache']

    predictors = [x for x in train_df.columns if x not in target + dropped_train_dataset]

    for item in target:
        predict_df[item] = 0

    for i in range(len(target)):
        rf = RandomForestRegressor()  # 这里使用了默认的参数设置
        rf.fit(train_df[predictors], train_df[target[i]])  # 进行模型的训练
        predict_df_predictions = rf.predict(predict_df[predictors])
        predict_df_predictions = [int(item) for item in predict_df_predictions]
        predict_df[target[i]] = predict_df_predictions
        print(predict_df[target[i]])

    result = predict_df.loc[:, ['微博ID', '用户ID', '转发', '评论', '赞']]
    # result.columns=['uid','mid','forward_count','comment_count','like_count']
    result.to_csv('result.csv')