'''
# author: xhs
# data: 2022/4/18
# 测试KFold交叉验证代码
n_splits：int, 默认为5。表示拆分成5折
shuffle： bool, 默认为False。切分数据集之前是否对数据进行洗牌。True洗牌，False不洗牌。
random_state：int, 默认为None 当shuffle为True时，如果random_state为None，则每次运行代码，获得的数据切分都不一样，
random_state指定的时候，则每次运行代码，都能获得同样的切分数据，保证实验可重复。
random_state可按自己喜好设定成整数，如random_state=42较为常用。当设定好后，就不能再更改。
'''
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])

kf = KFold(n_splits=5, shuffle=False)  # 初始化KFold
#
for train_index, test_index in kf.split(X):  # 调用split方法切分数据
    print('train_index:%s , test_index: %s ' % (train_index, test_index))
    fold1_train_data, fold1_train_label = X[train_index], y[train_index]
    # print(fold1_train_data)