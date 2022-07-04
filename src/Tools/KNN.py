import numpy as np

##尽量直接使用矩阵计算，避免使用for循环
def KNN_sample(x, dataset, k):
    '''
    这里直接采用最简单欧式距离，如果有需要，自行更改
    :param x:
    需要预测的数据
    :param dataset:
    依据数据集
    :param k:
    K值
    :return:
    result
    '''
    data, label = dataset[0], dataset[1]
    distance_mat = (np.tile(x, (len(dataset), 1)) - data)**2
    distance = (distance_mat.sum(axis=1))**0.5
    index = distance.argsort()
    count_dict = {}
    for i in range(k):
        count_dict[label[index[i]]] = count_dict.get(label[index[i]], 0) + 1
    res = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
    return res[0][0]
