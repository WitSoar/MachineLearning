import numpy as np

## 这里只实现最简单的离散数据集，连续的需要构造概密函数

def byes_dispersed_train(dataset):
    '''训练模型进行保存'''
    data, label = dataset
    table = {}
    model = {}
    for i in range(len(label)):
        if label[i] not in table:
            table[label[i]] = [data[i]]
            model[label[i]] = []
        else:
            table[label[i]].append(data[i])
    for key in table:
        table[key] = np.array(table[key]).T
        for x in table[key]:
            unique, count = np.unique(x, return_counts=True)
            data_count = dict(zip(unique, count/len(x)))
            model[key].append(data_count)
    return model

def byes_use(data, model):
    '''对上面训练出来的模型进行训练'''
    mid_val = float('-inf')
    res = None
    for key in model:
        judge_list = model[key]
        chance = 1
        for i in range(len(data)):
            if data[i] not in judge_list[i]:
                chance = 0
                break
            else:
                chance *= judge_list[i][data[i]]
        if chance > mid_val:
            mid_val = chance
            res = key
    return res