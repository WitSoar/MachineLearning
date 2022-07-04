import numpy as np
import random
from MachineLearning.Sharing import random_weight

def train_perc(data, rate, max_iter):
    '''
    :param data:List
    Include data and labels
    :param rate:float
    Learning rate
    :param max_iter:int
    Maximum iterations
    :return loss:List[float]
    '''
    #准备工作
    input_data, label = data[0], data[1]
    w = random_weight(len(input_data[0]))
    b = random.random()
    loss_list = []
    n = 0
    while n < max_iter:
        #寻找错误集
        loss = 0
        delta_w = np.zeros(w.shape)
        delta_b = 0
        for i in range(len(label)):
            res = 1 if np.dot(input_data[i], w.T) + b > 0 else -1
            print(res)
            if res != label[i]:
                loss += res*np.dot(input_data[i], w.T)
                delta_w -= rate*res*input_data[i]#这里和书本不太一样的是采用了批梯度下降
                delta_b -= rate*res
        loss_list.append(loss)
        if loss == 0:
            break
        w += delta_w
        b += delta_b
        n += 1
    return w, b, loss_list

## 对偶问题
def train_perc_dual(data, rate, max_iter):
    '''
    Using the dual problem, the above constraints are solved
    '''
    input_data, label = data[0], data[1]
    b = 0
    w = np.zeros(len(input_data[0]))
    weight = np.zeros(len(input_data))
    count = 0
    #求解Gram矩阵
    table = np.zeros((len(input_data), len(input_data)))
    for i in range(len(input_data)):
        for j in range(len(input_data)):
            table[i][j] = label[j]*np.dot(input_data[i], input_data[j].T)
    #误分条件
    while count < max_iter:
        for i in range(len(label)):
            loss = label[i]*(np.dot(table[i], weight.T) + b)
            if loss <=  0:
                weight[i] += rate
                b += label[i]
                count += 1
                break
        else:
            break
    return weight, b
