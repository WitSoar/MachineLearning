import numpy as np

def pca(data, n):
    '''
    :param n:
    降维后的维度
    :return accum_num:
    贡献度
    :return redeiVects:
    变换矩阵
    :return mid_data * redeiVects:
    降维后的结果
    '''
    assert n <= len(data)
    mean_val = np.mean(data.T, axis=0)
    mid_data = data.T - mean_val
    cov_mat = np.cov(mid_data, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(cov_mat))
    print(eigVects.real)
    eigVals_ind = np.argsort(eigVals)
    eigVals_ind = eigVals_ind[:-(n + 1):-1]
    redeiVects = eigVects[:, eigVals_ind]
    total = eigVals.sum()
    eigVals.sort()
    accum_val = eigVals[-n:].sum()
    accum_num = accum_val / total
    return accum_num, mid_data * redeiVects, redeiVects