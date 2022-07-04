# @Time    : 2022-6-17
# @Author  : zhengzhiyu yinxiaohui
# @File    : Sharing.py
from numpy.random import default_rng

def random_weight(size):
    '''
    Generate random coefficient
    :param  size:int
    '''
    rng = default_rng()
    w = rng.standard_normal(size)
    L2 = ((w*w).sum())**0.5
    return w/L2
