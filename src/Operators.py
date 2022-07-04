# @Time    : 2022-6-17
# @Author  : zhengzhiyu yinxiaohui
# @File    : Operators.py

from MachineLearning import Sharing

table = {}
method_list = [
    'filter', 'check', 'pretreat', 'train', 'model_accuracy', 'visual',
    'uncertainty', 'use_model'
    ]
method_table = set(method_list)

def shield_func(*args):
    '''This is a function used to return errors. Its main function
    is to shield object methods'''
    raise AttributeError("Your algorithm does not have this method")

def shield(obj, name):
    '''
    :arg:List[str]
    Fill in the way you want to behave
    '''
    for method in method_table:
        if method not in name:
            setattr(obj, method, shield_func)
    return None

def register(name):
    '''Here we define a decorator. We hope that through this decorator,
    we can simply record the implemented functions'''
    def decorate(func):
        table[name] = func
        def wrapped_func(*args, **kwargs):
            func(*args, **kwargs)
        return wrapped_func
    return decorate

class Classifier(object):
    '''As a proxy object for users, this class is used to control
    a series of problems such as algorithm training, visualization
    and data processing. It should be used in every algorithm call'''
    def __init__(self, algorithm, data, *args, **kwargs):
        '''
        :param algorithm:str
        The name of your algorithm
        :param data:List
        The input data here should preferably follow a unified
        specification. We want the input data to contain two elements,
        input data and label, which should be in a list or numpy format
        '''
        self.table = table
        self.data = data
        if algorithm not in self.table:
            raise ValueError("The algorithm does not exist")
        self.work_obj = self.table[algorithm]()

    def filter(self):
        '''Deal with the problem of missing data'''
        pass

    def check(self):
        '''Data hypothesis test'''
        pass

    def pretreat(self, *args, **kwargs):
        '''Data preprocessing '''
        pass

    def train(self, *args, **kwargs):
        return self.work_obj.train(*args, **kwargs)

    def model_accuracy(self, *args, **kwargs):
        return self.work_obj.model_accuracy(*args, **kwargs)

    def visual(self, *args, **kwargs):
        return self.work_obj.train(*args, **kwargs)

    def use_model(self, *args, **kwargs):
        return self.work_obj.train(*args, **kwargs)

    def uncertainty(self, *args, **kwargs):
        return self.work_obj.uncertainty(*args, **kwargs)

'''
关于这个类的设计，我是这么考虑的，首先是目录结构上，本py文件是主文件，供用户引入，
sharing.py文件，是共享区域，一些所有算法通用的方法和常量，放在里面，Tools里面就是
各种算法，这个包不希望用户对其进行导入，这部分导入需要在算法注册的过程中实现，关于函数
注册，这里采用装饰器，一个简单的例子如下：
@register('PCA')
def PCA(obj, *args, **kwargs):
    from MachineLearning.Tools.PCA import * #在这里进行环境配置
    name = [] #这里面填写希望用户可以访问到的方法
    shield(obj, name) #将算法没有的方法封闭掉，不允许外部访问
    #类中的一些方法，我将其分为了常规方法，就是各个算法基本都没有太大区别的
    实现上是pass的，这些算法需要再写
    #另一种是特异方法，不同的算法其实现可能都不同
    中间你可以实现一些你需要的操作，一般不需要参数，但这里还是设计了，为了有些可能的
    特殊用处
    if 你是函数实现：
        obj.train = PCA #将PCA.py文件中的函数实现引用进来
    elif 你是类实现：
        return cls #你的类，上面Classifier的实现，默认你是采用类实现，用函数实现的话，
        按照上述操作，这样保证灵活性，如果你是类实现，需要在你想保留的方法中，实现同名方法
'''