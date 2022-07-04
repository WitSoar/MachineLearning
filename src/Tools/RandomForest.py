import math
import random
import copy


def cal_shannon(dataset):
    '''计算香农熵，这里我们希望数据集最后一列为类别'''
    result = 0
    total = len(dataset)
    label_count = {}
    for sample in dataset:
        label_count[sample[-1]] = label_count.get(sample[-1], 0) + 1
    for key in label_count:
        result -= label_count[key]/total*math.log(label_count[key]/total,2)
    return result

# print(cal_shannon(dataset))
# 计算尝试证明函数准确无误

def sep_data(dataset, axis, value):
    '''根据轴和数值划分数据集'''
    result = []
    for sample in dataset:
        if sample[axis] == value:
            mid_list = sample[:axis]
            mid_list.extend(sample[axis+1:])
            result.append(mid_list)
    return result

def choose_best(dataset):
    '''我们希望通过该函数找出信息增益最大的特征'''
    num_feature = len(dataset[0]) - 1
    best_increase = -1
    best_feature = -1
    base_shannon = cal_shannon(dataset)
    best_choice = None
    for i in range(num_feature):
        all_choice = [sample[i] for sample in dataset]
        all_choice = set(all_choice)
        new_shannon = 0
        for choice in all_choice:
            sub_dataset = sep_data(dataset, i, choice)
            prob = len(sub_dataset)/len(dataset)
            new_shannon += prob * cal_shannon(sub_dataset)
        shannon_incr = base_shannon - new_shannon
        if shannon_incr > best_increase:
            best_increase = shannon_incr
            best_feature = i
            best_choice = all_choice
    return best_feature, best_choice


class MyTree(object):
    '''我们在这里构建树结构，方便后面随机森林的构建'''
    def __init__(self):
        self.is_leaf = False
        self.pre_val = ''
        self.child = {}
        self.use_feature = None

def most_number(data, axis):
    '''我们期望得到一个列表的输入，并返回最大数量的值'''
    table = {}
    result = ''
    count = 0
    for sample in data:
        table[sample[axis]] = table.get(sample[axis], 0) + 1
        if table[sample[axis]] > count:
            result = sample[axis]
            count = table[sample[axis]]
    return result

def check_one(data, axis):
    '''该函数检查轴中元素是否单一'''
    table = [sample[axis] for sample in data]
    return len(set(table)) == 1

def train_tree(dataset):
    '''通过这个函数训练出我们要的决策树,这里采用递归方式,深度优先'''
    tree = MyTree()
    if check_one(dataset, -1):#结果相同则为叶节点
        tree.is_leaf = True
        tree.pre_val = dataset[0][-1]
        return tree
    best_feature, all_choice = choose_best(dataset)
    if check_one(dataset, best_feature) or len(dataset) <= 5 or len(dataset[0])==1:#如果属性相同，统计最多结果
        tree.is_leaf = True
        tree.pre_val = most_number(dataset, -1)
        return tree
    tree.use_feature = best_feature
    tree.pre_val = most_number(dataset, -1)
    for value in all_choice:
        new_data = sep_data(dataset, best_feature, value)
        child_tree = train_tree(new_data)
        tree.child[value] = child_tree
    return tree

def use_model(model, dataset):
    '''使用训练出来的模型，然后给出分类结果'''
    result = []
    for sample in dataset:
        distinguish = model
        while not distinguish.is_leaf:
            index = distinguish.use_feature
            try:##防止出现我们没有见过的属性
                distinguish = distinguish.child[sample[index]]
            except:
                break
            mid_list = sample[:index]
            mid_list.extend(sample[index+1:])
            sample = mid_list
        result.append(distinguish.pre_val)
    return result

def pre_data(dataset, max_child=3):
    '''如果我们的数据中含有离散，去将他们进行离散化，当然我们
       希望数据，最后一列是类别'''
    number = len(dataset[0]) - 1
    count = len(dataset)
    result = copy.deepcopy(dataset)
    base_shannon = cal_shannon(dataset)
    node = []
    for i in range(number):
        all_attr = [sample[i] for sample in dataset]
        node.append(None)
        if len(set(all_attr)) >= max_child:
            best_increase = -1
            best_node = 0
            cal_table = [[sample[i], sample[-1]] for sample in dataset]
            cal_table.sort(key=lambda x:x[0])
            for cil in range(1, count):
                # 二分类问题，直接算，不用循环
                shannon_val = cil/count*cal_shannon(cal_table[0:cil]) + (1-cil/count)*cal_shannon(cal_table[cil:])
                shannon_incr = base_shannon - shannon_val
                if shannon_incr > best_increase:
                    best_increase = shannon_incr
                    best_node = cil
            div_val = (cal_table[best_node][0] + cal_table[best_node-1][0])/2
            ### 这里注意，其是没有解决中间几个值相等的问题，这种的话，下面默认归为右侧
            node[-1] = div_val
            for num in range(count):
                if result[num][i] < div_val:
                    result[num][i] = 0
                else:
                    result[num][i] = 1
    return result, node


# 这里算法上进入随机森林的构建
def knuth(dataset):
    '''这里我们采用这个大名鼎鼎的洗牌算法随数据集中的数据进行洗牌'''
    for i in range(len(dataset)-1, -1, -1):
        ran_index = random.randint(0, i)
        mid_val = dataset[i]
        dataset[i] = dataset[ran_index]
        dataset[ran_index] = mid_val
    return dataset

def bootstrap(dataset, number = None):
    '''用于产生随机数据集'''
    if number == None:
        number = len(dataset)
    result = []
    oob_data = set([i for i in range(len(dataset))])
    for i in range(number):
        index = random.randint(0, len(dataset) - 1)
        result.append(dataset[index])
        if index in oob_data:
            oob_data.remove(index)
    return result, oob_data


def random_attr(dataset, max_features = 'log2'):
    '''用于选择构建树的属性，也就是特征采样'''
    number = 0
    if max_features == 'log2':
        number = int(math.log(len(dataset[0])-1,2))
    elif max_features == 'sqrt':
        number = int(math.sqrt(len(dataset[0]-1)))
    table = [[sample[i] for sample in dataset] for i in range(len(dataset[0]))]
    index_table = [i for i in range(len(table)-1)]
    index_table = random.sample(index_table, number)
    index_table.sort()
    result = []
    for i in index_table:
        result.append(table[i])
    result.append(table[-1])
    result = [[sample[i] for sample in result]for i in range(len(result[0]))]
    return result, index_table

#print(random_attr(dataset))

def random_forest(dataset, node_number, random_seed=None, number = None, max_features = 'log2'):
    '''随机森林模型训练'''
    random.seed(random_seed)
    dataset, node = pre_data(dataset)
    dataset = knuth(dataset)
    tree_table = []
    oob_data_list = []
    for i in range(node_number):
        mid_data, oob_data = bootstrap(dataset, number)
        mid_data, index_table = random_attr(mid_data, max_features)
        tree = train_tree(mid_data)
        tree_table.append([tree, index_table])
        oob_data_list.append(oob_data)
    return tree_table, node, oob_data_list

def dispersed_data(dataset, node, seq = None):
    table = copy.deepcopy(dataset)
    if seq != None:
        node = [node[i] for i in seq]
    for i in range(len(node)):
        if node[i] != None:
            for j in range(len(table)):
                if table[j][i] < node[i]:
                    table[j][i] = 0
                else:
                    table[j][i] = 1
    return table

def use_random_forest(tree_table, dataset, node):
    '''使用随机森林模型'''
    result = []
    new_data = dispersed_data(dataset, node)
    for sample in new_data:
        table = []
        for tree, index_table in tree_table:
            mid = [sample[i] for i in index_table]
            ans = use_model(tree, [mid])[0]
            table.append(ans)
        count = 0
        count_table = {}
        re_val = ''
        for val in table:
            count_table[val] = count_table.get(val, 0) + 1
            if count_table[val] > count:
                count = count_table[val]
                re_val = val
        result.append(re_val)
    return result

def acc_test(predict_data, result, tree, node):
    '''测试准确率'''
    print(predict_data)
    pre_result = use_random_forest(tree, predict_data, node)
    count = 0
    for i in range(len(pre_result)):
        if pre_result[i] == result[i]:
            count += 1
    return count/len(pre_result)

def cal_err(result_1, result):
    count = 0
    for i in range(len(result_1)):
        if result_1[i] != result[i]:
            count += 1
    return count/len(result)

def oob_cal(dataset, oob_data_list, tree_table, node, ratio = 0.3):
    '''计算袋外误差'''
    table = [[i,0] for i in range(len(dataset[0])-1)]
    for i in range(len(tree_table)):
        result = [dataset[index][-1] for index in oob_data_list[i]]
        oob_data = [dataset[index] for index in oob_data_list[i]]
        oob_data = [[sample[index] for index in tree_table[i][1]] for sample in oob_data]
        dis_oob_data = dispersed_data(oob_data, node, seq = tree_table[i][1])
        result_1 = use_model(tree_table[i][0], dis_oob_data)
        err_1 = cal_err(result_1, result)
        for j in range(len(tree_table[i][1])):
            mid_data = copy.deepcopy(oob_data)
            for m in range(len(mid_data)):
                mid_data[m][j] += random.normalvariate(mid_data[m][j], ratio*mid_data[m][j])
            dis_mid_data = dispersed_data(mid_data, node, seq=tree_table[i][1])
            result_2 = use_model(tree_table[i][0], dis_mid_data)
            err_2 = cal_err(result_2, result)
            table[tree_table[i][1][j]][1] += abs(err_1-err_2)
    count_table = [0]*len(table)
    for i in range(len(tree_table)):
        for j in range(len(tree_table[i][1])):
            count_table[tree_table[i][1][j]] += 1
    for i in range(len(table)):
        assert count_table[i] != 0
        table[i][1] = table[i][1]/count_table[i]
    table.sort(key=lambda x:x[1], reverse=True)
    return table

if __name__ == '__main__':
    path = 'wine.data'
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_list = line.rstrip().split(',')
        val = line_list[0]
        line_int = [float(line_list[i]) for i in range(1, len(line_list))]
        line_int.append(val)
        data.append(line_int)
    predict_data = [sample[:-1] for sample in data]
    result = [sample[-1] for sample in data]
    print(predict_data, result)
    tree, node, oob_data_list = random_forest(data, 1000)
    print(acc_test(predict_data, result, tree, node))
    print('-----------------------------')
    print(oob_cal(data, oob_data_list, tree, node))