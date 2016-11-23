#encoding:utf8
'''
C4.5生成算法生成；
*仅支持离散型变量；
未进行剪枝；
没有参数调节，如最小节点数量约束。
'''
import pandas as pd
import numpy as np
import math,numbers
from math import log
class Node:
    '''
    Represents a decision tree node.
    '''
    def __init__(self, parent = None, dataset = None):
        self.dataset = dataset # 落在该结点的训练实例集
        self.result = None # 结果类标签
        self.attr = None # 该结点的分裂属性ID
        self.childs = {} # 该结点的子树列表，key-value pair: (属性attr的值, 对应的子树)
        self.parent = parent # 该结点的父亲结点

        self.isdiscrete = True
        self.splitval = None

attr_values = {}

def TreeGenerate(D, A, parent, threshold = 0.0001):
    y = D['y'].values
    max_y = D['y'].value_counts().index.values[0]
    #counts = np.bincount(y)

    if len(set(y.tolist()))==1: #D中样本都属于同一类别
        node = Node(parent =parent, dataset = D)
        node.result = y[0]
        return node
    if len(A) == 0 or len(D.drop_duplicates(A))==1: #没有候选属性或者D中候选属性的值都一样
        node = Node(parent =parent, dataset = D)
        node.result = max_y
        return node

    opt_attr,max_gain,split_opt = SelectPartitionAttr(D, A) #选取最优切分属性

    #若信息增益率小于阈值则不再分裂，设为叶节点
    if max_gain < threshold:
        node = Node(parent =parent, dataset = D)
        node.result = max_y
        return node

    opt_attr_values = attr_values[opt_attr]

    node = Node(parent =parent, dataset = D)
    node.attr = opt_attr

    if split_opt is None:
        #若为离散变量
        for v in opt_attr_values:
            #为该属性每一个值划分一个分支
            Dv = D[D[opt_attr]==v]
            if len(Dv) == 0:
                child = Node(parent =node)
                child.result = max_y
                node.childs[v] = child
            else:
                child = TreeGenerate(D[D[opt_attr]==v], A[A!=opt_attr], node)
                node.childs[v] = child
    else:
        #若为连续变量
        child1 = TreeGenerate(D[D[opt_attr]<=split_opt], A, node)
        child2 = TreeGenerate(D[D[opt_attr]>split_opt], A, node)
        node.childs['0'] = child1
        node.childs['1'] = child2
        node.splitval = split_opt
        node.isdiscrete = False

    return node

def log2(v):
    return log(v)/log(2)

def Ent(D):
    counts = D['y'].value_counts()*1.0/len(D)
    return sum([-1*pv*log2(pv) for pv in counts])

def EntGainRatio(D, a):
    #对离散属性a来计算信息增益率
    Ent_D = Ent(D)
    a_values = D[a].unique()
    tmp = 0
    IV = 0
    for v in a_values:
        Dv_ratio = len(D[D[a]==v])*1.0/len(D)
        tmp += (Dv_ratio * Ent(D[D[a]==v]))
        IV += (-1.0*Dv_ratio*log2(Dv_ratio))
    return (Ent_D - tmp)/IV

def EntGainRatio_Continuous(D, a):
    #对连续属性a来计算信息增益率

    vals = D[a].unique()
    vals.sort()
    Ent_D = Ent(D)

    tmp = Ent_D
    split_opt = None
    IV = None #信息增益率 = 信息增益/IV
    for i in range(len(vals)-1):
        #把连续值当作离散变量处理，寻找最优的切分点，使得切分后信息增益最大
        split_val = (vals[i]+vals[i+1])/2
        Dv_ratio = (len(D[D[a]>split_val])*1.0/len(D))
        Dv_ratio2 = (len(D[D[a]<=split_val])*1.0/len(D))
        if tmp > Dv_ratio*Ent(D[D[a]>split_val]) + Dv_ratio2*Ent(D[D[a]<=split_val]):
            tmp = Dv_ratio*Ent(D[D[a]>split_val]) + Dv_ratio2*Ent(D[D[a]<=split_val])
            split_opt = split_val
            IV = -1*(Dv_ratio*log2(Dv_ratio)+Dv_ratio2*log2(Dv_ratio2))

    return split_opt, (Ent_D-tmp)/IV


def SelectPartitionAttr(D, A):
    #寻找最优分裂属性
    max_gain = 0 #最大信息增益率
    opt_a = None #最优属性
    split_opt = None #若opt_a,split_opt是切分值
    for a in A:
        split_val = None
        #若为连续变量
        if len(attr_values[a]) > 5 and isinstance(attr_values[a][0], numbers.Number):
            split_val, gain_a = EntGainRatio_Continuous(D, a)
        else:
            gain_a = EntGainRatio(D,a)

        if gain_a > max_gain:
            split_opt = split_val
            opt_a = a
            max_gain = gain_a
    return opt_a, max_gain, split_opt

def train(D):
    for attr in D:
        if attr == 'y':
            continue
        #统计每个属性的属性取值
        attr_values[attr] = D[attr].values.tolist()
    A = np.array(list(D)[:-1])
    root = TreeGenerate(D,A,None)
    return root

def predict(root, x):
    if len(root.childs)==0:
        return root.result
    x_attr_value = x[root.attr].values[0]
    if root.isdiscrete == False:
        if x_attr_value > root.splitval:
            x_attr_value = '1'
        else:
            x_attr_value = '0'
    return predict(root.childs[x_attr_value], x)

#属性值对应'年龄'，'有工作','有自己的房子','信贷情况','类别'

sample_dataset = [
   ("青年", "否", "否", "一般", "否")
   ,("青年", "否", "否", "好", "否")
   ,("青年", "是", "否", "好", "是")
   ,("青年", "是", "是", "一般", "是")
   ,("青年", "否", "否", "一般", "否")
   ,("中年", "否", "否", "一般", "否")
   ,("中年", "否", "否", "好", "否")
   ,("中年", "是", "是", "好", "是")
   ,("中年", "否", "是", "非常好", "是")
   ,("中年", "否", "是", "非常好", "是")
   ,("老年", "否", "是", "非常好", "是")
   ,("老年", "否", "是", "好", "是")
   ,("老年", "是", "否", "好", "是")
   ,("老年", "是", "否", "非常好", "是")
   ,("老年", "否", "否", "一般", "否")
]

def print_node_cnt(Tree):
    if None == Tree:
        return
    if None != Tree.result:
        print "%d," % len(Tree.dataset)
        return
    for key, value in Tree.childs.items():
        print_node_cnt(value)
