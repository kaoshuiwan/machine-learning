#encoding:utf8
'''
C4.5生成算法生成；
仅支持离散型变量；
未进行剪枝；
没有参数调节，如最小节点数量约束。
'''
import pandas as pd
import numpy as np
import math

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

attr_values = {}

def TreeGenerate(D, A, parent):
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
    #opt_attr_index = SelectPartitionAttr(D, A) #选取最优切分属性
    #opt_attr = A[opt_attr_index]
    opt_attr = SelectPartitionAttr(D, A) #选取最优切分属性
    opt_attr_values = attr_values[opt_attr]

    node = Node(parent =parent, dataset = D)
    node.attr = opt_attr

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
    return node

def Ent(D):
    counts = D['y'].value_counts()*1.0/len(D)
    return sum([-1*pv*math.log(pv) for pv in counts])

def EntGain(D, a):
    Ent_D = Ent(D)
    a_values = D[a].unique()
    tmp = 0
    for v in a_values:
        tmp += (len(D[D[a]==v])*1.0/len(D))*Ent(D[D[a]==v])
    return Ent_D - tmp

def SelectPartitionAttr(D, A):
    max_gain = 0
    opt_a = None
    for a in A:
        gain_a = EntGain(D,a)
        if gain_a > max_gain:
            opt_a = a
            max_gain = gain_a
    return opt_a

def train(D):
    for attr in D:
        if attr == 'y':
            continue
        attr_values[attr] = D[attr].values.tolist()
    A = np.array(list(D)[:-1])
    root = TreeGenerate(D,A,None)
    return root

def predict(root, x):
    if len(root.childs)==0:
        return root.result
    x_attr_value = x[root.attr].values[0]
    return predict(root.childs[x_attr_value], x)

dataset = [
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
