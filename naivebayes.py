#encoding:utf8
'''
朴素贝叶斯模型
仅支持离散型变量
'''
import pandas as pd
import numpy as np
import math,numbers
from math import log

attr_values = {}

def prob_y(D):
    counts = D['y'].value_counts()*1.0/len(D)
    return counts.to_dict()

def proba_y_a(D, y):
    Dy = D[D['y']==y]
    ret = {}
    for a in attr_values:
        conditional_prob_a = {v : (len(Dy[Dy[a]==v])+1)*1.0/(len(Dy)+len(attr_values)) for v in attr_values[a]}
        ret[a] = conditional_prob_a
    return ret

def generate(D):
    for attr in D:
        if attr == 'y':
            continue
        #统计每个属性的属性取值
        attr_values[attr] = D[attr].values.tolist()
    Py = prob_y(D)
    Pa = {y:proba_y_a(D, y) for y in D['y'].unique()}
    return Py,Pa

def predict(Py, Pa, x):
    max_proba = 0
    max_y = None
    for y in Py:
        tmp = Py[y]
        for a in attr_values:
            tmp *= Pa[y][a][x[a].values[0]]
        if tmp > max_proba:
            max_y = y
            max_proba = tmp
    return max_y

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
