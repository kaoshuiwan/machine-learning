#coding:utf8
from numpy import *
import math
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import KFold
'''
#利用训练的decision stump做分类器，计算Z
alpha:  根据公式计算出的alpha的值
j:      decision stump选择的特征
thresh: 阈值
vote:   向量，判断是否属于某一类
target: 训练数据的结果
data:   训练数据
weight: 权重

返回：
Z:    分类器分类后的Z值
'''
def stumpclassfier(alpha,vote,j,thresh,target,data,weight):
    n,d = data.shape
    scalar = 1
    Z = 0
    for i in range(n):
        if data.loc[i,j]>thresh:
            scalar = 1
        else:
            scalar = -1
        pre = -alpha*vote*scalar*(target.iloc[i].values)
        for k in range(len(pre)):
            Z += weight[i][k]*math.pow(math.e,pre[k])
    return Z

'''
通过遍历每个特征列，训练出d个弱分离器，再通过比较d个弱分类器的优化目标Z值，选出cross-feature的最优弱分类器
返回：
alpha[bestj]:   最优列训练出的分类器计算出的alpha值
vote[bestj]:    
bestj:
thresh[bestj]:  以上三个值是最优的分类函数
Z:              Z值
'''
def stumpbase(data,target,weight):
    n,k = shape(weight) #n为数据个数，k是类数
    d = shape(data)[1]  #d为数据特征维度
    gmma = zeros(k)     #gmma，优化目标
    alpha = zeros(d)    #存放每个特征列计算出的alpha值 
    thresh = zeros(d)   #d个树桩分类的阈值
    vote = zeros((d,k)) #每个列的vote向量
    Z = 0               #优化目标，最小

    #计算初始gmma值，树桩把所有的数据分到右子结点上
    for l in range(k):
        gmma[l] = sum(target.loc[:,l]*weight[:,l])

    #选每个特征列训练出decision stump，再比较各个列训练的decision stump计算的Z值，选出最优decision stump
    for j in range(d):
        #按照特征列j的值对数据排序，s是升序排列后数据矩阵第j列的值
        s = data.sort(j)[j]
        vote[j],thresh[j],gj = beststump(s,target,weight,gmma)
        alpha[j] = math.log((1+gj)/(1-gj))/2
        #print j,gj

    #最优的作为分类依据的特征列
    bestj = 0
    for j in range(d):
        if j==0:
            Z = stumpclassfier(alpha[j],vote[j],j,thresh[j],target,data,weight)
        else:
            tmp = stumpclassfier(alpha[j],vote[j],j,thresh[j],target,data,weight)
            if tmp<Z:
                Z = tmp
                bestj = j
    return alpha[bestj],vote[bestj],bestj,thresh[bestj],Z

'''
根据一个特征列的值s，找出这个列上最优的切分点，优化目标是最大gmma的1-范数

'''
def beststump(s,target,weight,gmma):
    k = weight.shape[1] #类数
    gmmabest = gmma.copy()
    gmmainit = gmma.copy()
    threshVal = 0
    vote = ones(k)

    for j in range(134):
        i = s.index[j]
        for l in range(k):
            gmmainit[l] = gmmainit[l]-2*weight[i][l]*target.loc[i,l]
        if s.iloc[j] != s.iloc[j+1]:
            if sum(abs(gmmainit))>sum(abs(gmmabest)):
                gmmabest = gmmainit.copy()
                threshVal = (s.iloc[j]+s.iloc[j+1])/2

    '''
    最初默认对于三个类来说，小于thresh的全都设成-1。但是到底左节点是-1还是+1是都可行的，目的是让gmma值的和最大，
    如果某一类的gmma值小于0，那么可以通过颠倒左右节点的分类结果，让左节点为+1，右节点为-1，此时gmma值为正值。
    故最后判断gmmabest的每个类的值，如果有负值，就颠倒这个类的左右节点分类结果。这也是为什么是先计算各类绝对值再加和的原因。
    vote默认是[1,1,1],需要颠倒的类设为-1。
    '''
    for l in range(k):
        if gmmabest[l]>0:
            vote[l] = 1
        else:
            vote[l] = -1

    #如果最初的gmma值就是最优，那么切分点<最小的值,即负无穷，所有的点都被且分到右节点
    flag = True
    for g1,g2 in zip(gmmabest,gmma):
        if g1!=g2:
            flag = False
    if flag:
        return vote,-123123231321,sum(abs(gmmabest))
    else:
        return vote,threshVal,sum(abs(gmmabest))

'''
data:   训练数据
target: 训练数据的结果,三维数组
weight: 数据权重
T:      集成弱分类器个数
targetexpected: 数据结果，[0,0,0,1,1,1,2,2....]形式
'''
def adaboostmh(data,target,weight,T,targetexpected):
    n,d = data.shape    #n为数据个数
    k = weight.shape[1]
    alpha = zeros(T+1)  #存储每次迭代产生弱分类器的alpha值，代表了这个弱分类器在最终模型的重要程度
    vote = zeros((T+1,k))   
    bestj = zeros(T+1)
    thresh = zeros(T+1) #以上三个组成每个弱分类器的分类函数
    Z = zeros(T+1)      #每个弱分类器分类后计算出的Z值，用作权值调整
    #迭代10次训练出10个decision stump，然后集成
    for t in range(1,T+1):
        alpha[t],vote[t],bestj[t],thresh[t],Z[t] = stumpbase(data,target,weight)
        for i in range(n):
            #大于阈值落在右节点
            if data.loc[i,bestj[t]]>thresh[t]:
                scalar = 1
            else:
                scalar = -1
            pre = -1*alpha[t]*vote[t]*scalar
            for l in range(k):
                #权值调整
                weight[i][l] = weight[i][l]*(math.pow(math.e,pre[l]*target.loc[i,l]))/Z[t]
        
        #没有误分类点时终止迭代
        predict_target = [classifier(alpha,vote,bestj,thresh,dt,t) for dt in data.values]
        if all(targetexpected==predict_target):
            return alpha,vote,bestj,thresh,t

    return alpha,vote,bestj,thresh,T

'''
分别用T个分类器分类，把分类结果乘上权重后加和，根据三元组中值的符号判断是否属于这一类
'''
def classifier(alpha,vote,bestj,thresh,data,T):
    f = zeros(3)
    for t in range(1,T+1):
        scalar = -1
        if data[bestj[t]]>thresh[t]:
            scalar = 1
        f += (alpha[t]*vote[t]*scalar)
    f = sign(f)
    for i in range(len(f)):
        if f[i]>0:
            return i
