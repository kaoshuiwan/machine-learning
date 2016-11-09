#encoding:utf8
import pandas as pd
import numpy as np
import random

C = 1.0 #regularization parameter
tol = 0.0001 #numerical tolerance
max_passes = 1


def f(alpha, b, X, y, xi):
    tmp = np.array([sum(x*xi) for x in X])
    tmp = tmp*y
    tmp = tmp*alpha
    return sum(tmp)+b

def LandH(alpha_i, alpha_j, yi, yj):
    if yi!=yj:
        L = max(0, alpha_j-alpha_i)
        H = min(C, C+alpha_j-alpha_i)
    else:
        L = max(0, alpha_j+alpha_i-C)
        H = min(C, alpha_j+alpha_i)
    return L,H

def clip(v, L, H):
    if v > H:
        return H
    elif v < L:
        return L
    else:
        return v

def predict(alpha, b, X, y):
    res = np.array([f(alpha, b, X, y, x) for x in X])
    res[res>0] = 1
    res[res<=0] = -1
    return res

def loss(alpha, b, X, y):
    res_y = predict(alpha, b, X, y)
    return len(res_y[res_y==y])*1.0/len(res_y)

def train(X, y):
    alpha = np.array([0.0]*len(X))
    b = 0.0
    passes = 0

    m = len(X)
    index = np.array(range(m))

    while passes < max_passes:
        loss_val = loss(alpha, b, X, y)
        print "passes %d , accuracy %f" % (passes, loss_val)
        num_changed_alphas = 0
        for i in index:
            Ei = f(alpha, b, X, y, X[i]) - y[i]
            if (y[i]*Ei < -tol and alpha[i] < C) or (y[i]*Ei > tol and alpha[i] > 0):
                j = random.choice(index[index!=i])
                Ej = f(alpha, b, X, y, X[j]) - y[j]
                alpha_i = alpha[i]
                alpha_j = alpha[j]
                L,H = LandH(alpha_i, alpha_j, y[i], y[j])
                if L == H:
                    continue
                yita = 2*sum(X[i]*X[j]) - sum(X[i]*X[i]) - sum(X[j]*X[j])
                if yita >= 0:
                    continue
                alpha[j] = alpha[j] - y[j]*(Ei-Ej)/yita
                alpha[j] = clip(alpha[j],L,H)
                if abs(alpha[j]-alpha_j) < 1e-5:
                    continue
                alpha[i] = alpha[i] + y[i]*y[j]*(alpha_j - alpha[j])
                b1 = b - Ei -y[i]*(alpha[i]-alpha_i)*sum(X[i]*X[i]) - y[j]*(alpha[j]-alpha_j)*sum(X[i]*X[j])
                b2 = b - Ej -y[i]*(alpha[i]-alpha_i)*sum(X[i]*X[j]) - y[j]*(alpha[j]-alpha_j)*sum(X[j]*X[j])

                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b
