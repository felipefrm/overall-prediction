# -*- coding: utf-8
import pandas as pd
import numpy as np

#Funções básicas utilizando matriz

def transposta(matriz):
    mT = np.zeros(shape=(len(matriz[0]),len(matriz)))
    for i in range(len(matriz[0])):
        for j in range(len(matriz)):
            mT[i][j] = matriz[j][i]
    return mT

def multiplica(m1,m2):
    mR = np.zeros(shape=(len(m1),len(m2[0])))
    #print("linhas: "+str(linhas)+" COl2:"+str(col2)+" COl1:"+str(col1))
    for i in range(len(m1)):
        for k in range(len(m2[0])):
            soma = 0
            for j in range(len(m1[0])):
                soma = soma + m1[i][j] * m2[j][k]
            mR[i][k] = soma
    return mR

#Algoritmo de gauss-seidel para resolução de sistemas lineares
def seidel(g,x,y):
    n = len(g)
    for j in range(n):
        d = y[j]
        for i in range(n):
            if j != i:
                d = d - g[j][i]*x[i]
        x[j] = d/g[j][j]
    return x
