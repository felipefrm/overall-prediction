# -*- coding: utf-8
import numpy as np
import pandas as pd
from Matrix import*

# Lê o arquivo csv
df = pd.read_csv('data.csv')
# Seleciona apenas as features que interessam o estudo
dataframe = pd.DataFrame(df, columns = ['ID','Position','InternationalReputation','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes','Overall'])
# Na feature 'Position' seleciona apenas os que possui o valor 'GK'
dfGK = dataframe[dataframe['Position'] == 'GK']
# Remove a feature 'Position' (não é mais necessária) e randomiza as linhas.
dfGK = dfGK.drop('Position', axis = 1).sample(frac=1)
# overall recebe a lista de valores do atributo 'Overall'
overall = dfGK['Overall'].values
# atributos recebe a matriz com o restante dos atributos excluindo o 'Overall'
atributos = dfGK.drop('Overall', axis = 1).values

# Transforma overall de array de uma dimensão para matriz com 1 coluna
overall = np.reshape(overall,(len(atributos), 1))

# Troca o id por 1, x ^ 0
for i in range(len(atributos)):
    atributos[i][0] = 1

# 70% dos dados é separado para treino do algoritmo
treino = int((len(atributos)/10) * 7)
# Cria uma matriz g e y vazia
g = np.zeros(shape=(treino, len(atributos[0])))
y = np.zeros(shape=(treino, 1))

# Atribui os dados de treino às matriz g e y
for i in range(treino):
    g[i] = atributos[i]
    y[i] = overall[i]

# Calcula transposta de g
gt = transposta(g)
# Multiplica gt * g
gtg = multiplica(gt, g)
# Multiplica gt * y
gty = multiplica(gt, y)

# Cria a matriz de parametros x, e aplica o algoritmo de gauss-seidel para o sistema gTgx = gTy
x = np.zeros(shape=(len(g[0])))
for i in range(10000):
    x = seidel(gtg, x, gty)
# Ao fim da execução do Seidel temos calculado todos os valores aproximados de x
arquivo = open('saida1.txt','w')

# Aplica-se o polinomio obtido aos dados de teste para tentar prever o overall com base nos atributos
erroAbsoluto = []
erroRelativo = []
for i in range(treino, len(atributos)):
    predict = x[0] + int(atributos[i][1])*x[1] + atributos[i][2]*x[2] + atributos[i][3]*x[3] + atributos[i][4]*x[4] + atributos[i][5]*x[5] + atributos[i][6]*x[6]
    erroAbsoluto.append(abs(predict - overall[i]))
    erroRelativo.append(abs((predict - overall[i])/overall[i]))
    arquivo.write("True: " + str(overall[i])+ " Predict: [" + str(predict)+ "] Erro absoluto: " + str(erroAbsoluto[-1]) + " Erro relativo: " + str(erroRelativo[-1]) + "\n")
arquivo.write("\nErro absoluto médio: " + str(sum(erroAbsoluto)/len(erroAbsoluto)) + "\nErro relativo médio: " + str(sum(erroRelativo)/len(erroRelativo)))

arquivo.close()
