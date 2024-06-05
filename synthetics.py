import pandas as pd
import numpy as np
import random
import time
import math
from MFCM import MFCM
from filters import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from datasets import selectDataset

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import pickle
import os

random.seed(42) # Verificar se interfere na geração resultado dos sintéticos

def execute(nRep, dataset, centersAll):
    Jmin = 2147483647	# int_max do R
    bestL = 0			# melhor valor de L_resp
    bestM = 0			# melhor valor de M_resp
    best_centers = 0

    for r in range(nRep):
        print(f'MFCM rep: {r}')
        centers = list(map(int, centersAll[r,].tolist()))

        resp = MFCM(dataset, centers, 2)

        J = resp[0]
        L_resp = resp[1]
        M_resp = resp[2]

        if (Jmin > J):
            Jmin = J
            bestL = L_resp
            bestM = M_resp
            best_centers = centers

    dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'best_centers': centersAll}
    # Retorna os centers de todas as iterações para o KMeans (mudar para criar uma nova lista exclusiva para o KMeans)

    return dict

def exec_mfcm_filter(data, nRep, nClusters):
    ## Inicializando variáveis
    result = {}
    Jmin = 2147483647
    centers = 0

    nObj = len(data)

    centersMC = np.zeros((nRep, nClusters))

    for c in range(nRep):
        centersMC[c] = random.sample(range(1, nObj), nClusters)

    clustering = execute(nRep, data, centersMC)

    if clustering['Jmin'] < Jmin:
        Jmin = clustering['Jmin']
        result = clustering
    centers = clustering['best_centers']

    return result

def run_filter(dataset, result, numVar, numClasses):
	
    data = np.vstack((dataset[0], dataset[1]))
    target = np.hstack((dataset[2], dataset[3]))

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

    data = apply_filter(data, resultado_filtro, numVar)

    return (data, target)

def filter(data, result, numVar, numClasses):

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

    data = apply_filter(data, resultado_filtro, numVar)

    return data

def exec_knn(data_train, data_test, target_train, target_test, n_neighbors):

    start = time.time()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    end = time.time()

    f1 = f1_score(target_test, target_pred, average="macro")
    accuracy = accuracy_score(target_test, target_pred)
    precision = precision_score(target_test, target_pred, average="macro")
    recall = recall_score(target_test, target_pred, average="macro")
    
    # print(classification_report(target_test, target_pred))
    # print(f'F1 Score: {score}')
    return f1, accuracy, precision, recall, (end - start)

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	arquivo.write(lista)
	arquivo.write('\n')
	arquivo.close()

def filtro_mutual_info(X, y, numVar):

    resultado_filtro = mutual_info_regression(X, y)

    resultado_filtro = [(pontuacao, indice) for indice, pontuacao in enumerate(resultado_filtro)]
    resultado_filtro.sort(key=lambda x: x[0])

    resultado_filtro = (resultado_filtro, 'Filtro por Mutual Info')

    X = apply_filter(X, resultado_filtro, numVar)

    return X

def media_desvio_padrao(lista):
    f1_avg = []
    accuracy_avg = []
    precision_avg = []
    recall_avg = []
    time_avg = []

    f1_std = []
    accuracy_std = []
    precision_std = []
    recall_std = []
    time_std = []

    for percentage in range(len(lista[0])):
        f1_values = []
        accuracy_values = []
        precision_values = []
        recall_values = []
        time_values = []

        for fold in lista:
            f1_values.append(fold[percentage][0])
            accuracy_values.append(fold[percentage][1])
            precision_values.append(fold[percentage][2])
            recall_values.append(fold[percentage][3])
            time_values.append(fold[percentage][4])

        f1_avg.append(np.mean(f1_values))
        accuracy_avg.append(np.mean(accuracy_values))
        precision_avg.append(np.mean(precision_values))
        recall_avg.append(np.mean(recall_values))
        time_avg.append(np.mean(time_values))
        
        f1_std.append(np.std(f1_values))
        accuracy_std.append(np.std(accuracy_values))
        precision_std.append(np.std(precision_values))
        recall_std.append(np.std(recall_values))
        time_std.append(np.std(time_values))

    print("Médias:")
    print("F1-Score:", f1_avg)
    print("Acurácia:", accuracy_avg)
    print("Precisão:", precision_avg)
    print("Recall:", recall_avg)

    print("\nDesvios Padrão:")
    print("F1-Score:", f1_std)
    print("Acurácia:", accuracy_std)
    print("Precisão:", precision_std)
    print("Recall:", recall_std)

    return f1_avg, accuracy_avg, precision_avg, recall_avg, time_avg, f1_std, accuracy_std, precision_std, recall_std, time_std

def cross_validation(data, target, seed, n_neighbors, nFilterRep, nClasses, porcentagemVar, filter_name):

    # inicio = time.time()

        # print('Fold Interno')

        # # print(f'var: {data[test].shape[1]}')
        # numVar = (data[test].shape[1] // 2)

        # if filter_name == 'MFCM':

        #     mfcm = exec_mfcm_filter(data, nFilterRep, nClasses)

        #     filtered_train = filter(data[train], mfcm, numVar, nClasses)
        #     filtered_test = filter(data[test], mfcm, numVar, nClasses)
        # elif filter_name == 'MUTUAL':
        #     filtered_train = filtro_mutual_info(data[train], target[train], numVar)
        #     filtered_test = filtro_mutual_info(data[test], target[test], numVar)

        # f1, accuracy, precision, recall, tempo = exec_knn(filtered_train, filtered_test, target[train], target[test], n_neighbors)

        # if f1 > best_result:
        #     best_result = f1
        #     if filter_name == 'MFCM':
        #         best_mfcm = mfcm
        #     best_data = (data[train], data[test], target[train], target[test])

    # fim = time.time()

    # print(f'TEMPO DE EXECUÇÃO DA FOLD: {fim - inicio} segundos')

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    scores_porcentagem = []

    for i in porcentagemVar:
        numVar = int(data.shape[1] * (i/100))

        if filter_name == 'MFCM':
            mfcm = exec_mfcm_filter(data, nFilterRep, nClasses)
            filtered_train = filter(X_train, mfcm, numVar, nClasses)
            filtered_test = filter(X_test, mfcm, numVar, nClasses)
        elif filter_name == 'MUTUAL':
            filtered_train = filtro_mutual_info(X_train, y_train, numVar)
            filtered_test = filtro_mutual_info(X_test, y_test, numVar)

        f1, accuracy, precision, recall, tempo = exec_knn(filtered_train, filtered_test, y_train, y_test, n_neighbors)

        scores_porcentagem.append((f1, accuracy, precision, recall, tempo))

    # print(scores_porcentagem)

    return scores_porcentagem

def synthetic(indexData, n_neighbors, nFilterRep, mc):

    seed = 42
    # porcentagemVar = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    porcentagemVar = [0, 25, 50, 75]

    f1_avg_list, acc_avg_list, prec_avg_list, rec_avg_list, time_avg_list, f1_std_list, acc_std_list, prec_std_list, rec_std_list, time_std_list = [], [], [], [], [], [], [], [], [], []

    for m in range(mc):
        print(f'Monte carlo: {m}')
        dataset = selectDataset(indexData)

        data, target, nClasses, data_name = dataset

        lista_resultados_mfcm = []  

        result_mfcm = cross_validation(data, target, seed, n_neighbors, nFilterRep, nClasses, porcentagemVar, 'MFCM')
        result_mutual = cross_validation(data, target, seed, n_neighbors, nFilterRep, nClasses, porcentagemVar, 'MUTUAL')
        lista_resultados_mfcm.append(result_mfcm)

    f1_avg, accuracy_avg, precision_avg, recall_avg, time_avg, f1_std, accuracy_std, precision_std, recall_std, time_std = media_desvio_padrao(lista_resultados_mfcm)
    f1_avg_list.append(f1_avg)
    acc_avg_list.append(accuracy_avg)
    prec_avg_list.append(precision_avg)
    rec_avg_list.append(recall_avg)
    time_avg_list.append(time_avg)
    f1_std_list.append(f1_std)
    acc_std_list.append(accuracy_std)
    prec_std_list.append(precision_std)
    rec_std_list.append(recall_std)
    time_std_list.append(time_std)

    # print(lista_resultados_mfcm)

    # print(f'TAMANHO LISTA: {len(f1_avg_list)}')

    # print(f'Média F1 Score: {sum(f1_avg_list) / len(f1_avg_list)}')
    # print(f'Média Acurácia: {sum(acc_avg_list) / len(acc_avg_list)}')
    # print(f'Média Precisão: {sum(prec_avg_list) / len(prec_avg_list)}')
    # print(f'Média Recall: {sum(rec_avg_list) / len(rec_avg_list)}')
    # print(f'Média Tempo: {sum(time_avg_list) / len(time_avg_list)}')

    # data = {
    #     'F1-Score (Avg)': f1_avg,
    #     'Acurácia (Avg)': accuracy_avg,
    #     'Precisão (Avg)': precision_avg,
    #     'Recall (Avg)': recall_avg,
    #     'Tempo (Avg)': time_avg,
    #     'F1-Score (Std)': f1_std,
    #     'Acurácia (Std)': accuracy_std,
    #     'Precisão (Std)': precision_std,
    #     'Recall (Std)': recall_std,
    #     'Tempo (Std)': time_std
    # }

    # criando txt de parametros
    # path = f'resultados/{data_name}'
    # if not os.path.exists(f'resultados/{data_name}'):
    #     os.makedirs(path)
    # basics_info = (f'Seed: {seed} | Dataset: {data_name} | K: {n_neighbors} | MFCM Reps: {nFilterRep} | Neighbors: {n_neighbors}')
    # variables_cut_info = f'Porcentagens de variaveis cortadas: {porcentagemVar}'
    # os.makedirs(f'resultados/{data_name}/resultado_{len(os.listdir(path)) + 1}')
    # atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/parameters.txt', basics_info)
    # atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/parameters.txt', variables_cut_info)

    # # criando txt de resultados
    # atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/resultados.txt', basics_info)
    # for _, i in enumerate(result_mfcm):
    #     var_info = (f'Porcentagem de variaveis cortadas: {porcentagemVar[_]}%')
    #     metrics_mfcm = (f'Com filtro MFCM - F1 Score: {i[0]} | Tempo: {i[4]}')
    #     metrics_mutual = (f'Com filtro Mutual - F1 Score: {result_mutual[_][0]} | Tempo: {result_mutual[_][4]}')

    #     atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/resultados.txt', var_info)
    #     atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/resultados.txt', metrics_mfcm)
    #     atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/resultados.txt', metrics_mutual)
    #     atualizaTxt(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/resultados.txt', '')

    # # criando dataframe de resultados
    # df = pd.DataFrame(data)
    # df = df.transpose()
    # df.columns = ['Porcentagem ' + str(i*10) for i in range(len(df.columns))]
    # df.to_csv(f'resultados/{data_name}/resultado_{len(os.listdir(path))}/mfcm_stats.csv', index=False)
    # print(df)

if __name__ == "__main__":

    datasets = [1]
    n_neighbors = 5
    monteCarlo = 1
    nRepMFCM = 10

    synthetic(16, n_neighbors, nRepMFCM, monteCarlo)

    # for _, id in enumerate(datasets):
        
    #     experimento(id, n_neighbors, nRepMFCM)
        