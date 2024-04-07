import numpy as np
import random
import time
import math
from MFCM import MFCM
from filters import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from datasets import selectDataset

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import pickle
import os
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

random.seed(42)

scaler = StandardScaler()

def execute(nRep, dataset, centersAll):
    Jmin = 2147483647	# int_max do R
    bestL = 0			# melhor valor de L_resp
    bestM = 0			# melhor valor de M_resp
    best_centers = 0

    for r in range(nRep):
        # print(f'MFCM rep: {r}')
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

    score = f1_score(target_test, target_pred, average="macro")
    
    # print(classification_report(target_test, target_pred))
    print(f'F1 Score: {score}')
    return score, (end - start)

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	arquivo.write(lista)
	arquivo.write('\n')
	arquivo.close()

def save_list(list, dataset_name, i_externo, i_interno):
    file_name = f'matrices/{dataset_name}/lista_{i_externo}{i_interno}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(list, f)

def load_list(dataset_name, i_externo, i_interno):
    file_name = f'matrices/{dataset_name}/lista_{i_externo}{i_interno}.pkl'
    with open(file_name, 'rb') as f:
        mfcm = pickle.load(f)
    return mfcm

def filtro_mutual_info(X, y, numVar):

    resultado_filtro = mutual_info_regression(X, y)

    resultado_filtro = [(pontuacao, indice) for indice, pontuacao in enumerate(resultado_filtro)]
    resultado_filtro.sort(key=lambda x: x[0])

    resultado_filtro = (resultado_filtro, 'Filtro por Mutual Info')

    X = apply_filter(X, resultado_filtro, numVar)

    return X

def media_desvio_padrao(lista):
    medias = []
    desvios_padrao = []

    for i in range(len(lista[0])):
        soma_scores = 0
        soma_tempos = 0
        scores_quadrados = 0
        tempos_quadrados = 0
        for lista_interna in lista:
            score, tempo = lista_interna[i]
            soma_scores += score
            soma_tempos += tempo
            scores_quadrados += score ** 2
            tempos_quadrados += tempo ** 2

        n = len(lista)
        media_score = soma_scores / n
        media_tempo = soma_tempos / n
        variancia_score = (scores_quadrados / n) - (media_score ** 2)
        variancia_tempo = (tempos_quadrados / n) - (media_tempo ** 2)
        desvio_padrao_score = math.sqrt(variancia_score)
        desvio_padrao_tempo = math.sqrt(variancia_tempo)
        
        medias.append((media_score, media_tempo))
        desvios_padrao.append((desvio_padrao_score, desvio_padrao_tempo))

    return medias, desvios_padrao

def cross_validation(data, target, seed, n_neighbors, n_folds, nFilterRep, nClasses, porcentagemVar, filter_name, data_name, i_externo):

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    numTotalFolds = 5

    resultados = []
    best_result = -1
    best_data = []
    best_mfcm = []

    inicio = time.time()

    for i_interno, (train, test) in enumerate(kfold.split(data, target)):

        print('Split')

        if filter_name == 'MFCM':
            if not os.path.exists(f'matrices/{data_name}'):
                os.makedirs(f'matrices/{data_name}')
            if os.path.exists(f'matrices/{data_name}/lista_{i_externo}{i_interno}.pkl'):
                mfcm = load_list(data_name, i_externo, i_interno)
            else:
                mfcm = exec_mfcm_filter(data[train], nFilterRep, nClasses)
                save_list(mfcm, data_name, i_externo, i_interno)

        # print(f'var: {data[test].shape[1]}')
        numVar = (data[test].shape[1] // 2)

        if filter_name == 'MFCM':
            filtered_train = filter(data[train], mfcm, numVar, nClasses)
            filtered_test = filter(data[test], mfcm, numVar, nClasses)
        elif filter_name == 'MUTUAL':
            filtered_train = filtro_mutual_info(data[train], target[train], numVar)
            filtered_test = filtro_mutual_info(data[test], target[test], numVar)

        score, tempo = exec_knn(filtered_train, filtered_test, target[train], target[test], n_neighbors)

        if score > best_result:
            best_result = score
            if filter_name == 'MFCM':
                best_mfcm = mfcm
            best_data = (data[train], data[test], target[train], target[test])

    fim = time.time()

    # print(f'TEMPO DE EXECUÇÃO DA FOLD: {fim - inicio} segundos')

    scores_porcentagem = []
    data_train, data_test, target_train, target_test = best_data

    for i in porcentagemVar:
        numVar = int(data.shape[1] * (i/100))
        # print(f'Porcentagem de variáveis cortadas: {i}%')
        # print(f'Número de variáveis apos filtro: {data.shape[1] - numVar}')

        if filter_name == 'MFCM':
            filtered_train = filter(data_train, best_mfcm, numVar, nClasses)
            filtered_test = filter(data_test, best_mfcm, numVar, nClasses)
        elif filter_name == 'MUTUAL':
            filtered_train = filtro_mutual_info(data_train, target_train, numVar)
            filtered_test = filtro_mutual_info(data_test, target_test, numVar)

        score, tempo = exec_knn(filtered_train, filtered_test, target_train, target_test, n_neighbors)

        scores_porcentagem.append((score, tempo))

    print(scores_porcentagem)

    return scores_porcentagem

def experimento(indexData, n_neighbors, nFilterRep):

    seed = 42

    dataset = selectDataset(indexData)

    porcentagemVar = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # porcentagemVar = [0, 25, 50, 75]

    # K fold dataset original
    data, target, nClasses, data_name = dataset

    info_data = (f'############################## {data_name} ##############################\n')
    atualizaTxt('logs/resultados.txt', info_data)

    # Executando filtros e classificadores: 

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    lista_resultados = []

    for i_externo, (train, test) in enumerate(kfold.split(data, target)):
        result_mfcm = cross_validation(data[train], target[train], seed, n_neighbors, 5, nFilterRep, nClasses, porcentagemVar, 'MFCM', data_name, i_externo)
        lista_resultados.append(result_mfcm)
        result_mutual = cross_validation(data[train], target[train], seed, n_neighbors, 5, nFilterRep, nClasses, porcentagemVar, 'MUTUAL', data_name, i_externo)

    # Armazenando resultados:
    basics_info = (f'Seed: {seed} | Dataset: {data_name} | K: {n_neighbors} | MFCM Reps: {nFilterRep}\n')
    atualizaTxt('logs/resultados.txt', basics_info)

    for _, i in enumerate(result_mfcm):
        var_info = (f'Porcentagem de variaveis cortadas: {porcentagemVar[_]}%')
        metrics_mfcm = (f'Com filtro MFCM - F1 Score: {i[0]} | Tempo: {i[1]}')
        metrics_mutual = (f'Com filtro Mutual - F1 Score: {result_mutual[_][0]} | Tempo: {result_mutual[_][1]}')

        atualizaTxt('logs/resultados.txt', var_info)
        atualizaTxt('logs/resultados.txt', metrics_mfcm)
        atualizaTxt('logs/resultados.txt', metrics_mutual)
        atualizaTxt('logs/resultados.txt', '')

    print(f'RESULTADO FINAL MEDIA: {media_desvio_padrao(lista_resultados)[0]}')
    print(f'RESULTADO FINAL DESVIO: {media_desvio_padrao(lista_resultados)[1]}')

if __name__ == "__main__":

    datasets = [3]
    n_neighbors = 5
    nRepMFCM = 10

    # experimento(3, n_neighbors, nRepMFCM)

    for _, id in enumerate(datasets):
        
        experimento(id, n_neighbors, nRepMFCM)
        