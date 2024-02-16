import numpy as np
import random
import time
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

from sklearn.feature_selection import mutual_info_classif, chi2

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
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

def filtro_mutual_info(X, y, numVar):

    resultado_filtro = mutual_info_classif(X, y)

    resultado_filtro = [(pontuacao, indice) for indice, pontuacao in enumerate(resultado_filtro)]
    resultado_filtro.sort(key=lambda x: x[0])

    resultado_filtro = (resultado_filtro, 'Filtro por Mutual Info')

    X = apply_filter(X, resultado_filtro, numVar)

    return X

def media_listas(lista):
    medias = []

    for i in range(len(lista[0])):
        soma_scores = 0
        soma_tempos = 0
        for lista_interna in lista:
            score, tempo = lista_interna[i]
            soma_scores += score
            soma_tempos += tempo
        media_score = soma_scores / len(lista)
        media_tempo = soma_tempos / len(lista)
        medias.append((media_score, media_tempo))

    return medias

def cross_validation(data, target, seed, n_neighbors, n_folds, nFilterRep, nClasses, porcentagemVar, filter_name):

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    resultados = []

    for train, test in kfold.split(data, target):

        print('Split')

        if filter_name == 'MFCM':
            mfcm = exec_mfcm_filter(data[train], nFilterRep, nClasses)

        scores_porcentagem = []

        for i in porcentagemVar:
            numVar = int(data.shape[1] * (i/100))
            # print(f'Porcentagem de variáveis cortadas: {i}%')
            # print(f'Número de variáveis apos filtro: {data.shape[1] - numVar}')

            if filter_name == 'MFCM':
                filtered_train = filter(data[train], mfcm, numVar, nClasses)
                filtered_test = filter(data[test], mfcm, numVar, nClasses)
            elif filter_name == 'MUTUAL':
                filtered_train = filtro_mutual_info(data[train], target[train], numVar)
                filtered_test = filtro_mutual_info(data[test], target[test], numVar)

            score, time = exec_knn(filtered_train, filtered_test, target[train], target[test], n_neighbors)

            scores_porcentagem.append((score, time))

        resultados.append(scores_porcentagem)

    resultados = media_listas(resultados)

    print(resultados)

    return resultados

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
    result_mfcm = cross_validation(data, target, seed, n_neighbors, 5, nFilterRep, nClasses, porcentagemVar, 'MFCM')
    result_mutual = cross_validation(data, target, seed, n_neighbors, 5, nFilterRep, nClasses, porcentagemVar, 'MUTUAL')

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

if __name__ == "__main__":

    datasets = [1]
    n_neighbors = 5
    nRepMFCM = 10

    # experimento(3, n_neighbors, nRepMFCM)

    for _, id in enumerate(datasets):
        
        experimento(id, n_neighbors, nRepMFCM)
