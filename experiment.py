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

scaler = StandardScaler()

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

def exec_mfcm_filter(dataset, nRep, nClusters):
    ## Inicializando variáveis
    result = {}
    Jmin = 2147483647
    centers = 0

    data = np.vstack((dataset[0], dataset[1]))
    ref = np.hstack((dataset[2], dataset[3]))

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

def exec_knn(data_train, data_test, target_train, target_test, n_neighbors):

    start = time.time()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    end = time.time()
    
    # print(classification_report(target_test, target_pred))
    # print(f'F1 Score: {f1_score(target_test, target_pred, average="macro")}')
    return target_pred, (end - start)

def kFold(r_data, r_target, seed, n_neighbors):

    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_fold = 0
    r_data_train, r_data_test, r_target_train, r_target_test = 0, 0, 0, 0

    for train, test in sk.split(r_data, r_target):
        x_train, x_test = r_data[train], r_data[test]
        y_train, y_test = r_target[train], r_target[test]

        pred, exec_time = exec_knn(x_train, x_test, y_train, y_test, n_neighbors)
        score = f1_score(y_test, pred, average='macro')
        if score > best_fold:
            r_data_train, r_data_test, r_target_train, r_target_test = x_train, x_test, y_train, y_test
            best_fold = score

    print(f'train set size: {len(r_data_train)}')
    print(f'test set size: {len(r_data_test)}')

    return (r_data_train, r_data_test, r_target_train, r_target_test)

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	arquivo.write(lista)
	arquivo.write('\n')
	arquivo.close()

def filtro_mutual_info(dataset, numVar):
    print('MUTUAL INFO TESTE')
    X = np.vstack((dataset[0], dataset[1]))
    y = np.hstack((dataset[2], dataset[3]))

    resultado_filtro = mutual_info_classif(X, y)

    resultado_filtro = [(pontuacao, indice) for indice, pontuacao in enumerate(resultado_filtro)]
    resultado_filtro.sort(key=lambda x: x[0])

    resultado_filtro = (resultado_filtro, 'Filtro por Mutual Info')

    X = apply_filter(X, resultado_filtro, numVar)

    return (X, y)

def experimento(indexData, n_neighbors, nFilterRep):

    seed = 42

    dataset = selectDataset(indexData)
    
    print(dataset)

    numTotalVar = dataset[0].shape[1]

    # K fold dataset original
    r_data, r_target, nClasses, data_name = dataset
    r_data_train, r_data_test, r_target_train, r_target_test = kFold(r_data, r_target, seed, n_neighbors)

    # KNN sem filtro
    print(f'Executando KNN com dataset original: {data_name}')
    pred, exec_time = exec_knn(r_data_train, r_data_test, r_target_train, r_target_test, n_neighbors)
    score = f1_score(r_target_test, pred, average='macro')
    print(f'Sem filtro - F1 Score: {score}')
    metrics_info1 = (f'Sem filtro - F1 Score: {score} | Tempo: {exec_time}')

    # Execução do filtro
    dataset_to_filter = (r_data_train, r_data_test, r_target_train, r_target_test)
    result_mfcm = exec_mfcm_filter(dataset_to_filter, nFilterRep, nClasses)

    porcentagemVar = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # porcentagemVar = [25, 50]

    for i in porcentagemVar:
        numVar = int(numTotalVar * (i/100))
        print(f'Porcentagem de variáveis: {i}%')
        print(f'Número de variáveis: {numVar}')
        print(f'Executando filtro com {numVar} variáveis')
        metrics_info0 = (f'Seed: {seed} | Dataset: {indexData} | K: {n_neighbors} | VarCortadas: {numVar} | NumReps Filtro: {nFilterRep}')

        # Dataset filtrado com MFCM
        filtered_dataset = run_filter(dataset_to_filter, result_mfcm, numVar, nClasses)
    
        # K fold dataset filtrado MFCM
        f_data, f_target = filtered_dataset
        f_data_train, f_data_test, f_target_train, f_target_test = kFold(f_data, f_target, seed, n_neighbors)

        # KNN com filtro MFCM
        print(f'Executando KNN com dataset filtrado MFCM: {data_name}')
        pred, exec_time = exec_knn(f_data_train, f_data_test, f_target_train, f_target_test, n_neighbors)
        score = f1_score(f_target_test, pred, average='macro')
        print(f'Com filtro MFCM - F1 Score: {score}')
        metrics_info2 = (f'Com filtro MFCM - F1 Score: {score} | Tempo: {exec_time}')

        # Dataset filtrado mutual info
        filtered_dataset = filtro_mutual_info(dataset_to_filter, numVar)
        
        # K fold dataset filtrado MFCM
        f_data, f_target = filtered_dataset
        f_data_train, f_data_test, f_target_train, f_target_test = kFold(f_data, f_target, seed, n_neighbors)

        # KNN com filtro mutual info
        print(f'Executando KNN com dataset filtrado mutual info: {data_name}')
        pred, exec_time = exec_knn(f_data_train, f_data_test, f_target_train, f_target_test, n_neighbors)
        score = f1_score(f_target_test, pred, average='macro')
        print(f'Com filtro mutual info - F1 Score: {score}')
        metrics_info3 = (f'Com filtro mutual info - F1 Score: {score} | Tempo: {exec_time}')

        # Escrevendo no arquivo
        atualizaTxt('logs/resultados.txt', metrics_info0)
        atualizaTxt('logs/resultados.txt', metrics_info1)
        atualizaTxt('logs/resultados.txt', metrics_info2)
        atualizaTxt('logs/resultados.txt', metrics_info3)
        atualizaTxt('logs/resultados.txt', '')

if __name__ == "__main__":

    num_datasets = 8
    n_neighbors = 5
    nRepMFCM = 1

    for id_dataset in range(1, num_datasets + 1):

        print(id_dataset)

        dataset_name = selectDataset(id_dataset)[-1]
        info_data = (f'############################## {dataset_name} ##############################\n')

        atualizaTxt('logs/resultados.txt', info_data)

        experimento(id_dataset, n_neighbors, nRepMFCM)
