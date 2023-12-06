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

    resultado_filtro = variance_filter(data, result['bestM'], numClasses)
    resultado_filtro[0].sort(key=lambda k : k[0])

	## Aplicando filtro
    data = apply_filter(data, resultado_filtro, numVar)
    target = np.hstack((dataset[2], dataset[3]))

    return (data, target)

def select_dataset(indexData):
    if indexData == 1:
        nClasses = 3
        dataset_name = 'Iris'
        data, target = load_iris(return_X_y=True)
        data = scaler.fit_transform(data)
        return (data, target, nClasses, dataset_name)
    if indexData == 2:
        nClasses = 10
        dataset_name = 'Digits'
        data, target = load_digits(return_X_y=True)
        data = scaler.fit_transform(data)
        return (data, target, nClasses, dataset_name)
    if indexData == 3:
        nClasses = 3
        dataset_name = 'Wine'
        data, target = load_wine(return_X_y=True)
        data = scaler.fit_transform(data)
        # data = preprocessing.normalize(data)
        return (data, target, nClasses, dataset_name)
    if indexData == 4:
        nClasses = 2
        dataset_name = 'Breast Cancer'
        data, target = load_breast_cancer(return_X_y=True)
        data = scaler.fit_transform(data)
        # data = preprocessing.normalize(data)
        return (data, target, nClasses, dataset_name)

def exec_knn(data_train, data_test, target_train, target_test, n_neighbors):

    start = time.time()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    end = time.time()
    
    # print(classification_report(target_test, target_pred))
    # print(f'F1 Score: {f1_score(target_test, target_pred, average="macro")}')
    return target_pred, (end - start)

def kFold(r_data, r_target, seed):

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

if __name__ == "__main__":

    # Parâmetros
    seed = 2
    indexData = 4
    n_neighbors = 15
    # numVar = 6                      # Número de variáveis a serem cortadas
    nFilterRep = 10 

    dataset = select_dataset(indexData)

    numTotalVar = dataset[0].shape[1]

    # K fold dataset original
    r_data, r_target, nClasses, data_name = dataset
    r_data_train, r_data_test, r_target_train, r_target_test = kFold(r_data, r_target, seed)

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

    for i in porcentagemVar:
        numVar = int(numTotalVar * (i/100))
        print(f'Porcentagem de variáveis: {i}%')
        print(f'Número de variáveis: {numVar}')
        print(f'Executando filtro com {numVar} variáveis')
        metrics_info0 = (f'Seed: {seed} | Dataset: {indexData} | K: {n_neighbors} | VarCortadas: {numVar} | NumReps Filtro: {nFilterRep}')

        filtered_dataset = run_filter(dataset_to_filter, result_mfcm, numVar, nClasses)
    
        # K fold dataset filtrado
        f_data, f_target = filtered_dataset
        f_data_train, f_data_test, f_target_train, f_target_test = kFold(f_data, f_target, seed)

        # KNN com filtro
        print(f'Executando KNN com dataset filtrado: {data_name}')
        pred, exec_time = exec_knn(f_data_train, f_data_test, f_target_train, f_target_test, n_neighbors)
        score = f1_score(f_target_test, pred, average='macro')
        print(f'Com filtro - F1 Score: {score}')
        metrics_info2 = (f'Com filtro - F1 Score: {score} | Tempo: {exec_time}')

        # Escrevendo no arquivo

        atualizaTxt('logs/resultados.txt', metrics_info0)
        atualizaTxt('logs/resultados.txt', metrics_info1)
        atualizaTxt('logs/resultados.txt', metrics_info2)
        atualizaTxt('logs/resultados.txt', '')

    atualizaTxt('logs/resultados.txt', '##########################################################\n')