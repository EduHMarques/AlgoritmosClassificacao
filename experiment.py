import numpy as np
import random
from MFCM import MFCM
from filters import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

def execute(nRep, dataset, centersAll):
	Jmin = 2147483647	# int_max do R
	bestL = 0			# melhor valor de L_resp
	bestM = 0			# melhor valor de M_resp
	best_centers = 0

	for r in range(nRep):
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

    nObj = len(dataset)

    centersMC = np.zeros((nRep, nClusters))

    for c in range(nRep):
        centersMC[c] = random.sample(range(1, nObj), nClusters)

    clustering = execute(nRep, data, centersMC)

    if clustering['Jmin'] < Jmin:
        Jmin = clustering['Jmin']
        result = clustering
    centers = clustering['best_centers']

    return result

def run_filter(dataset, result, numVar, numClusters):
	
    data = np.vstack((dataset[0], dataset[1]))

    resultado_filtro = variance_filter(data, result['bestM'], numClusters)
    resultado_filtro[0].sort(key=lambda k : k[0])

	## Aplicando filtro
    data = apply_filter(data, resultado_filtro, numVar)
    target = np.hstack((dataset[2], dataset[3]))

    return (data, target)

def select_dataset(indexData, seed):
    if indexData == 1:
        data, target = load_iris(return_X_y=True)
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=seed)
        return (data_train, data_test, target_train, target_test)

def exec_knn(data_train, data_test, target_train, target_test, n_neighbors):

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    print(classification_report(target_test, target_pred))

if __name__ == "__main__":

    seed = 0
    n_neighbors = 3
    numVar = 1
    dataset = select_dataset(1, seed)

    print('Executando KNN com dataset original')
    r_data_train, r_data_test, r_target_train, r_target_test = dataset
    exec_knn(r_data_train, r_data_test, r_target_train, r_target_test, n_neighbors)

    result_mfcm = exec_mfcm_filter(dataset, 10, 3)
    filtered_dataset = run_filter(dataset, result_mfcm, numVar, 3)

    print('Executando KNN com dataset filtrado')
    f_data, f_target = filtered_dataset
    f_data_train, f_data_test, f_target_train, f_target_test = train_test_split(f_data, f_target, test_size=0.3, random_state=seed)
    exec_knn(f_data_train, f_data_test, f_target_train, f_target_test, n_neighbors)
