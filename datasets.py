import numpy as np
import pandas as pd
from scipy.io import arff

def selectDataset(id):
	if id == 1:
		# Scene | OPENML: ID 312 | 300 features
		data = arff.loadarff('datasets/scene.arff')
		
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)
		dataset = dataset.drop(dataset.columns[294:299], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset = normalize(dataset.to_numpy())
		nClusters = 2
		
		print("Dataset selecionado: Scene\n")
		
		return [dataset, dataset_ref, nClusters, "Scene Dataset"]
	elif id == 2:
		# Madelon | OPENML: ID 1485 | 500 features
		data = arff.loadarff('datasets/madelon.arff')
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset = normalize(dataset.to_numpy())
		nClusters = 2
		
		print("Dataset selecionado: Madelon\n")
		
		return [dataset, dataset_ref, nClusters, "Madelon Dataset"]
	elif id == 3:
		# Hiva Agnostic | OPENML: ID 1039 | 1000 features
		data = arff.loadarff('datasets/hiva_agnostic.arff')
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset = normalize(dataset.to_numpy())
		nClusters = 2
		
		print("Dataset selecionado: Hiva Agnostic\n")
		
		return [dataset, dataset_ref, nClusters, "Hiva Agnostic Dataset"]
	elif id == 4:
		# Musk (Version 1) | UCI Machine Learning Repository | 165 features
		dataset = pd.read_csv('musk1.data', header=None)
		dataset = dataset.drop(dataset.columns[[0, 1]], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset = normalize(dataset.to_numpy())
		nClusters = 2
		
		print("Dataset selecionado: Musk (Version 1)\n")
		
		return [dataset, dataset_ref, nClusters, "Musk (Version 1) Dataset"]
	
def normalize(dataset):
	nRows = dataset.shape[0]
	nCol = dataset.shape[1]

	normalizedArray = np.arange(nRows * nCol)
	normalizedArray = normalizedArray.reshape((nRows, nCol))
	normalizedArray = (np.zeros_like(normalizedArray)).astype('float64')

	media = np.mean(dataset, axis=0)
	dp = np.std(dataset, axis=0)

	novo = np.arange(nRows * nCol)
	novo = normalizedArray.reshape((nRows, nCol))
	novo = (np.zeros_like(novo)).astype('float64')

	for i in range (0, nRows):
		for j in range (0, nCol):
			novo[i, j] = (dataset[i, j] - media[j]) / dp[j]

	return novo