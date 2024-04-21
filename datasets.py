import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

scaler = StandardScaler()

def selectDataset(id):
	if id == 1:
		dataset_name = 'Iris'
		nClasses = 3
		data, target = load_iris(return_X_y=True)		# Iris | OPENML: ID 61 | 4 features | 150 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Iris\n")

		return (data, target, nClasses, dataset_name)
	elif id == 2:
		dataset_name = 'Digits'
		nClasses = 10
		data, target = load_digits(return_X_y=True)		# Digits | OPENML: ID 28 | 64 features | 1797 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Digits\n")

		# Digits é um dataset de imagem

		return (data, target, nClasses, dataset_name)
	elif id == 3:
		dataset_name = 'Wine'
		nClasses = 3
		data, target = load_wine(return_X_y=True)		# Wine | OPENML: ID 187 | 13 features | 178 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Wine\n")

		return (data, target, nClasses, dataset_name)
	elif id == 4:
		dataset_name = 'Breast Cancer'
		nClasses = 2
		data, target = load_breast_cancer(return_X_y=True)	# Breast cancer | UCI: ID 17 | 30 features | 569 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Breast Cancer\n")

		return (data, target, nClasses, dataset_name)
	elif id == 5:
		dataset_name = 'Scene'
		data = arff.loadarff('datasets/scene.arff')		# Scene | OPENML: ID 312 | 294 features	| 2407 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)
		dataset = dataset.drop(dataset.columns[294:299], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2

		# Scene é um dataset de imagem
		
		print("Dataset selecionado: Scene\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 6:
		dataset_name = 'Madelon'
		data = arff.loadarff('datasets/madelon.arff')		# Madelon | OPENML: ID 1485 | 500 features | 4400 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2
		
		print("Dataset selecionado: Madelon\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 7:
		dataset_name = 'Hiva Agnostic'
		data = arff.loadarff('datasets/hiva_agnostic.arff')		# Hiva Agnostic | OPENML: ID 1039 | 1000 features | 4228 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2
		
		print("Dataset selecionado: Hiva Agnostic\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 8:
		dataset_name = 'Musk Version 1'
		dataset = pd.read_csv('datasets/musk1.data', header=None)		# Musk (Version 1) | UCI Machine Learning Repository | 165 features
		dataset = dataset.drop(dataset.columns[[0, 1]], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2
		
		print("Dataset selecionado: Musk (Version 1)\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	