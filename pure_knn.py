# from swaptest import cswaptest
import numpy as np
from qiskit import aqua
from sklearn import datasets, neighbors
import qiskit as qk
import random as rd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from knn import encode, predict, bench_mark
import itertools 

# hyperparameter
n_variables = 4 
n_train_points = 16
n_test_points = int(n_train_points*0.3)
k = 7
iteration = 10

# use iris dataset
iris = datasets.load_iris()
labels = iris.target
data_raw = iris.data

# get training indices
randomIndices0 = rd.sample(range(0, 50), int(n_train_points/3))
randomIndices1 = rd.sample(range(55, 100), int(n_train_points/3))
randomIndices2 = rd.sample(range(105, 150), n_train_points-int(n_train_points/3)*2)
indicsTrain = list(itertools.chain(randomIndices0, randomIndices1, randomIndices2))

# get test indices
n_test = n_test_points
indicsTest = []
while n_test != 0:
    random = (rd.sample(range(0, 150), 1))[0]
    if random not in indicsTest and random not in indicsTrain:
        indicsTest.append(random)
        n_test = n_test - 1

# now pick these data with given indices
train_datas = np.asarray([data_raw[i] for i in indicsTrain])
train_labels =  np.asarray([labels[i] for i in indicsTrain])
test_datas = np.asarray([data_raw[i] for i in indicsTest])
test_labels =  np.asarray([labels[i] for i in indicsTest])

# predict
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(train_datas, train_labels)
y_pred = clf.predict(test_datas)
accuracy, precision, recall, matrix = bench_mark(test_labels, y_pred)
print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('matrix: ', matrix)
