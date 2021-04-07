# from swaptest import cswaptest
import numpy as np
from qiskit import aqua
from sklearn import datasets
import qiskit as qk
import random as rd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from knn import encode, predict
import itertools 




n_variables = 4  
n_train_points = 16 
n_test_points = 2  
k = 3
# use iris dataset
iris = datasets.load_iris()
labels = iris.target
data_raw = iris.data
# encode data
data_raw = encode(data_raw[:, :n_variables])

randomIndices0 = rd.sample(range(0, 50), 6)
randomIndices1 = rd.sample(range(55, 100), 5)
randomIndices2 = rd.sample(range(105, 150), 5)

indicsTrain = list(itertools.chain(randomIndices0, randomIndices1, randomIndices2))

print(indicsTrain)
n_test = n_test_points
indicsTest = []
while n_test != 0:
    random = (rd.sample(range(0, 150), 1))[0]
    if random in indicsTrain:
        indicsTest.append(random)
        n_test = n_test - 1
print(indicsTest)
# now pick these indices from the data
train_datas = np.asarray([data_raw[i] for i in indicsTrain])
train_labels =  np.asarray([labels[i] for i in indicsTrain])

test_datas = np.asarray([data_raw[i] for i in indicsTest])
test_labels =  np.asarray([labels[i] for i in indicsTest])

print(train_datas)
print(train_labels)
print(test_datas)
print(predict(train_datas, train_labels, test_datas, k))
print(test_labels)