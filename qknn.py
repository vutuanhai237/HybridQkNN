# from swaptest import cswaptest
import numpy as np
from qiskit import aqua
from sklearn import datasets
import qiskit as qk
import random as rd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from knn import *
n_variables = 4  
n_train_points = 16 
n_test_points = 2  

# use iris dataset
iris = datasets.load_iris()
labels = iris.target
data_raw = iris.data
# encode data
data_raw = encode(data_raw[:, :n_variables])

randomIndices = rd.sample(range(0, 150), 64)


indicsTrain = randomIndices[:n_train_points]
indicsTrain.sort()
indicsTest = randomIndices[n_train_points:n_train_points + n_test_points]
indicsTest.sort()
# now pick these indices from the data
train_datas = np.asarray([data_raw[i] for i in indicsTrain])
train_labels =  np.asarray([labels[i] for i in indicsTrain])

test_datas = np.asarray([data_raw[i] for i in indicsTest])
test_labels =  np.asarray([labels[i] for i in indicsTest])

print(train_datas)
print(data_raw[:20])
