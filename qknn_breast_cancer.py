# from swaptest import cswaptest
import numpy as np
from qiskit import aqua
from sklearn import datasets
import qiskit as qk
import random as rd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from knn import encode, predict, bench_mark
import itertools 

n_variables = 4 
n_train_points = 128 
n_test_points = int(n_train_points*0.3)
k = 3
iteration = 10
# use iris dataset
breast_cancer = datasets.load_breast_cancer()
labels = breast_cancer.target
data_raw = breast_cancer.data
print((labels[:n_train_points]))
# # encode data
# data_raw = encode(data_raw[:, :n_variables])


# randomIndices0 = rd.sample(range(0, 50), int(n_train_points/3))
# randomIndices1 = rd.sample(range(55, 100), int(n_train_points/3))
# randomIndices2 = rd.sample(range(105, 150), n_train_points-int(n_train_points/3)*2)

# indicsTrain = list(itertools.chain(randomIndices0, randomIndices1, randomIndices2))

# print(indicsTrain)
# n_test = n_test_points
# indicsTest = []
# while n_test != 0:
#     random = (rd.sample(range(0, 150), 1))[0]
#     if random not in indicsTrain and random not in indicsTest:
#         indicsTest.append(random)
#         n_test = n_test - 1
# print(indicsTest)
# # now pick these indices from the data
# train_datas = np.asarray([data_raw[i] for i in indicsTrain])
# train_labels =  np.asarray([labels[i] for i in indicsTrain])

# test_datas = np.asarray([data_raw[i] for i in indicsTest])
# test_labels =  np.asarray([labels[i] for i in indicsTest])

# import matplotlib.pyplot as plt

# colours = {0:'orange', 1:'yellow', 2:'violet'}

# colours2 = {0:'red', 1:'green', 2:'blue'}


# # for i in range(len(train_datas)):
# #     plt.scatter(train_datas[i][0], train_datas[i][1], color = colours[train_labels[i]])
# # for i in range(len(test_datas)):
# #     plt.scatter(test_datas[i][0], test_datas[i][1], color = colours2[test_labels[i]])
# # plt.title('Iris')
# # plt.xlabel('petal length')
# # plt.ylabel('petal width')
# # plt.grid(True)
# # plt.show()

# print(train_datas)
# print(train_labels)
# print(test_datas)
# print('Test labels: ', test_labels)

# predict = np.asarray(predict(train_datas, train_labels, test_datas, k, iteration))
# print('Predict labels: ', predict)

# accuracy, precision, recall, matrix = bench_mark(test_labels, predict)
# print('accuracy: ', accuracy)
# print('precision: ', precision)
# print('recall: ', recall)
# print('matrix: ', matrix)
