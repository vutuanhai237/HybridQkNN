from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from qiskit import aqua
from sklearn import datasets
import qiskit as qk
import numpy as np
import random as rd
import itertools
# initialising the quantum instance
backend = qk.BasicAer.get_backend('qasm_simulator')
instance = aqua.QuantumInstance(backend, shots=10000)

# initialising the qknn model
qknn = QKNeighborsClassifier(
    n_neighbors=3,
    quantum_instance=instance
)

n_variables = 4       # should be positive power of 2
n_train_points = 128    # can be any positive integer
n_test_points = 38     # c an be any positive integer

# use iris dataset
iris = datasets.load_iris()
labels = iris.target
data_raw = iris.data

# encode data
encoded_data = analog.encode(data_raw[:, :n_variables])

# now pick these indices from the data
randomIndices0 = rd.sample(range(0, 50), int(n_train_points/3))
randomIndices1 = rd.sample(range(55, 100), int(n_train_points/3))
randomIndices2 = rd.sample(range(105, 150), n_train_points-int(n_train_points/3)*2)
indicsTrain = list(itertools.chain(randomIndices0, randomIndices1, randomIndices2))

# get test indices
n_test = n_test_points
indicsTest = []
while n_test != 0:
    random = (rd.sample(range(0, 150), 1))[0]
    if random not in indicsTest:
        indicsTest.append(random)
        n_test = n_test - 1

# pick these state and its labels with given indices
train_datas = np.asarray([encoded_data[i] for i in indicsTrain])
train_labels =  np.asarray([labels[i] for i in indicsTrain])
test_datas = np.asarray([encoded_data[i] for i in indicsTest])
test_labels =  np.asarray([labels[i] for i in indicsTest])

qknn.fit(train_datas, train_labels)
qknn_prediction = qknn.predict(test_datas)

print(qknn_prediction)
print(test_labels)