import numpy as np
from encoding import Encoding
from qiskit import Aer, ClassicalRegister, execute, visualization, QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.aqua import QuantumInstance
def encoded_circuit(input_vector):
    encode = Encoding(input_vector, 'dc_amplitude_encoding')
    output = ClassicalRegister(len(encode.output_qubits))
    return encode.qcircuit

def get_index_out(len_vector, padding = 0):
    indexs = []
    for i in range(0, int(np.log2(len_vector))):
        indexs.append(2**i + padding)
    return indexs
def scalar_product(vector1, vector2):
    product = 0
    for i in range(len(vector1)):
        product += vector1[i]*vector2[i]
    return product
def cswaptest(vector1, vector2):
    """Return fidelity between two same - dimension vectors by cswaptest

    Args:
        vector1 (numpy array): First vector, don't need to normalize
        vector2 (numpy array): Second vector, don't need to normalize

    Returns:
        float: Fidelity = sqrt((p(0) - p(1)) / shots) with shots = 1024
    """
    if len(vector1) != len(vector2): 
        return 0
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector1)
    n = len(vector1)

    total_qubits = (n-1)*2+1
    circuit = QuantumCircuit(total_qubits, 1)
    circuit.h(0)

    split_point = n
    circuit.compose(encoded_circuit(vector1), qubits=[*range(1, split_point)], inplace=True)
    circuit.compose(encoded_circuit(vector2), qubits=[*range(split_point, split_point + (n - 1))], inplace=True)

    cs1 = get_index_out(n)
    cs2 = get_index_out(n,n-1)
    for i in range(len(cs1)):
        circuit.cswap(0, cs1[i], cs2[i])
    circuit.h(0)
    circuit.measure(0,0)
    
    # visualization.circuit_drawer(circuit, filename="images/encode_swaptest/encode16", output='mpl', style={'backgroundcolor': '#FFFFFF'})

    shots = 16384
    qasm_sim = Aer.get_backend('qasm_simulator')
    qobj = assemble(circuit, qasm_sim, shots=shots)
    results = qasm_sim.run(qobj).result()
    answer = results.get_counts()
    if answer.get("1") is not None:
        return np.sqrt(np.abs((answer["0"] - answer["1"]) / shots))
    else:
        return 1

def fidelity(vector1, vector2, iteration):
    fidelities = np.array([])
    for i in range(0, iteration):
        fidelity = cswaptest(vector1, vector2)
        fidelities = np.append(fidelities, fidelity)
    return np.average(fidelities)

# testing

# fidelities = np.array([])
# expected_results = np.array([])
# deltas = np.array([])

# vector1 = np.asarray([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])
# vector2 = np.asarray([3,2,1,3,4,2,4,2,3,2,1,3,4,2,4,2])
# vector1 = vector1 / np.linalg.norm(vector1)
# vector2 = vector2 / np.linalg.norm(vector2)
# cswaptest(vector1, vector2)
# for i in range(0, 10):

#     fidelity = cswaptest(vector1, vector2)
#     expected_result = np.dot(vector1, vector2)
#     fidelities = np.append(fidelities, fidelity)
#     expected_results = np.append(expected_results, expected_result)
#     deltas = np.append(deltas, np.abs(fidelity - expected_result))

# average = np.average(fidelities)
# average_delta = np.average(deltas)
# print("Expected: ", expected_results[0])
# print("Result: ", average)
# print("Delta: ",average_delta)
# print("Percent: " + str(average_delta/average*100) + "%" )