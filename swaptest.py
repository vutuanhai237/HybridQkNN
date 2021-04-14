import numpy as np
from encoding import Encoding
from qiskit import Aer, ClassicalRegister, execute, visualization, QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.aqua import QuantumInstance

def encoded_circuit(input_vector):
    encode = Encoding(input_vector, 'dc_amplitude_encoding')
    output = ClassicalRegister(len(encode.output_qubits))
    return encode.qcircuit

def get_i_v(N, padding = 0):
    """Get paramaters for log2(N) Fredkin gates

    Args:
        N (int): dimensional of states
        padding (int, optional): Defaults to 0.

    Returns:
        list of integer: params for the second and third Frekin gates
    """
    indices = []
    for i in range(0, int(np.log2(N))):
        indices.append(2**i + padding)
    return indices

def integrated_swap_test_circuit(vector1, vector2):
    """Return fidelity between two same - dimension vectors by cswaptest

    Args:
        vector1 (numpy array): First vector, don't need to normalize
        vector2 (numpy array): Second vector, don't need to normalize

    Returns:
        float: Fidelity = sqrt((p(0) - p(1)) / n_shot)
    """
    if len(vector1) != len(vector2): 
        return 0
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector1)
    N = len(vector1)

    total_qubits = (N-1)*2+1
    circuit = QuantumCircuit(total_qubits, 1)
    circuit.h(0)
    circuit.compose(encoded_circuit(vector1), qubits=[*range(1, N)], inplace=True)
    circuit.compose(encoded_circuit(vector2), qubits=[*range(N, 2*N - 1)], inplace=True)

    cs1 = get_i_v(N)
    cs2 = get_i_v(N, N - 1)
    for i in range(len(cs1)):
        circuit.cswap(0, cs1[i], cs2[i])
    circuit.h(0)
    circuit.measure(0,0)

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
        fidelity = integrated_swap_test_circuit(vector1, vector2)
        fidelities = np.append(fidelities, fidelity)
    return np.average(fidelities)

