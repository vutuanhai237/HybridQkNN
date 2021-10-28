import numpy as np
import base.encoding, base.constant, qiskit


def encoded_circuit(input_vector):
    encode = base.encoding.Encoding(input_vector, 'dc_amplitude_encoding')
    return encode.qcircuit

def get_index_fredkin_gate(N, padding = 0):
    """Get paramaters for log2(N) Fredkin gates

    Args:
        - N (int): dimensional of states
        - padding (int, optional): Defaults to 0.

    Returns:
        - list of int: params for the second and third Frekin gates
    """
    indices = []
    for i in range(0, int(np.log2(N))):
        indices.append(2**i + padding)
    return indices

def create_integrated_swap_test_circuit(vector1, vector2):
    """[summary]

    Args:
        - vector1 (numpy array): First vector
        - vector2 (numpy array): Second vector

    Returns:
        QuantumCircuit: full circuit
    """
    N = len(vector1)
    cs1 = get_index_fredkin_gate(N)
    cs2 = get_index_fredkin_gate(N, N - 1)
    total_qubits = (N-1)*2+1
    # Construct circuit
    qc = qiskit.QuantumCircuit(total_qubits, 1)
    qc.h(0)
    qc.compose(encoded_circuit(vector1), qubits=[*range(1, N)], inplace=True)
    qc.compose(encoded_circuit(vector2), qubits=[*range(N, 2*N - 1)], inplace=True)
    for i in range(len(cs1)):
        qc.cswap(0, cs1[i], cs2[i])
    qc.h(0)
    qc.measure(0,0)
    return qc

def integrated_swap_test_circuit(vector1, vector2):
    """Return fidelity between two same - dimension vectors by cswaptest

    Args:
        - vector1 (numpy array): First vector, don't need to normalize
        - vector2 (numpy array): Second vector, don't need to normalize

    Returns:
        - float: Fidelity = sqrt((p(0) - p(1)) / n_shot)
    """
    if len(vector1) != len(vector2): 
        raise Exception('Two states must have the same dimensional')
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector1)
    qc = create_integrated_swap_test_circuit(vector1, vector2)
    counts = qiskit.execute(qc, backend = qiskit.Aer.get_backend('qasm_simulator'), shots = base.constant.num_shots).result().get_counts()
    return np.sqrt(np.abs((counts.get("0", 0) - counts.get("1", 0)) / base.constant.num_shots))


def get_fidelity(vector1, vector2, iteration: int = 1):
    """Run ist circuit many times

    Args:
        - vector1 (numpy array): First vector, don't need to normalize
        - vector2 (numpy array): Second vector, don't need to normalize
        - iteration (int): Number of iteration

    Returns:
        - Float: mean fidelity
    """
    fidelities = np.array([])
    for _ in range(0, iteration):
        fidelity = integrated_swap_test_circuit(vector1, vector2)
        fidelities = np.append(fidelities, fidelity)
    return np.average(fidelities)

