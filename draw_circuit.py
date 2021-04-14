import numpy as np
from encoding import Encoding
from qiskit import Aer, ClassicalRegister, execute, visualization

input_vector = 2*np.random.rand(2)-1

input_vector = input_vector / np.linalg.norm(input_vector)
expected_probabilities = input_vector ** 2
print((input_vector))

encode = Encoding(input_vector, 'dc_amplitude_encoding')
output = ClassicalRegister(len(encode.output_qubits))
encode.qcircuit.add_register(output)
encode.qcircuit.barrier()

for k, value in enumerate(reversed(encode.output_qubits)):
    encode.qcircuit.measure(encode.quantum_data[value], output[k])

visualization.circuit_drawer(encode.qcircuit, filename="images/encode2", output='mpl', style={'backgroundcolor': '#FFFFFF'})