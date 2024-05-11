import numpy as np 
from math import pi,sin,cos,asin,acos,sqrt,ceil, log10, log2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

def qubit_by_qubit_swaptest(v1_state_norm, v2_state_norm):
    """
    Constructs the qubit by qubit swap test circuit following the algorithm
    https://handwiki.org/wiki/Physics:Swap_test

    The circuit is divided into three phases:
    1. State preparation for the given normalized vectors
    2. Swap Test
    3. Similarity Measure
    Args:
        v1_state_norm   :   given normalized vector (e.g. Test Vector)
        v2_state_norm   :   given normalized vector (e.g. FP Vector)
    Returns:
        qc              :   Quantum Circuit object ready to be measured
        num_qubits      :   Number of qubits to represent the given vector 

    The dimension of RSSI space is restricted to the powers of 2.
    """
    # 1. State Preparation
    # Based on section `Arbitrary Initialization` : 
    # Commit : https://github.com/Qiskit/qiskit-tutorials/blob/7e48d1e79e20a2679fb8d6417310c8d2a110a8fd/tutorials/circuits/3_summary_of_quantum_operations.ipynb
    vector_dim = len(v1_state_norm) # determine the dimension
    # len(v1_state_norm) == len(v2_state_norm)
    num_qubits = ceil(log2(vector_dim)) # Note: Vector Dimension should be powers of 2
    # Declare the number of quantum and classical registers to be used
    q = QuantumRegister(2 * num_qubits + 1, name = "q")
    c = ClassicalRegister(1, name = "c")
    # Initialize arbitrary states
    q1_l = list()
    q2_l = list()

    for i in range(0, num_qubits):
        q1_l.append(i)

    for i in range(num_qubits, 2 * num_qubits):
        q2_l.append(i)

    qc = QuantumCircuit(q, c)
    qc.initialize(v1_state_norm, q1_l)
    qc.initialize(v2_state_norm, q2_l)
    
    # End of 1. State Preparation

    # 2. Swap Test
    # Algorithm : https://handwiki.org/wiki/Physics:Swap_test
    # Apply Hadamard gate to the ancilla qubit 
    qc.h( q[2 * num_qubits] ) 

    # apply cswap over each pair of qubits in the two registers
    for i in range(0, num_qubits):
        qc.cswap(2 * num_qubits, i, i+num_qubits)

    # Apply Hadamard gate to the ancilla qubit
    qc.h( q[2 * num_qubits] ) 

    # End of 2. Swap Test

    # 3. Measure the ancilla qubit in the Z-basis and record the result
    # We assume the results are either 0 or 1
    qc.measure(q[2*num_qubits], c[0])

    return qc, num_qubits


def compute_similarity(fp_vector, test_vector):
    shots = 10000
    v1_state = np.array(test_vector)
    v2_state = np.array(fp_vector)

    v1_state_norm = v1_state / np.linalg.norm(v1_state)
    v2_state_norm = v2_state / np.linalg.norm(v2_state)

    qc, num_qubits = qubit_by_qubit_swaptest(v1_state_norm, v2_state_norm)

    counts = run_circuit(qc, shots = shots)

    print(counts.get_counts(0))
    return counts.get_counts(0)['0'] / shots 


def run_circuit(qc, backend='qasm_simulator', shots=10000):
    """
    Run the given circuit
    Args:
        qc          :       Quantum Circuit to simulate

        backend     :       default='qasm_simulator'
                            The backend to use. List of available backends can be accessed using Aer.backends()
        shots       :       default=10000
                            Number of runs

    Returns:
        counts      :       Dictionary of counts, can be used for histogram
    """
    # Transpile for simulator
    sim = Aer.get_backend(backend)
    qc = transpile(qc, sim)
    # run and get counts
    jobs = sim.run(qc, shots=shots)
    counts = jobs.result()

    return counts 
