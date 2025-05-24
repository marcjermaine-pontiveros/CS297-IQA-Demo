import numpy as np
from math import pi, sin, cos, asin, acos, sqrt, ceil, log10, log2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer


def qubit_by_qubit_swaptest(v1_state_norm, v2_state_norm):
    """
    Constructs the qubit-by-qubit SWAP test circuit.
    This circuit is used to estimate the fidelity (similarity) between two quantum states,
    which are encoded from the input vectors v1_state_norm and v2_state_norm.
    The algorithm is based on https://handwiki.org/wiki/Physics:Swap_test.

    The circuit involves three main stages:
    1. State Preparation: Initialize qubits to represent v1_state_norm and v2_state_norm.
    2. SWAP Test Core: Apply Hadamard and controlled-SWAP gates.
    3. Measurement: Measure an ancillary qubit to determine similarity.

    Args:
        v1_state_norm (np.array): The first normalized vector, pre-padded to a dimension that is a power of 2.
        v2_state_norm (np.array): The second normalized vector, pre-padded to a dimension that is a power of 2.
                                  Must have the same dimension as v1_state_norm.
    Returns:
        qc (QuantumCircuit): The constructed Qiskit QuantumCircuit object.
        num_qubits (int): The number of qubits used to encode a single input vector.
    """
    # 1. State Preparation
    # Determine the dimension of the input vectors. This dimension must be a power of 2
    # as these vectors are expected to be pre-padded by the `compute_similarity` function.
    vector_dim = len(v1_state_norm)

    if vector_dim == 0:
        # This case should ideally be prevented by compute_similarity before calling this function.
        raise ValueError("Input vector dimension is 0. Cannot create a quantum state.")
    if (vector_dim & (vector_dim - 1)) != 0 or vector_dim == 0:
        # Also a safeguard, as compute_similarity should handle this.
        raise ValueError(f"Input vector dimension {vector_dim} is not a power of 2.")

    # Number of qubits required to encode one vector via amplitude encoding.
    num_qubits = int(log2(vector_dim))

    # Declare quantum registers:
    # - num_qubits for the first vector (q1_l)
    # - num_qubits for the second vector (q2_l)
    # - 1 ancillary qubit for the SWAP test measurement
    q = QuantumRegister(2 * num_qubits + 1, name="q")
    # Classical register to store the measurement result of the ancillary qubit
    c = ClassicalRegister(1, name="c")

    qc = QuantumCircuit(q, c)

    # Define qubit lists for initialization
    q1_l = list(range(num_qubits))
    q2_l = list(range(num_qubits, 2 * num_qubits))

    # Initialize the quantum states using amplitude encoding
    qc.initialize(v1_state_norm, q1_l)
    qc.initialize(v2_state_norm, q2_l)

    # End of 1. State Preparation

    # 2. SWAP Test Core
    ancilla_qubit_idx = 2 * num_qubits  # Index of the ancillary qubit

    # Apply Hadamard gate to the ancillary qubit
    qc.h(q[ancilla_qubit_idx])

    # Apply controlled-SWAP (Fredkin) gates.
    # The ancillary qubit is the control.
    # The SWAP operations are performed between corresponding qubits of the two state registers.
    for i in range(num_qubits):
        qc.cswap(ancilla_qubit_idx, q1_l[i], q2_l[i])

    # Apply Hadamard gate to the ancillary qubit again
    qc.h(q[ancilla_qubit_idx])
    # End of 2. SWAP Test Core

    # 3. Measurement
    # Measure the ancillary qubit in the Z-basis. The outcome probability P(0) relates to similarity.
    qc.measure(q[ancilla_qubit_idx], c[0])

    return qc, num_qubits


def compute_similarity(fp_vector, test_vector):
    """
    Computes the similarity score between two input vectors using the SWAP test.
    The similarity score is the probability of measuring '0' on the ancillary qubit of the SWAP test circuit.

    Steps:
    1. Pre-process vectors: Convert to float, check for zero dimension.
    2. Pad vectors: If vector dimension is not a power of 2, pad with zeros to the next power of 2.
       - `vector_dim`: Original dimension of input vectors.
       - `num_qubits_for_vector`: Minimum qubits needed for original dimension (ceil(log2(vector_dim))).
       - `required_dim`: The dimension vectors must have to be encoded (2^num_qubits_for_vector).
       - Padding ensures vectors can be mapped to quantum states using amplitude encoding.
    3. Normalize vectors: Normalize the (potentially padded) vectors to unit length.
       - Handles zero vectors (if norm is 0, similarity is 0).
    4. Construct SWAP test circuit using `qubit_by_qubit_swaptest`.
    5. Run the circuit on a simulator.
    6. Calculate similarity as P(0) = (counts of '0') / (total shots).

    Args:
        fp_vector (list or np.array): The fingerprint vector.
        test_vector (list or np.array): The test vector.

    Returns:
        float: The similarity score (P(0) from SWAP test), ranging from 0.0 to 1.0.
               A score of 0.0 can mean zero vectors or maximal dissimilarity.
    """
    shots = 10000  # Number of times to run the quantum circuit for statistics

    vector_dim = len(fp_vector)

    if vector_dim == 0:  # Handles empty input vectors
        return 0.0

    # Ensure input vectors are numpy arrays of type float for numerical operations
    v1_state = np.array(test_vector, dtype=float)
    v2_state = np.array(fp_vector, dtype=float)

    # Determine the required dimension for quantum state encoding (must be a power of 2)
    # `num_qubits_for_vector` is the number of qubits needed to represent a vector of `vector_dim` using amplitude encoding.
    num_qubits_for_vector = ceil(log2(vector_dim)) if vector_dim > 0 else 0
    # `required_dim` is 2 raised to the power of `num_qubits_for_vector`. This is the target dimension for padding.
    required_dim = 2**num_qubits_for_vector

    # Pad vectors with zeros if their original dimension is less than the required power of 2.
    # Padding ensures the vector can be mapped to a quantum state of `num_qubits_for_vector` qubits.
    if vector_dim < required_dim:
        # Pad with constant value 0.0 at the end of the vectors.
        v1_padded = np.pad(v1_state, (0, required_dim - vector_dim), "constant", constant_values=0.0)
        v2_padded = np.pad(v2_state, (0, required_dim - vector_dim), "constant", constant_values=0.0)
    else:  # If vector_dim is already a power of 2, no padding is needed.
        v1_padded = v1_state
        v2_padded = v2_state

    # Normalize the (padded) vectors to unit length for quantum state preparation.
    norm_v1 = np.linalg.norm(v1_padded)
    norm_v2 = np.linalg.norm(v2_padded)

    if norm_v1 == 0 or norm_v2 == 0:
        # If either vector (after padding) is all zeros, its norm will be 0.
        # A zero vector cannot be normalized to a valid quantum state.
        # In this context, if one or both vectors are zero, their similarity is treated as 0.
        return 0.0

    v1_state_norm = v1_padded / norm_v1
    v2_state_norm = v2_padded / norm_v2

    # Construct the SWAP test circuit with the prepared (padded and normalized) states.
    # Their dimension is now `required_dim`, which is a power of 2.
    qc, _ = qubit_by_qubit_swaptest(v1_state_norm, v2_state_norm)

    # Run the quantum circuit on a simulator.
    counts_result = run_circuit(qc, shots=shots)

    # Calculate the probability of measuring '0', which is the similarity score.
    # If '0' was never measured, .get('0', 0) returns 0.
    measured_prob_0 = counts_result.get_counts(0).get("0", 0) / shots
    return measured_prob_0


def run_circuit(qc, backend="qasm_simulator", shots=10000):
    """
    Runs the given Qiskit quantum circuit on a specified Aer backend.

    Args:
        qc (QuantumCircuit): The Qiskit QuantumCircuit to simulate.
        backend (str, optional): The Aer backend to use (e.g., 'qasm_simulator', 'statevector_simulator').
                                 Defaults to 'qasm_simulator'.
        shots (int, optional): Number of times the circuit is run to collect measurement statistics.
                               Defaults to 10000. Not used by 'statevector_simulator'.

    Returns:
        qiskit.result.Result: The result object containing the outcomes of the simulation.
    """
    # Get the specified Aer backend
    sim = Aer.get_backend(backend)
    # Transpile the circuit for the chosen backend (optimization step)
    qc_transpiled = transpile(qc, sim)
    # Run the circuit and get the results
    job = sim.run(qc_transpiled, shots=shots)
    result = job.result()

    return result
