import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Importing standard Qiskit libraries
from math import pi, sin, cos, asin, acos, sqrt, ceil
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
import random
import os


# Importing the helper functions
from swap_test_solver import *
from rssi_sim import rssi_test

backend = Aer.get_backend("statevector_simulator")
backend2 = Aer.get_backend("qasm_simulator")


def swap_test_info():
    st.markdown("# SWAP Test")
    st.write("""
        The SWAP test is a simple quantum circuit (as you will see in the further sections of this web app) which, given two states, allows to compute how much do they differ from each other. 
        In this web app we will try to cover the following sub-tasks:  

        1. Build a variational circuit which generates the most general 1 qubit state (any point in the Bloch sphere can be reached).  
        
        2. Using the circuit in step 1, and the SWAP test, find the best choice of parameters reproduce a randomly generated quantum state with 1 qubit.
        
        3. Generalize the SWAP test for a random $N$-qubit product state (each of the qubits are in the state $|0⟩$ or $|1⟩$). 

        Example of a product state:  
        
        $$
        |a⟩ = |01⟩
        $$
        
        Example of a non-product (entangled) state:
        
        $$
        |b⟩ = |00⟩ + |11⟩
        $$
        

        

        """)


def subtask1(fp_vector, x_p, y_p):
    st.markdown(
        """
        ## Location Determination Quantum Circuit
        
        """
    )
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector, random_statevector
    from qiskit.providers.aer import QasmSimulator
    from math import atan
    from numpy import linalg as la
    import numpy as np

    # v1_state = np.array([0.39, 0.92]) #shokry
    # v2_state = np.array([0.24, 0.97]) #shokry

    v1_state = np.array([x_p, y_p])
    v2_state = np.array(fp_vector)

    v1_state_norm = v1_state / np.linalg.norm(v1_state)
    v2_state_norm = v2_state / np.linalg.norm(v2_state)

    st.markdown(f"""
        Testing RSS Vector (Normalized): {v1_state_norm} \n
        Fingerprint RSS Vector (Normalized): {v2_state_norm} 
    """)

    qc = QuantumCircuit(3, 1)
    # qc = QuantumCircuit(3,1)
    # create_state(qc,1,pi/2,0)
    # create_state(qc,2,pi/2,pi/2)

    qc.initialize(v1_state_norm, [1])
    qc.initialize(v2_state_norm, [2])

    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)

    qc.measure(0, 0)

    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)

    job = simulator.run(compiled_circuit, shots=1981)
    result = job.result()
    counts = result.get_counts(compiled_circuit)

    hist_plot = plot_histogram(counts)

    qc_plot = qc.draw(output="mpl", style={"backgroundcolor": "#000000"})

    st.pyplot(fig=qc_plot)

    st.markdown(
        """
        Measurement
        """
    )

    st.pyplot(fig=hist_plot)


def get_fp_rssi(fp_d, ap_d):
    fp_rssi = dict()
    for key, value in fp_d.items():
        rssi_1, rssi_2 = (
            rssi_test(ap_d["ap1"][0], ap_d["ap1"][1], value[0], value[1]),
            rssi_test(ap_d["ap2"][0], ap_d["ap2"][1], value[0], value[1]),
        )
        fp_rssi[key] = [rssi_1, rssi_2]

    return fp_rssi


def get_fp_probability_database(fp_d, x_p, y_p):
    fp_probability_db = dict()
    for key, value in fp_d.items():
        fp_probability_db[key] = get_probability_swap([x_p, y_p], fp_d[key])

    key_max = max(zip(fp_probability_db.values(), fp_probability_db.keys()))[1]

    return fp_probability_db, key_max


def side():
    st.sidebar.markdown("Sidebar here")


if __name__ == "__main__":
    all_info = st.expander("Information")
    fp_to_pass = None
    x_p_pass = None
    y_p_pass = None
    # with all_info:
    #    swap_test_info()
    # side()

    fp_database = {
        "sun_deck_1": [400, 60],
        "living_room": [350, 400],
        "fdr_bedroom": [100, 450],
        "er_bedroom": [100, 650],
        "entrance_hall": [350, 650],
        "kitchen": [650, 650],
        "sec_bedroom": [600, 400],
    }

    fp_l_database = {
        "sun_deck_1": "Sun Deck",
        "living_room": "Living Room",
        "fdr_bedroom": "FDR Bedroom",
        "er_bedroom": "ER Bedroom",
        "entrance_hall": "Entrance Hall",
        "kitchen": "Kitchen",
        "sec_bedroom": "Sec. Bedroom",
    }

    ap_database = {"ap0": [50, 50], "ap1": [700, 700], "ap2": [400, 200]}

    fp_rssi = get_fp_rssi(fp_database, ap_database)

    st.markdown("# Indoor Localization")

    im = plt.imread("resources/wh-floorplan.png")
    plt.imshow(im)
    ## Fingerprint Database
    x_coords = []
    y_coords = []
    for key, value in fp_database.items():
        x_coords.append(value[0])
        y_coords.append(value[1])

    plt.scatter(x=x_coords, y=y_coords, c="r", s=40)

    ## Access Points
    ap_x_coords = []
    ap_y_coords = []
    for key, value in ap_database.items():
        ap_x_coords.append(value[0])
        ap_y_coords.append(value[1])

    plt.scatter(x=ap_x_coords, y=ap_y_coords, c="g", s=40)

    with st.sidebar:
        st.markdown("## Move the Test Vector Location")
        x_p = st.slider(label="X-axis", min_value=0, value=400, max_value=700)
        y_p = st.slider(label="Y-axis", min_value=0, value=394, max_value=700)

        rssi_1 = rssi_test(ap_database["ap1"][0], ap_database["ap1"][1], x_p, y_p)
        rssi_2 = rssi_test(ap_database["ap2"][0], ap_database["ap2"][1], x_p, y_p)

        fp_prob_database, prob_loc = get_fp_probability_database(fp_rssi, rssi_1, rssi_2)

        st.markdown(f"## Probable Location: {fp_l_database[prob_loc]}")

        st.dataframe(pd.DataFrame.from_dict(fp_rssi, orient="index", columns=["RSSI_A", "RSSI_B"]))

        x_p_pass, y_p_pass, fp_to_pass = rssi_1, rssi_2, prob_loc

    plt.scatter(x=[x_p], y=[y_p], c="b", s=40)
    plt.savefig("x.png")

    st.image("x.png")
    os.remove("x.png")

    with st.expander("Quantum Circuit Implementing the Fingerprint Matching"):
        subtask1(fp_rssi[fp_to_pass], x_p_pass, y_p_pass)
