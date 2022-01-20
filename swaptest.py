import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from qiskit.visualization import *

# Importing qiskit libraries
from math import pi,sin,cos,asin,acos,sqrt,ceil
from rssi_sim import rssi, compute_rssi_d, compute_rssi
from swap_test_circuit import compute_similarity, qubit_by_qubit_swaptest, run_circuit
import os

# Reference Points in map

fp_database = {
    "sun_deck_1" : [350, 50],
    "living_room" : [400, 400],
    "fdr_bedroom" : [100, 400],
    "er_bedroom" : [100, 650],
    "entrance_hall" : [350, 650],
    "kitchen" : [650, 650],
    "sec_bedroom" : [600, 400],

}

# Access Points in map

ap_database = {
    "ap0" : [20, 700],
    "ap1" : [700, 700],
    "ap2" : [50, 200],         # TO-DO : Swap test for arbitrary dimensions
    "ap3" : [700, 200],         # TO-DO : Swap test for arbitrary dimensions
}

# Location names of Reference Points

fp_l_database = {
    "sun_deck_1" : "Sun Deck",
    "living_room" : "Living Room",
    "fdr_bedroom" : "FDR Bedroom",
    "er_bedroom" : "ER Bedroom",
    "entrance_hall" : "Entrance Hall",
    "kitchen" : "Kitchen",
    "sec_bedroom" : "Sec. Bedroom",

}

# Simulated RSSI values based on Prakabar et al

fp_rssi_database = compute_rssi_d(fp_database, ap_database)

def get_fp_rssi_probability_database(fp_rssi_d, rssi_vector):

    fp_rssi_probability_d = dict()

    for key, value in fp_rssi_d.items():
        fp_rssi_probability_d[key] = compute_similarity(value, rssi_test_vector)

    key_max = max(zip( fp_rssi_probability_d.values(),  fp_rssi_probability_d.keys()))[1]

    return fp_rssi_probability_d, key_max

if __name__ == "__main__":
    st.markdown("# Indoor Localization ")
    st.markdown("### Quantum Fingerprint Matching Protocol")

    # map resource location
    im = plt.imread('resources/wh-floorplan.PNG')
    plt.imshow(im)

    ## Fingerprint Database
    x_coords = []
    y_coords = []
    for key, value in fp_database.items():
        x_coords.append(value[0])
        y_coords.append(value[1])
    # plot the reference points on map    
    plt.scatter(x=x_coords, y=y_coords, c='r', s=40)

    ## Access Points Database
    ap_x_coords = []
    ap_y_coords = []
    for key, value in ap_database.items():
        ap_x_coords.append(value[0])
        ap_y_coords.append(value[1])
    # plot the access points on map    
    plt.scatter(x=ap_x_coords, y=ap_y_coords, c='g', s=40)

    # Controls
    rssi_test_vector_pass = None 
    prob_loc_pass = None
    with st.sidebar:
        st.markdown("## Move the Test Vector Location")
        x_p = st.slider(label="X-axis", min_value = 0, value=400, max_value = 700)
        y_p = st.slider(label="Y-axis", min_value = 0, value=394, max_value = 700)

        # plot the location of point
        plt.scatter(x=[x_p], y=[y_p], c='b', s=40)

        # compute rssi of test vector
        rssi_test_vector = compute_rssi([x_p, y_p], ap_database)

        # compute probability database
        fp_rssi_probability_database, prob_loc = get_fp_rssi_probability_database(fp_rssi_database, rssi_test_vector)
        
        st.markdown(f"""
            RSS Test Vector: \n
            {rssi_test_vector}
            """)
        st.markdown(f"""**Probable Location**: \n 
                    {fp_l_database[prob_loc]}
                    """)

        # st.dataframe(pd.DataFrame.from_dict(fp_rssi_database, orient='index', columns = ['RSSI_A', 'RSSI_B', 'RSSI_C', 'RSSI_D']))
        prob_loc_pass, rssi_test_vector_pass = prob_loc, rssi_test_vector

    # Show image
    plt.savefig('x.png')

    st.image('x.png')
    os.remove('x.png')
    
    st.markdown(""" 
    In this map, you may navigate as a blue dot to check the Received Signal Strength (RSS) of your receiver from four Access Points (APs) (green dots).\n 
    Based on RSS vector of your receiver and the RSS fingerprint vectors recorded in Reference Points (RPs) marked as red dots, we can approximate the indoor location of blue dot using
    swap test or quantum fingerprint matching protocol.\n
    To estimate the location, the web app applies swap test to 7 fingerprint locations (red dots) to determine the similarity score of each RPs to the current position, denoted
    by RSS vector. The fingerprint location with the highest score is selected as the estimated user location.\n
    """)


    with st.expander("Quantum Circuit Implementing the Fingerprint Matching"):
        shots = 10000
        v1_state = np.array(rssi_test_vector)
        v2_state = np.array(fp_rssi_database[prob_loc_pass])

        v1_state_norm = v1_state / np.linalg.norm(v1_state)
        v2_state_norm = v2_state / np.linalg.norm(v2_state)

        qc, num_qubits = qubit_by_qubit_swaptest(v1_state_norm, v2_state_norm)
        qc_plot = qc.draw(output='mpl')
        st.pyplot(fig=qc_plot)

        counts = run_circuit(qc, shots = shots)

        hist_plot = plot_histogram(counts.get_counts(0))

        st.markdown(
            """
            Measurement
            """
        )

        st.pyplot(fig = hist_plot)
        
    with st.expander("More Information and About the Simulation"):
        st.markdown("""
            When developing the simulation system, several assumptions are made:
            
            1. The map is obstacle-free. Meaning the walls do not affect the Received Signal Strength from different APs.
            2. The RSS values are based on distance estimation model presented by [Prabakar et al., 2015](https://www.youtube.com/watch?v=CWvRJdF7oVE).
            3. To perform fingerprint matching, the quantum algorithm proposed by [Shokry et al., 2020](https://arxiv.org/pdf/2106.11751.pdf) is used. The algorithm consists of three parts:
            
                - State Preparation (Amplitude encoding of $n$-dimensional vectors to $log_2(n)$ qubits
            
                - Input Processing using Swap Test
            
                - Computing Similarity Scores using Measurement Gates. 
            
            4. To estimate the similarity score, the swap test is applied 10000 times.
            
            The web app renders the circuit diagram for computing the similarity score between the current RSS vector of the receiver (Test Vector) and the Fingerprint Vector identified as match
            by the quantum algorithm. Feedbacks are very welcome, especially how the computation of RSS values can be improved/simulated and how the effect of the walls can be added to the system.
            
        """)




