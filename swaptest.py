import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import *

# Importing qiskit libraries
from math import pi, sin, cos, asin, acos, sqrt, ceil, log2
from rssi_sim import rssi, compute_rssi_d, compute_rssi  # For RSSI simulation
from swap_test_circuit import (
    compute_similarity,
    qubit_by_qubit_swaptest,
    run_circuit,
)  # For SWAP test and circuit execution
import os

# --- Static Databases ---
# These dictionaries define the fixed layout of Reference Points (FPs) and Access Points (APs) in the simulated environment.

# Reference Points (FPs) in the map: {fp_id: [x_coordinate, y_coordinate]}
fp_database = {
    "sun_deck_1": [350, 50],
    "living_room": [400, 400],
    "fdr_bedroom": [100, 400],
    "er_bedroom": [100, 650],
    "entrance_hall": [350, 650],
    "kitchen": [650, 650],
    "sec_bedroom": [600, 400],
}

# Access Points (APs) in the map: {ap_id: [x_coordinate, y_coordinate]}
# The number of APs determines the dimension of the RSSI vectors.
ap_database = {
    "ap0": [20, 700],
    "ap1": [700, 700],
    "ap2": [50, 200],
    "ap3": [700, 200],
}

# Location names for Reference Points: {fp_id: "User-friendly Name"}
# Used for displaying results in a readable format.
fp_l_database = {
    "sun_deck_1": "Sun Deck",
    "living_room": "Living Room",
    "fdr_bedroom": "FDR Bedroom",
    "er_bedroom": "ER Bedroom",
    "entrance_hall": "Entrance Hall",
    "kitchen": "Kitchen",
    "sec_bedroom": "Sec. Bedroom",
}

# --- Main Application Logic ---


def get_fp_rssi_probability_database(fp_rssi_d, current_rssi_test_vector):
    """
    Calculates the similarity probability for each fingerprint against the current test RSSI vector.

    Args:
        fp_rssi_d (dict): The database of RSSI vectors for all fingerprint locations.
                          (e.g., {'fp1': [rssi_ap0, rssi_ap1, ...], ...})
        current_rssi_test_vector (list): The RSSI vector of the current test location.

    Returns:
        tuple:
            - fp_rssi_probability_d (dict): Dictionary of similarity probabilities
                                            {fp_id: probability_score, ...}.
            - key_max (str): The fingerprint ID with the highest similarity score.
    """
    fp_rssi_probability_d = dict()
    for key, value in fp_rssi_d.items():
        # compute_similarity uses the SWAP test to get the P(0) value.
        fp_rssi_probability_d[key] = compute_similarity(value, current_rssi_test_vector)

    # Determine the fingerprint location with the maximum similarity score.
    key_max = max(zip(fp_rssi_probability_d.values(), fp_rssi_probability_d.keys()))[1]
    return fp_rssi_probability_d, key_max


if __name__ == "__main__":
    # --- Page Setup and Static Map Display ---
    st.set_page_config(layout="wide")  # Use wide layout for better space utilization.

    # Load and display the floor plan image.
    # The image is plotted once, and subsequent UI updates (sliders, etc.) will update points on this existing plot.
    im = plt.imread("resources/wh-floorplan.PNG")
    fig, ax = plt.subplots()  # Create a figure and an axes for Matplotlib plotting.
    ax.imshow(im)

    # Plot Fingerprint (Reference) Points (RPs) on the map.
    x_coords_fp = [value[0] for value in fp_database.values()]
    y_coords_fp = [value[1] for value in fp_database.values()]
    ax.scatter(x=x_coords_fp, y=y_coords_fp, c="r", s=40, label="Reference Points (RPs)")

    # Plot Access Points (APs) on the map.
    ap_x_coords = [value[0] for value in ap_database.values()]
    ap_y_coords = [value[1] for value in ap_database.values()]
    ax.scatter(x=ap_x_coords, y=ap_y_coords, c="g", s=40, label="Access Points (APs)")

    # Variables to hold data passed from sidebar to main area for display.
    rssi_test_vector_pass = None
    prob_loc_pass = None
    fp_rssi_database_computed = None

    # --- Sidebar Controls ---
    # Initialize user's location coordinates in Streamlit's session state if they don't already exist.
    # This allows the coordinates to persist across reruns triggered by widget interactions.
    if "x_p" not in st.session_state:
        st.session_state.x_p = 400  # Initial X coordinate for the blue dot (Test Location)
    if "y_p" not in st.session_state:
        st.session_state.y_p = 394  # Initial Y coordinate for the blue dot (Test Location)

    with st.sidebar:
        st.markdown("# Indoor Localization ")
        st.markdown("## Using Quantum Fingerprint Matching Protocol")

        st.markdown("---")  # Separator

        st.markdown("### User Location Controls")
        st.markdown(
            "Use the buttons below to move the blue dot (Test Location) on the map. Adjust the step size for finer or coarser movement."
        )

        # Slider to control the step size for button-based navigation.
        step_size = st.slider(
            label="Movement Step Size (pixels)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Number of pixels the blue dot moves per button click.",
        )

        # Navigation Buttons for moving the user's location (blue dot).
        # Arranged in three columns for a compact layout.
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:  # Middle column for Up/Down buttons
            if st.button("Up", use_container_width=True):
                st.session_state.y_p -= step_size  # Move up by decreasing y-coordinate
            if st.button("Down", use_container_width=True):
                st.session_state.y_p += step_size  # Move down by increasing y-coordinate

        with col1:  # Left column for Left button
            if st.button("Left", use_container_width=True):
                st.session_state.x_p -= step_size  # Move left by decreasing x-coordinate

        with col3:  # Right column for Right button
            if st.button("Right", use_container_width=True):
                st.session_state.x_p += step_size  # Move right by increasing x-coordinate

        # Boundary clamping: Ensure the user's location stays within the map limits (0-700 for both axes).
        st.session_state.x_p = max(0, min(st.session_state.x_p, 700))
        st.session_state.y_p = max(0, min(st.session_state.y_p, 700))

        # Display the current coordinates of the user's location.
        st.markdown(f"**Current Location (X, Y):** `{st.session_state.x_p}, {st.session_state.y_p}`")

        st.markdown("---")  # Separator

        st.markdown("### Simulation Settings")

        st.markdown("#### RSSI Model Parameters (Log-Distance Model)")
        # Sliders for configuring the Standard Log-Distance Path Model parameters.
        A_ref_val = st.slider(
            label="Reference RSSI (A_ref, dBm)",
            min_value=-60.0,
            max_value=-20.0,
            value=-40.0,
            step=1.0,
            help="RSSI value measured at the reference distance d0 (typically 1m).",
        )
        n_val_ple = st.slider(
            label="Path Loss Exponent (n)",
            min_value=1.5,
            max_value=6.0,
            value=3.0,
            step=0.1,
            help="Rate at which signal strength decreases with distance. Higher is faster attenuation.",
        )
        d0_val = st.slider(
            label="Reference Distance (d0, m)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Reference distance (meters) at which A_ref is measured.",
        )

        st.markdown("#### General RSSI Settings")
        # Slider for setting the standard deviation of Gaussian noise added to RSSI values.
        noise_std_dev = st.slider(
            label="RSSI Noise Std Dev (dBm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Standard deviation of Gaussian noise added to each RSSI reading. 0 means no noise.",
        )
        # Slider for setting the number of RSSI samples to average for each AP-location pair.
        num_samples_for_avg = st.slider(
            label="RSSI Samples for Averaging",
            min_value=1,
            max_value=20,
            value=1,
            help="Number of RSSI samples to generate and average. 1 means a single (potentially noisy) reading.",
        )

        st.markdown("---")  # Separator

        # Recompute the fingerprint RSSI database based on current simulation settings.
        # This ensures that reference fingerprints reflect the chosen model parameters, noise, and averaging.
        fp_rssi_database_computed = compute_rssi_d(
            fp_database,
            ap_database,
            noise_std_dev=noise_std_dev,
            num_samples_for_avg=num_samples_for_avg,
            A_ref=A_ref_val,
            n_val=n_val_ple,
            d0=d0_val,
        )

        # Plot the current test location (blue dot) on the map using coordinates from session state.
        blue_dot_scatter = ax.scatter(
            x=[st.session_state.x_p], y=[st.session_state.y_p], c="b", s=40, label="Test Location"
        )

        # Compute RSSI vector for the current test location using session state coordinates and simulation settings.
        rssi_test_vector = compute_rssi(
            [st.session_state.x_p, st.session_state.y_p],
            ap_database,
            noise_std_dev=noise_std_dev,
            num_samples_for_avg=num_samples_for_avg,
            A_ref=A_ref_val,
            n_val=n_val_ple,
            d0=d0_val,
        )

        # Compute similarity probabilities against all fingerprints and find the most probable location.
        fp_rssi_probability_database, prob_loc = get_fp_rssi_probability_database(
            fp_rssi_database_computed, rssi_test_vector
        )

        st.markdown("### Results Summary")
        # Display the simulated RSSI vector for the test location.
        st.markdown(f"""
            **Simulated RSS Test Vector (Blue Dot):** \n
            `{np.round(np.array(rssi_test_vector), 3)}`
            """)
        # Display the most probable location based on the highest similarity score.
        st.markdown(f"**Probable Location**: {fp_l_database[prob_loc]}")

        # Section: Detailed Similarity Scores
        # Displays a sorted table of all fingerprint locations and their similarity scores to the test vector.
        st.markdown("### Detailed Similarity Scores")
        if fp_rssi_probability_database:
            similarity_data = [
                {"Reference Point": fp_l_database[key], "Similarity Score": score}
                for key, score in fp_rssi_probability_database.items()
            ]
            similarity_df = pd.DataFrame(similarity_data)
            similarity_df_sorted = similarity_df.sort_values(by="Similarity Score", ascending=False)
            st.dataframe(similarity_df_sorted.style.format({"Similarity Score": "{:.4f}"}))
        else:
            st.markdown("Similarity scores are being computed...")

        # Pass necessary variables to the main area for further display (e.g., in expanders).
        prob_loc_pass, rssi_test_vector_pass = prob_loc, rssi_test_vector

    # --- Main Area Display ---
    # Display the updated map with all points.
    ax.legend()  # Show legend for points
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to free memory, important for Streamlit apps that redraw.

    # General explanation of the application.
    st.markdown(""" 
    Navigate the blue dot (Test Location) on the map using the **User Location Controls** in the sidebar. 
    The application calculates the Received Signal Strength (RSS) from four Access Points (APs, green dots) to this Test Location. 
    This RSS vector is then compared against a pre-computed database of RSS fingerprints from known Reference Points (RPs, red dots) using a quantum fingerprint matching protocol (SWAP test).
    The most similar RP is identified as the **Probable Location**.
    
    You can adjust RSSI simulation parameters in the sidebar, including the **Log-Distance Path Model** settings (`A_ref`, `n`, `d0`), **Gaussian noise**, and **signal averaging**. 
    These settings affect both the generation of fingerprint data and the calculation of the Test Location's RSSI vector.
    """)

    # Expander: Quantum Circuit and Vector Details
    # This section explains the quantum circuit used and shows the vector transformations.
    with st.expander("Quantum Circuit Implementing the Fingerprint Matching"):
        # Prepare data for the displayed circuit.
        # `display_v1_state` is the original (potentially noisy, averaged) test vector.
        # `display_v2_state` is the original (potentially noisy, averaged) fingerprint vector corresponding to the most probable location.
        if fp_rssi_database_computed and prob_loc_pass in fp_rssi_database_computed:
            display_v1_state = np.array(rssi_test_vector_pass, dtype=float)
            display_v2_state = np.array(fp_rssi_database_computed[prob_loc_pass], dtype=float)
        else:
            # Fallback if data isn't ready (should be rare).
            st.markdown(
                "Fingerprint data not yet computed or probable location is invalid. Displaying with potentially stale/default data."
            )
            display_v1_state = np.array(rssi_test_vector_pass if rssi_test_vector_pass is not None else [], dtype=float)
            default_fp_key = list(fp_database.keys())[0] if fp_database else None  # Pick first FP as a default
            if default_fp_key and fp_rssi_database_computed and default_fp_key in fp_rssi_database_computed:
                display_v2_state = np.array(fp_rssi_database_computed[default_fp_key], dtype=float)
            else:
                display_v2_state = np.array([], dtype=float)  # Absolute fallback

        # Perform padding and normalization for display purposes (mimicking `compute_similarity` steps).
        display_vector_dim = len(display_v1_state)
        num_encoding_qubits = 0
        ancilla_qubit_index = 0
        qc = None  # Initialize qc to None

        if display_vector_dim > 0:
            display_num_qubits_for_vector = ceil(log2(display_vector_dim)) if display_vector_dim > 0 else 0
            display_required_dim = 2**display_num_qubits_for_vector

            if display_vector_dim < display_required_dim:
                display_v1_padded = np.pad(
                    display_v1_state, (0, display_required_dim - display_vector_dim), "constant", constant_values=0.0
                )
                display_v2_padded = np.pad(
                    display_v2_state, (0, display_required_dim - display_vector_dim), "constant", constant_values=0.0
                )
            else:
                display_v1_padded = display_v1_state
                display_v2_padded = display_v2_state

            display_norm_v1 = np.linalg.norm(display_v1_padded)
            display_norm_v2 = np.linalg.norm(display_v2_padded)

            if display_norm_v1 == 0 or display_norm_v2 == 0:
                st.markdown(
                    "One of the vectors for display (after padding) is zero, cannot generate or display quantum circuit."
                )
                display_v1_state_norm = display_v1_padded  # Keep as is if norm is 0
                display_v2_state_norm = display_v2_padded  # Keep as is if norm is 0
            else:
                display_v1_state_norm = display_v1_padded / display_norm_v1
                display_v2_state_norm = display_v2_padded / display_norm_v2
                num_encoding_qubits = int(log2(len(display_v1_state_norm)))
                ancilla_qubit_index = 2 * num_encoding_qubits
                # Generate the quantum circuit for the selected vectors.
                qc, _ = qubit_by_qubit_swaptest(display_v1_state_norm, display_v2_state_norm)
        else:
            st.markdown("Input vector for display is of zero dimension, cannot generate or display quantum circuit.")
            display_v1_padded, display_v2_padded = display_v1_state, display_v2_state  # Empty if original was empty
            display_norm_v1, display_norm_v2 = 0, 0
            display_v1_state_norm, display_v2_state_norm = display_v1_state, display_v2_state

        st.markdown(
            rf"""
            ### The circuit used for estimating the user's location.
            The SWAP test compares the Test Vector (from blue dot's current location) with the Fingerprint Vector of the most probable location: **{fp_l_database[prob_loc_pass if prob_loc_pass else list(fp_l_database.keys())[0]]}**.
            Below are the vector transformations involved in preparing these RSSI vectors for the quantum circuit.
            """
        )

        # Section: Vector Preparation for Quantum Circuit
        # Explains how original RSSI vectors are padded (if needed) and normalized.
        st.markdown("#### Vector Preparation for Quantum Circuit")
        if display_vector_dim > 0:
            st.markdown(f"**1. Original RSSI Vectors (Length: {len(display_v1_state)})**")
            st.markdown(f"   - Test Vector (Current Location): `{np.round(display_v1_state, 3)}`")
            st.markdown(
                f"   - Fingerprint Vector ({fp_l_database[prob_loc_pass if prob_loc_pass else list(fp_l_database.keys())[0]]}): `{np.round(display_v2_state, 3)}`"
            )

            if len(display_v1_state) != len(display_v1_padded):
                st.markdown(f"**2. Padded RSSI Vectors (Padded to Length: {len(display_v1_padded)})**")
                st.markdown(f"   - Test Vector (Padded): `{np.round(display_v1_padded, 3)}`")
                st.markdown(f"   - Fingerprint Vector (Padded): `{np.round(display_v2_padded, 3)}`")
                st.markdown(
                    f"**3. Normalized Padded Vectors (for Quantum State Encoding - {num_encoding_qubits} qubits)**"
                )
            else:
                st.markdown(
                    f"**2. Normalized Original Vectors (for Quantum State Encoding - {num_encoding_qubits} qubits)**"
                )

            if display_norm_v1 != 0 and display_norm_v2 != 0:
                st.markdown(f"   - Test Vector (Normalized): `{np.round(display_v1_state_norm, 3)}`")
                st.markdown(f"   - Fingerprint Vector (Normalized): `{np.round(display_v2_state_norm, 3)}`")
            else:
                st.markdown(
                    "   - One or both vectors are zero (or became zero after padding), so normalization is not applicable or results in a zero vector. Circuit cannot be run."
                )
        else:
            st.markdown("Input vectors are empty, cannot display preparation steps.")

        st.markdown(
            rf"""
            In this circuit, the (normalized and potentially padded) vectors are represented as {num_encoding_qubits}-qubit states. 
            The test vector is represented by $| \psi_1 \rangle = \sum_{{i=0}}^{{{2**num_encoding_qubits - 1}}} \alpha_i |i\rangle$, 
            and the fingerprint vector is represented by $| \psi_2 \rangle = \sum_{{i=0}}^{{{2**num_encoding_qubits - 1}}} \beta_i |i\rangle$. 
            The vectors are encoded into qubit states using amplitude encoding.
            The original length of the input RSSI vectors was {display_vector_dim}. Since quantum state preparation requires vector dimensions that are powers of 2, 
            these vectors are padded with zeros (if necessary) to reach a dimension of {2**num_encoding_qubits} before normalization and encoding.
            """
        )

        # Display the quantum circuit and measurement histogram if vectors were valid and circuit was generated.
        if qc and display_vector_dim > 0 and (display_norm_v1 != 0 and display_norm_v2 != 0):
            shots = 10000  # Number of shots for running this display circuit.
            qc_plot = qc.draw(output="mpl", style={"figwidth": 10, "fontsize": 8})
            st.pyplot(fig=qc_plot)

            counts_result = run_circuit(qc, shots=shots)
            hist_plot = plot_histogram(counts_result.get_counts(0))

            # Section: Measurement Explanation
            # Explains how the SWAP test results (P(0)) relate to vector similarity.
            st.markdown(
                rf"""
                ### Measurement

                The circuit computes the similarity between the two quantum states ($|\psi_1\rangle$ for the Test Vector and $|\psi_2\rangle$ for the Fingerprint Vector) by measuring the ancillary qubit (at $q_{{{ancilla_qubit_index}}}$).
                When this ancillary qubit is measured, it collapses to either state $|0\rangle$ or $|1\rangle$. The circuit is executed {shots} times, and the outcomes are tallied. The histogram below shows these results.

                The probability of measuring the ancillary qubit in state $|0\rangle$, denoted as $P(0)$, is directly related to the similarity of the two input states. This probability is given by the formula:
                $P(0) = \frac{{1 + |\langle\psi_1|\psi_2\rangle|^2}}{2}$

                Here, $|\langle\psi_1|\psi_2\rangle|^2$ is the squared inner product (or squared fidelity) of the two normalized quantum states $|\psi_1\rangle$ and $|\psi_2\rangle$. 
                - $|\langle\psi_1|\psi_2\rangle|^2 = 1$ if the states are identical (maximum similarity).
                - $|\langle\psi_1|\psi_2\rangle|^2 = 0$ if the states are orthogonal (no similarity).
                This value ranges from 0 to 1.

                Consequently, the probability $P(0)$:
                - Is $1$ when the states are identical ($|\langle\psi_1|\psi_2\rangle|^2 = 1$), indicating maximum similarity between the original RSSI vectors.
                - Is $0.5$ (or $\frac{1}{2}$) when the states are orthogonal ($|\langle\psi_1|\psi_2\rangle|^2 = 0$), indicating that the original RSSI vectors are maximally dissimilar in the context of this test.
                
                The "Similarity Score" displayed in the sidebar is this $P(0)$ value, obtained from the simulation runs.
                A score closer to 1 implies higher similarity between the test vector and the fingerprint vector.
                """
            )
            st.pyplot(fig=hist_plot)
        else:
            st.markdown(
                "Quantum circuit and histogram cannot be displayed, likely because one or both input vectors were zero or empty."
            )

    # Expander: More Information
    # Provides context about simulation assumptions and references.
    with st.expander("More Information and About the Simulation"):
        st.markdown("""
            When developing the simulation system, several assumptions are made:
            
            1. The map is obstacle-free. Meaning the walls do not affect the Received Signal Strength from different APs.
            2. The RSS values are now calculated using the **Standard Log-Distance Path Model**: `RSSI(d) = A_ref - 10 * n * log10(d / d0)`. 
               Parameters `A_ref` (Reference RSSI), `n` (Path Loss Exponent), and `d0` (Reference Distance) are configurable in the sidebar. 
               The previous model was based on `RSSI = -10*n*log10(distance + 1) - C`.
            3. To perform fingerprint matching, the quantum algorithm proposed by [Shokry et al., 2020](https://arxiv.org/pdf/2106.11751.pdf) is used. This primarily involves the SWAP test circuit for comparing state vector similarity. The state vectors are prepared using amplitude encoding of the (potentially padded and normalized) RSSI vectors.
            4. To estimate the similarity score (probability of measuring '0'), the SWAP test circuit is simulated 10000 times (shots).
            
            The web app renders the circuit diagram for computing the similarity score between the current RSS vector of the receiver (Test Vector) and the Fingerprint Vector identified as the most probable match.
            Feedbacks are very welcome, especially on how the computation of RSS values can be improved/simulated and how the effect of obstacles like walls can be added to the system.
        """)
