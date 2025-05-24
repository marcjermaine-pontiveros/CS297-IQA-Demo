# Quantum Fingerprint Matching for Indoor Localization - Streamlit Demo

CS 297 : Introduction to Quantum Algorithms Final Report

This project demonstrates an indoor localization system using a quantum fingerprint matching protocol. The core of the protocol relies on the **SWAP test algorithm**, implemented using Qiskit, to determine the similarity between a real-time Received Signal Strength Indicator (RSSI) vector (from a "test location") and a database of pre-recorded RSSI fingerprint vectors.

The application is built with Streamlit, providing an interactive interface to simulate user movement, adjust RSSI signal properties, and visualize the quantum circuit and its results.

## Features & Recent Improvements

*   **Interactive Map Navigation:**
    *   Users simulate movement on a floor plan using **Up, Down, Left, and Right buttons** in the sidebar.
    *   The **Movement Step Size** is configurable via a slider, allowing for finer or coarser navigation.
    *   The application displays Access Points (APs), pre-defined Reference Points (RPs), and the user's current test location (blue dot).
*   **Advanced RSSI Simulation:**
    *   RSSI values are calculated using the **Standard Log-Distance Path Model**: `RSSI(d) = A_ref - 10 * n * log10(d / d0)`.
    *   **Configurable Model Parameters:** The key parameters of this model, `A_ref` (Reference RSSI), `n` (Path Loss Exponent), and `d0` (Reference Distance), are all adjustable via sliders in the UI.
    *   **Simulation Realism Enhancements:**
        *   **RSSI Noise:** Gaussian noise can be added to RSSI readings (both for fingerprint generation and test vector calculation) to simulate real-world signal fluctuations. The standard deviation of this noise is user-configurable.
        *   **RSSI Averaging:** Multiple RSSI samples can be generated and averaged for each AP-location pair to provide a more stable signal reading. The number of samples is user-configurable.
*   **Quantum Fingerprint Matching:**
    *   RSSI vectors from the test location and reference points are compared using a quantum SWAP test.
    *   **Robustness for Variable Vector Dimensions:** The system supports RSSI vectors of arbitrary dimensions (e.g., varying number of APs). Vectors are automatically padded with zeros to the next power-of-2 dimension, a requirement for efficient amplitude encoding into quantum states.
    *   The similarity score (probability of measuring '0' in the SWAP test) is used to determine the most probable location of the user.
*   **Enhanced User Interface (UI) & Explainability:**
    *   **Detailed Similarity Scores:** The sidebar displays a sorted table of all reference points and their computed similarity scores to the current test location.
    *   **Vector Transformation Visualization:** The "Quantum Circuit" expander details the step-by-step transformation of the test vector and the selected fingerprint vector: (1) Original RSSI vectors, (2) Padded vectors (if applicable, showing new dimensions), and (3) Normalized vectors for quantum state encoding.
    *   **Clearer Explanations:** The "Measurement" section provides an in-depth explanation of the SWAP test results, including the formula `P(0) = (1 + |⟨ψ|φ⟩|^2) / 2` and its implications for vector similarity.
*   **Qiskit Integration:** Quantum circuits for the SWAP test are built and simulated using Qiskit and Qiskit Aer.

## How It Works

1.  **Fingerprint Database:** A set of Reference Points (RPs) with known coordinates has a corresponding RSSI vector (list of RSSI values from each AP). This database is generated dynamically based on the configured RSSI simulation settings (Log-Distance model parameters, noise, averaging).
2.  **Test Vector:** The user's current location (the "blue dot"), navigated via sidebar buttons, generates an RSSI test vector, also subject to the same simulation settings.
3.  **Padding & Normalization:** Both test and fingerprint vectors are padded with zeros if their dimension is not a power of 2. They are then normalized to unit length for quantum state encoding.
4.  **SWAP Test:** For each fingerprint in the database, a SWAP test circuit is constructed. The test vector and the fingerprint vector are encoded into two separate quantum registers. The SWAP test measures an ancillary qubit. The probability of measuring '0' (`P(0)`) is `(1 + |⟨ψ_test|ψ_fp⟩|^2) / 2`, where `|⟨ψ_test|ψ_fp⟩|^2` is the squared inner product of the two quantum states. This `P(0)` value serves as the similarity score.
5.  **Localization:** The fingerprint with the highest `P(0)` is identified as the most probable location of the user.

## Setup and Running the Application

To run the Streamlit app locally:

1.  **Prerequisites:**
    *   Python 3.7+
    *   Pip (Python package installer)

2.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt 
    ```
    *(If `requirements.txt` is not provided, you'll need to install `streamlit`, `numpy`, `pandas`, `matplotlib`, `qiskit`, `qiskit-aer` manually via pip).*

4.  **Run the Streamlit app:**
    ```bash
    streamlit run swaptest.py
    ```
    This will typically open the application in your default web browser.

## Sample Visuals

**Sample Circuit Diagram (as shown in the app):**
*(The actual circuit diagram displayed will vary based on the number of APs/vector dimension)*
![](resources/circuit.png)

**Sample Measurement Histogram (as shown in the app):**
![](resources/measurement.png)
