from math import log10 as math_log10  # Use math.log10 for scalar, np.log10 for array if needed.
import numpy as np


def rssi(ap, vec, noise_std_dev=0.0, A_ref=-40.0, n_val=3.0, d0=1.0):
    """
    Computes the Received Signal Strength Indicator (RSSI) using the Standard Log-Distance Path Model.
    Optionally, Gaussian noise can be added to the RSSI value.

    The model is: RSSI(d) = A_ref - 10 * n * log10(d / d0)
    where:
        d is the distance between the transmitter and receiver.
        A_ref is the reference RSSI at a reference distance d0 (typically 1 meter).
        n is the path loss exponent, indicating how quickly the signal attenuates with distance.
        d0 is the reference distance.

    Args:
        ap (list or np.array): Coordinates of the access point (transmitter) [x, y].
        vec (list or np.array): Coordinates of the device/vector (receiver) [x, y].
        noise_std_dev (float, optional): Standard deviation of Gaussian noise to add to the RSSI. Defaults to 0.0.
        A_ref (float, optional): Reference RSSI value at distance d0, in dBm. Defaults to -40.0 dBm.
        n_val (float, optional): Path loss exponent. Defaults to 3.0.
        d0 (float, optional): Reference distance in meters. Defaults to 1.0 meter.

    Returns:
        float: The computed RSSI value in dBm.
    """
    # Calculate Euclidean distance between AP and device vector.
    # Assuming input coordinates are in units that, when divided by 100.0, yield meters.
    distance_m = np.linalg.norm(np.array(ap) - np.array(vec)) / 100.0

    # Distance capping: To avoid issues with log10(0) or extremely high RSSI at very short distances,
    # if the calculated distance is less than 10% of the reference distance d0,
    # cap it at 0.1 * d0. This prevents the argument of log10 from being <= 0.
    if distance_m < 0.1 * d0:
        distance_m = 0.1 * d0

    # Standard Log-Distance Path Model formula
    # Using math.log10 as distance_m and d0 are scalars here.
    rssi_val = A_ref - (10 * n_val * math_log10(distance_m / d0))

    if noise_std_dev > 0:
        # Add Gaussian noise if a standard deviation is provided
        noise = np.random.normal(0, noise_std_dev)
        rssi_val += noise

    return rssi_val


def compute_rssi_d(fp_d, ap_d, noise_std_dev=0.0, num_samples_for_avg=1, A_ref=-40.0, n_val=3.0, d0=1.0):
    """
    Computes RSSI values for a database of fingerprint (reference) points against a database of access points,
    using the Standard Log-Distance Path Model.
    This function can simulate more realistic RSSI readings by incorporating Gaussian noise and averaging multiple samples.

    Args:
        fp_d (dict): Database of fingerprint points {fp_id: [x, y], ...}.
        ap_d (dict): Database of access points {ap_id: [x, y], ...}.
        noise_std_dev (float, optional): Std dev of Gaussian noise for RSSI. Defaults to 0.0.
        num_samples_for_avg (int, optional): Number of RSSI samples to average. Defaults to 1.
        A_ref (float, optional): Reference RSSI value (dBm) at distance d0. Defaults to -40.0.
        n_val (float, optional): Path loss exponent. Defaults to 3.0.
        d0 (float, optional): Reference distance (meters). Defaults to 1.0.

    Returns:
        dict: Database of RSSI vectors {fp_id: [rssi_ap0, rssi_ap1, ...], ...}.
    """
    fp_rssi_d = dict()
    rssi_dim = len(ap_d)

    for fp_key, fp_value in fp_d.items():
        avg_rssi_l = list()
        for i in range(rssi_dim):
            ap_coords = ap_d[f"ap{i}"]
            samples = [rssi(fp_value, ap_coords, noise_std_dev, A_ref, n_val, d0) for _ in range(num_samples_for_avg)]
            avg_rssi_v = (
                np.mean(samples)
                if num_samples_for_avg > 0
                else rssi(fp_value, ap_coords, noise_std_dev, A_ref, n_val, d0)
            )
            avg_rssi_l.append(avg_rssi_v)

        fp_rssi_d[fp_key] = avg_rssi_l
    return fp_rssi_d


def compute_rssi(test_vector, ap_d, noise_std_dev=0.0, num_samples_for_avg=1, A_ref=-40.0, n_val=3.0, d0=1.0):
    """
    Computes the RSSI vector for a single test vector (e.g., user's current location)
    against a database of access points, using the Standard Log-Distance Path Model.
    This function can simulate more realistic RSSI readings by incorporating Gaussian noise and averaging multiple samples.

    Args:
        test_vector (list or np.array): Coordinates of the test point [x, y].
        ap_d (dict): Database of access points {ap_id: [x, y], ...}.
        noise_std_dev (float, optional): Std dev of Gaussian noise for RSSI. Defaults to 0.0.
        num_samples_for_avg (int, optional): Number of RSSI samples to average. Defaults to 1.
        A_ref (float, optional): Reference RSSI value (dBm) at distance d0. Defaults to -40.0.
        n_val (float, optional): Path loss exponent. Defaults to 3.0.
        d0 (float, optional): Reference distance (meters). Defaults to 1.0.

    Returns:
        list: A list of (averaged) RSSI values from each AP for the given test_vector.
    """
    rssi_dim = len(ap_d)
    _rssi_avg = list()

    for i in range(rssi_dim):
        ap_coords = ap_d[f"ap{i}"]
        samples = [rssi(test_vector, ap_coords, noise_std_dev, A_ref, n_val, d0) for _ in range(num_samples_for_avg)]
        avg_rssi_v = (
            np.mean(samples)
            if num_samples_for_avg > 0
            else rssi(test_vector, ap_coords, noise_std_dev, A_ref, n_val, d0)
        )
        _rssi_avg.append(avg_rssi_v)

    return _rssi_avg
