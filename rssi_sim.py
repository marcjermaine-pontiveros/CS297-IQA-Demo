from math import pi, sqrt, fabs, atan, e, log10
import numpy as np 

def rssi(ap, vec):
    """
    Inverse function of Distance Estimation from this presentation:
    https://www.youtube.com/watch?v=CWvRJdF7oVE

    Given the model, compute the rssi from the distance values
    Args:
        ap : access point vector
        vec : reference point vector

    Returns:
        rssi : computed rssi from distance value
    """
    # https://www.youtube.com/watch?v=CWvRJdF7oVE
    # rssi formula:
    n = 5
    C = -20
    distance = np.linalg.norm(np.array(ap) - np.array(vec)) / 100
    rssi = -10*n*log10(distance + 1) - C
    return rssi 
    


def compute_rssi_d(fp_d, ap_d):
    """
    Computes rssi given the database of access points and database of reference points(x, y).

    Args:
        fp_d : Database of reference points
        ap_d : Database of access points 

    Returns:
        fp_rssi_d : Database of RSSI
    """
    fp_rssi_d = dict()
    rssi_dim = len(ap_d)
    for fp_key, fp_value in fp_d.items():
        rssi_l = list()
        for i in range(rssi_dim):
            rssi_v = rssi(fp_value, ap_d[f"ap{i}"])
            rssi_l.append(rssi_v)
        
        fp_rssi_d[fp_key] = rssi_l 
    return fp_rssi_d 

def compute_rssi(test_vector, ap_d):
    rssi_dim = len(ap_d)
    _rssi = list()
    for i in range(rssi_dim):
        rssi_v = rssi(test_vector, ap_d[f"ap{i}"])
        _rssi.append(rssi_v)
    return _rssi 


"""
def rssi_test(ap_x, ap_y, vec_x, vec_y):
    MP = -69 # measured power
    N = 2 # (Constant depends on the Environmental factor. Range 2–4, low to-high strength as explained above)
    dx = vec_x - ap_x 
    dy = vec_y - ap_y 
    px_distance = sqrt(dy**2 + dx**2)
    distance = px_distance / 100
    print("distance is ", distance)

    rssi = MP - 10*N*log10(distance + 1)
    return rssi
"""
