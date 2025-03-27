# Author: Bishal Shrestha
# Date: 03-24-2025  

import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max
from scipy.stats import norm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def get_anomaly_hic(true_hic, pred_hic, dir_out, file_name, verbose=False):
    
    dir_out = os.path.join(dir_out, "combined")
    os.makedirs(dir_out, exist_ok=True)
    
    file_path = os.path.join(dir_out, f"{file_name}.npy")
    try:
        if os.path.exists(file_path):
            if verbose: print(f"Loading anomaly HiC matrix from: {file_path}", flush=True)
            return np.load(file_path)
    except:
        print(f"Error loading file: {file_path}, recomputing anomaly HiC matrix.")
        
    anomaly_hic = []
    for i in range(len(true_hic)):
        diff = np.abs(true_hic[i] - pred_hic[i])
        # clip the values at 0.5 and scale to 0-1
        # diff[diff > 0.5] = 0.5
        # diff = diff / 0.5
        anomaly_hic.append(diff)
    
    # save the anomaly hic matrix
    np.save(file_path, anomaly_hic)
    
    return anomaly_hic


def get_anomaly_hic_refed(anomaly_hic, dir_out, file_name, verbose=False):
    """Use the L1 loss matrix of first timepoint as reference to get more refined anomaly matrix for all other timepoints.
        Output = abs(Anomaly_t - Anomaly_1)

    Args:
        anomaly_hic (_type_): _description_
        dir_out (_type_): _description_
        file_name (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    dir_out = os.path.join(dir_out, "combined")
    os.makedirs(dir_out, exist_ok=True)
    
    file_path = os.path.join(dir_out, f"{file_name}.npy")
    
    anomaly_hic_refed = []
    for i in range(len(anomaly_hic)):
        if i == 0:
            anomaly_hic_refed.append(anomaly_hic[i])
        else:
            anomaly_hic_refed.append(np.abs(anomaly_hic[0] - anomaly_hic[i]))
    
    # save the anomaly hic matrix
    np.save(file_path, anomaly_hic_refed)
    
    return anomaly_hic_refed

