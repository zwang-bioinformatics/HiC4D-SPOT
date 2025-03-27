# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import numpy as np

def load_input_data(loc, partition, args, **kwargs):
    """
    Load the input data for a given partition from the location

    Args:
        loc (str): The location of the data
        partition (str): The partition to load the data for
        args (int): args

    Returns:
        numpy array: The input data for the given partition of shape (number_of_sub_matrices, number_of_time_points, image_size, image_size)
    """
    
    chrs_valid = args['chrs_valid']
    chrs_test = args['chrs_test']
    chrs_predict = args['predict_chr']
        
    print(f"Loading {partition} data", flush=True)
    
    data = []
    
    for chr_idx in range(1, args['num_chr']):
        chrid = str(chr_idx)
        if chr_idx == (args['num_chr']-1):
            chrid = f"X"
            
        if partition == "training":
            if chrid in chrs_valid: continue
            if chrid in chrs_test:  continue
        if partition == "validation" and (chrid not in chrs_valid):
            continue
        if partition == "test" and (chrid not in chrs_test):
            continue
        if partition == "predict" and (chrid != args['predict_chr']):
            continue
        
        # from the loc directory, load the data. eg data_10000bp_1.npy
        data.append(np.load(f"{loc}/data_{args['resolution']}bp_{args['chr_prefix']}{chrid}.npy"))
    
    assert len(data) > 0, f"No data found for {partition}"
    
    data = np.concatenate(data, axis=0) # Stack the data along the 0th axis, shape (number_of_sub_matrices, number_of_time_points, image_size, image_size)
    
    print(f"Data loaded for {partition} of shape {data.shape}", flush=True)
    
    return data