# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from .misc import fruitpunch

def generate_anomaly_loop(mat_hic, maxV, timePoint, resolution, perturb_info, region_range, args):
    """Generate anolamy in the HiC matrix 
        the perturb_info[perturn_type] is a list of df: ['chromosome1', 'x1', 'x2', 'chromosome2', 'y1', 'y2']

    Returns:
        mat_hic: output from clr.matrix(balance=balance).fetch(chrid, chrid)
    """
    
    # Get point of interest
    
    assert np.min(mat_hic) >= -1 and np.max(mat_hic) <= 1, "mat_hic values should be between -1 and 1"
    
    region_range_bin = (region_range[0] // resolution, region_range[1] // resolution)
    
    perturb_type_func = {
        'remove_loop': remove_loop,
        'add_loop': add_loop,
    }
    
    visualize_hic(mat_hic, region_range_bin, fig_name=f'{timePoint}_before', misc_dir=args['misc_output_dir'])
    for perturb_type in perturb_info:
        print(f"Generating anomaly for {perturb_type}")
        print(f"Number of regions: {len(perturb_info[perturb_type])}")
        for region in perturb_info[perturb_type][0]:
            
            # convert physical position to bin position
            midx = ( region[1] + region[2] ) // 2
            midy = ( region[4] + region[5] ) // 2
            
            region_bin = region.copy()
            region_bin[1] = region_bin[1] // resolution
            region_bin[2] = region_bin[2] // resolution
            region_bin[4] = region_bin[4] // resolution
            region_bin[5] = region_bin[5] // resolution 
            
            midx_bin = midx // resolution
            midy_bin = midy // resolution
            region_mid_bin = (midx_bin, midy_bin)
            
            mat_hic = perturb_type_func[perturb_type](mat_hic, region_bin, region_range_bin, top_loops=perturb_info[perturb_type][1], region_mid_bin=region_mid_bin)
            
        visualize_hic(mat_hic, region_range_bin, fig_name=f'{timePoint}_after_{perturb_type}', misc_dir=args['misc_output_dir'])
        print(f"Anomaly generated for {perturb_type}", flush=True)

    return mat_hic


def add_loop(mat_hic, region_bin, region_range_bin, **kwargs):
    print(f"Adding anomaly to the region: {region_bin}")
    mat_hic_anomaly = mat_hic.copy()
    loops = kwargs['top_loops']
    resolution = 10000
    
    tol_left = 1
    tol_right = 2
    length = tol_left + tol_right + 1
    middle = length // 2
    middle_range = (middle - 1, middle + 2)
    
    # Create an empty matrix to get the average value of the loops so I can use it to add to the region
    true_loop_mat = np.zeros((length, length))
    for loop in loops:
        loop_bin = loop.copy()
        loop_bin[1] = loop_bin[1] // resolution
        loop_bin[2] = loop_bin[2] // resolution
        loop_bin[4] = loop_bin[4] // resolution
        loop_bin[5] = loop_bin[5] // resolution
        
        temp_mat = mat_hic[loop_bin[1]-tol_left:loop_bin[2]+tol_right, loop_bin[4]-tol_left:loop_bin[5]+tol_right]
        
        # if the temp_mat is not 6x6, then pad it with zeros so it is in center
        assert temp_mat.shape == (length, length), f"Temp mat shape: {temp_mat.shape}"
        
        true_loop_mat += temp_mat
        
        # at the top and bottom corner, decrease the value by 1/2
        true_loop_mat[0, 0] = true_loop_mat[0, 0] * (1/2)
        true_loop_mat[0, -1] = true_loop_mat[0, -1] * (1/2)
        true_loop_mat[-1, 0] = true_loop_mat[-1, 0] * (1/2)
        true_loop_mat[-1, -1] = true_loop_mat[-1, -1] * (1/2)
        
        
    true_loop_mat = true_loop_mat / len(loops)
    
    # only replace the values if it is less than the true_loop_mat
    for i in range(region_bin[1]-tol_left, region_bin[2]+tol_right):
        for j in range(region_bin[4]-tol_left, region_bin[5]+tol_right):
            # corresponding value in the true_loop_mat
            i2 = i - (region_bin[1]-tol_left)
            j2 = j - (region_bin[4]-tol_left)
            if mat_hic_anomaly[i, j] < true_loop_mat[i-(region_bin[1]-tol_left), j-(region_bin[4]-tol_left)]:
                mat_hic_anomaly[i, j] = true_loop_mat[i-(region_bin[1]-tol_left), j-(region_bin[4]-tol_left)]
                mat_hic_anomaly[j, i] = mat_hic_anomaly[i, j]
    
    mat_hic_anomaly = mat_hic_anomaly.clip(0, 1)
    
    return mat_hic_anomaly


def remove_loop(mat_hic, region_bin, region_range_bin, **kwargs):
    """
    Remove a loop from a Hi-C matrix.

    Args:
        mat_hic (numpy.ndarray): Hi-C matrix.
        region_bin (list): List of bin indices defining the loop region.
        region_range_bin (list): List of bin indices defining the range of the loop region.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Hi-C matrix with the loop removed.
    """
    
    mat_hic_anomaly = mat_hic.copy()
    
    startx = region_bin[1] - 2
    endx = region_bin[2] + 2
    starty = region_bin[4] - 2
    endy = region_bin[5] + 2
    
    midx = (startx + endx) // 2
    midy = (starty + endy) // 2
    lengthx = endx - startx + 1
    lengthy = endy - starty + 1
    
    for (i, j) in [(i, j) for i in range(startx, endx+1) for j in range(starty, endy+1)]:
        mat_hic_anomaly[i, j] = np.mean([mat_hic_anomaly[i, j-lengthy], mat_hic_anomaly[i+lengthx, j], mat_hic_anomaly[i, j+lengthy], mat_hic_anomaly[i-lengthx, j]])
        mat_hic_anomaly[j, i] = mat_hic_anomaly[i, j]
        
    return mat_hic_anomaly



def visualize_hic(mat_hic, region_bin, fig_name, misc_dir):
    """
    Visualize the HiC matrix.

    Args:
        mat_hic (numpy.ndarray): HiC matrix.
        region_bin (tuple): Tuple containing the start and end bin indices of the region to visualize.
        fig_name (str): Name of the output figure.

    Returns:
        None
    """
    
    os.makedirs(misc_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    region_start = region_bin[0] 
    region_end = region_bin[1] 
    
    # mid1 = (region_bin[1] + region_bin[2]) // 2
    # mid2 = (region_bin[4] + region_bin[5]) // 2
    
    im = ax.imshow(mat_hic[region_start:region_end, region_start:region_end], cmap=fruitpunch)
    # ax.scatter(mid1 - region_start, mid2 - region_start, s=200, facecolors='none', edgecolors='blue', marker='o', alpha=1)
    # ax.scatter(mid2 - region_start, mid1 - region_start, s=200, facecolors='none', edgecolors='blue', marker='o', alpha=1)
    
    # Set the x and y ticks to represent the exact bin values
    n = 20  # Set every nth tick
    ax.set_xticks(np.arange(0, region_end - region_start, n))
    ax.set_yticks(np.arange(0, region_end - region_start, n))
    ax.set_xticklabels(np.arange(region_start, region_end, n))
    ax.set_yticklabels(np.arange(region_start, region_end, n))
    
    # title
    ax.set_title(f'HiC Matrix with loop region: {region_bin}')
    
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'{misc_dir}/testing_loops_anon_{fig_name}.png')
    