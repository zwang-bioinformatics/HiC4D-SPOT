# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Functions to generate the testing data with "perturbation"

import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from .misc import fruitpunch
from .plot_45_degree_heatmap import pcolormesh_45deg, format_ticks

def simulate_TAD_add(perturb_action, perturb_sub_mat, tol, prior, base, non_interacting_base, decay, noise_base, noise_decay, noise_prob):
    print("> Simulating TAD add")
    if tol == 0: perturb_sub_mat_focus = perturb_sub_mat.copy()
    else: perturb_sub_mat_focus = perturb_sub_mat[tol:-tol, tol:-tol].copy()
    
    x_len, y_len = perturb_sub_mat_focus.shape
    for i in range(x_len):
        for j in range(y_len):
            if i < j:   
                continue    # Skip the upper triangle
                
            noise = 0
            
            if i != j:
                distance = np.abs(i-j) + prior
                true_mean = base * (distance**decay)
                
                # calculate noise
                if random.random() < noise_prob:
                    noise = noise_base * (distance**noise_decay)
                
                # add noise
                if random.random() < 0.5:  true_mean += noise
                else:   true_mean -= noise
                
                if true_mean > perturb_sub_mat_focus[i,j]:
                    perturb_sub_mat_focus[i,j] = true_mean
                
    # copy the upper triangle to the lower triangle
    for i in range(x_len):
        for j in range(y_len):
            if i < j:
                perturb_sub_mat_focus[i,j] = perturb_sub_mat_focus[j,i]
    
    # copy the sub-matrix back to the original matrix
    if tol == 0: perturb_sub_mat = perturb_sub_mat_focus
    else: perturb_sub_mat[tol:-tol, tol:-tol] = perturb_sub_mat_focus
    
    return perturb_sub_mat

def simulate_TAD_split(perturb_action, perturb_sub_mat, tol, prior, base, non_interacting_base, decay, noise_base, noise_decay, noise_prob):
    # simulate the TAD split
    print("> Simulating TAD split")
    if tol == 0: perturb_sub_mat_focus = perturb_sub_mat.copy()
    else: perturb_sub_mat_focus = perturb_sub_mat[tol:-tol, tol:-tol].copy()
    
    x_len, y_len = perturb_sub_mat_focus.shape
    
    # Get the middle of the matrix
    middle = x_len // 2
    
    # For > x/2 and < y/2, reduce the values to simulate the split. Non TAD region
    for i in range(x_len):
        for j in range(y_len):
            if i < j:
                continue    # Skip the upper triangle
            
            if i > middle and j < middle:
                distance = np.abs(i-j) + prior
                true_mean = non_interacting_base * (distance**decay)
                
                # calculate noise
                noise = 0
                if random.random() < noise_prob:
                    noise = noise_base * (distance**noise_decay)
                
                # add noise
                # if random.random() < 0.5:  true_mean += noise
                # else:   true_mean -= noise
                true_mean -= noise
                
                if perturb_sub_mat_focus[i,j] > true_mean:
                    perturb_sub_mat_focus[i,j] = true_mean
            
    # copy the lower triangle to the upper triangle
    for i in range(x_len):
        for j in range(y_len):
            if i < j:
                perturb_sub_mat_focus[i,j] = perturb_sub_mat_focus[j,i]
    
    # copy the sub-matrix back to the original matrix
    if tol == 0: perturb_sub_mat = perturb_sub_mat_focus
    else: perturb_sub_mat[tol:-tol, tol:-tol] = perturb_sub_mat_focus
    
    return perturb_sub_mat


def simulate_TAD_strength(perturb_action, perturb_sub_mat, tol, prior, base, non_interacting_base, decay, noise_base, noise_decay, noise_prob):
    # simulate the TAD strength
    print("> Simulating TAD strength")
    if tol == 0: perturb_sub_mat_focus = perturb_sub_mat.copy()
    else: perturb_sub_mat_focus = perturb_sub_mat[tol:-tol, tol:-tol].copy()
    
    x_len, y_len = perturb_sub_mat_focus.shape
    
    for i in range(x_len):
        for j in range(y_len):
            if i <= j:
                continue    # Skip the upper triangle
            
            noise = 0
            
            distance = np.abs(i-j) + prior
            true_mean = non_interacting_base * (1/2) * (distance**decay)
            
            # calculate noise
            if random.random() < noise_prob:
                noise = noise_base * (distance**noise_decay)
            
            # add noise
            if random.random() < 0.5:  true_mean += noise
            else:   true_mean -= noise
            
            perturb_sub_mat_focus[i,j] = true_mean
                
    # copy the lower triangle to the upper triangle
    for i in range(x_len):
        for j in range(y_len):
            if i < j:
                perturb_sub_mat_focus[i,j] = perturb_sub_mat_focus[j,i]
    
    # copy the sub-matrix back to the original matrix
    if tol == 0: perturb_sub_mat = perturb_sub_mat_focus
    else: perturb_sub_mat[tol:-tol, tol:-tol] = perturb_sub_mat_focus
    
    return perturb_sub_mat
    

def simulate_TAD_shift(perturb_action, perturb_sub_mat, tol, prior, base, non_interacting_base, decay, noise_base, noise_decay, noise_prob):
    # simulate the TAD shift
    print("> Simulating TAD shift")
    if tol == 0: perturb_sub_mat_focus = perturb_sub_mat.copy()
    else: perturb_sub_mat_focus = perturb_sub_mat[tol:-tol, tol:-tol].copy()
    
    x_len, y_len = perturb_sub_mat_focus.shape
    
    # shift the sub-matrix to the right diagonally (x,y) -> (x+tol, y+tol)
    # perturb_sub_mat[tol:-tol, tol:-tol] = 0
    for i in range (tol, perturb_sub_mat.shape[0]-tol):
        for j in range(tol, perturb_sub_mat.shape[0]-tol):
            if i <= j:
                continue    # Skip the upper triangle
            
            noise = 0
            
            distance = np.abs(i-j) + prior
            true_mean = non_interacting_base * (distance**decay)
            
            # calculate noise
            if random.random() < noise_prob:
                noise = noise_base * (distance**noise_decay)
            
            # add noise
            true_mean -= noise
            
            perturb_sub_mat[i,j] = true_mean
            perturb_sub_mat[j,i] = true_mean
                
    perturb_sub_mat[2*tol:, 2*tol:] = perturb_sub_mat_focus
            
    return perturb_sub_mat




def simulate_show_hic(mat, timePoint, perturb_range_bin, resolution, fig_name, misc_dir):
    
    os.makedirs(misc_dir, exist_ok=True)
    
    tol = 20
    image_range = (perturb_range_bin[0]-tol, perturb_range_bin[1]+tol)
    sub_mat = mat[image_range[0]:image_range[1], image_range[0]:image_range[1]]
    
    import sys
    import seaborn as sns
    fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
    
    # Visualize the HiC square matrix in this region
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(sub_mat, vmax=100, cmap=fruitpunch)
    ax.set_title(f"{fig_name} HiC matrix - square")
    
    # Draw lines at the perturb_range_bin location
    ax.axvline(x=perturb_range_bin[0] - image_range[0], color='blue', linestyle='--')
    ax.axhline(y=perturb_range_bin[0] - image_range[0], color='blue', linestyle='--')
    ax.axvline(x=perturb_range_bin[1] - image_range[0], color='blue', linestyle='--')
    ax.axhline(y=perturb_range_bin[1] - image_range[0], color='blue', linestyle='--')
    
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(misc_dir, f"{fig_name}_HiC_matrix_square.png"))

    # Visualize the HiC square matrix in this region
    fig, ax = plt.subplots(1, 1, figsize=(15, 2))
    pcolormesh_45deg(ax, sub_mat, start=0, resolution = resolution, vmax=100, cmap=fruitpunch)
    ax.set_ylim(0, 50*resolution)
    format_ticks(ax, rotate=False);
    ax.set_title(f'{fig_name} HiC matrix - 45 degree')
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(misc_dir, f"{fig_name}_HiC_matrix_45deg.png"))