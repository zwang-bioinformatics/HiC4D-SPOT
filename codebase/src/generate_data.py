# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Generate data for training, validation, and testing, along with anomaly generation presented in the paper.
# In args_mega, select appropriate number before running the script. More detail on args_mega.py

import os
import sys
import time
import json
import math
import numpy as np
import argparse
import importlib

import cooler
import cooltools

import pyBigWig

# from args import args_Data
from utils.generate_anomaly_loop import generate_anomaly_loop
from utils.generate_anomaly_tad import *
from utils.loops import get_true_loops
from utils.seq_manager import *

from concurrent.futures import ProcessPoolExecutor

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
args = importlib.import_module(module_name).get_args()

if args['verbose']: print(f"Time: {time.ctime()}, Process ID: {os.getpid()}, Arguments: {json.dumps(args, indent=4)}", flush=True)

def generate_input(chrid, res, image_size, step, extend_right, maxV, balance, ids, num_chr,  fcool, dir_out, args, anomaly=False, verbose=True):
    """ Generate input data for training, validation, and testing

    Args: 
        chrid (str): Chromosome ID
        res (int): Resolution in bp
        image_size (int): Size of the sub-matrix
        step (int): Step size for the sub-matrix
        extend_right (int): Extend right for the sub-matrix
        maxV (int): Maximum value for normalization
        balance (bool): Whether to balance the matrix
        ids (list): List of time points
        num_chr (int): Number of chromosomes
        fcool (str): Path to the cooler file
        dir_out (str): Output directory
        args (dict): Arguments
        anomaly (bool): Whether to generate anomaly
        verbose (bool): Whether to print verbose output
    """
    if verbose: print(f"\n\nGenerating input data for {chrid}, resolution: {res}bp, image size: {image_size}, step: {step}, extend_right: {extend_right}, maxV: {maxV}", flush=True)
        
    data_timePoints = []
    data_index = []
    
    for idx, timePoint in enumerate(ids):

        ficool = os.path.join(fcool, timePoint + ".cool")
        clr = cooler.Cooler(ficool)
        mat_hic = clr.matrix(balance=balance).fetch(chrid, chrid)
        mat_hic = np.nan_to_num(mat_hic)    # nan to zero
        
        if anomaly:
            print(f"Generating anomaly for {chrid} at {timePoint}", flush=True)
            
            # =======================================================================
            if args['aug_type'] == "tad":
                assert args['data'] == 'Du', f"Data should be Du, not {args['data']}"
                
                resolution = args['resolution']
                
                base = 400          # Kt
                non_interacting_base = 200  
                prior = 1           # p
                decay = -0.8        # ct
                noise_base = 50     # Kn
                noise_decay = -0.7  # cn
                noise_prob = 0.5
                
                perturb_info = {
                    'early_2cell': {
                        'add': [(51_100_000, 52_200_000)], 
                    },
                    '8cell': {
                        'add': [(49_350_000, 50_560_000)],
                    },
                    'ICM':{
                        'shift': [(52_150_000, 52_980_000)],
                    },
                    'mESC_500': {
                        'split': [(49_350_000, 50_560_000)], 
                        'strength': [(54_450_000, 55_200_000)],
                    }
                }
                
                perturb_func = {
                    'add': simulate_TAD_add,
                    'split': simulate_TAD_split,
                    'strength': simulate_TAD_strength,
                    'shift': simulate_TAD_shift,
                }
                
                if timePoint in perturb_info:
                    print("Working on ", timePoint)
                    for perturb_action in perturb_info[timePoint]:
                        perturb_ranges = perturb_info[timePoint][perturb_action]
                        for perturb_range in perturb_ranges:
                            perturb_range_bin = (perturb_range[0] // resolution, perturb_range[1] // resolution)
                            simulate_show_hic(mat_hic, timePoint, perturb_range_bin, resolution, fig_name=f'{chrid}_{timePoint}_before_{perturb_action}', misc_dir=args['misc_output_dir'])
                            print("Perturbing", perturb_range_bin)
                            
                            tol = 10
                            
                            perturb_sub_mat = mat_hic[perturb_range_bin[0]-tol:perturb_range_bin[1]+tol, perturb_range_bin[0]-tol:perturb_range_bin[1]+tol]
                            perturb_sub_mat_perturbed = perturb_sub_mat.copy()
                            # perturb_sub_mat = mat_hic[perturb_range_bin[0]:perturb_range_bin[1], perturb_range_bin[0]:perturb_range_bin[1]]
                            
                            # use the perturb function to simulate the TAD
                            perturb_sub_mat_perturbed = perturb_func[perturb_action](perturb_action, perturb_sub_mat, tol, prior, base, non_interacting_base, decay, noise_base, noise_decay, noise_prob)
                            
                            # Integrate the perturbed sub-matrix back to the original matrix
                            mat_hic[perturb_range_bin[0]-tol:perturb_range_bin[1]+tol, perturb_range_bin[0]-tol:perturb_range_bin[1]+tol] = perturb_sub_mat_perturbed
                            simulate_show_hic(mat_hic, timePoint, perturb_range_bin, resolution, fig_name=f'{chrid}_{timePoint}_after_{perturb_action}', misc_dir=args['misc_output_dir'])

                    
                mat_hic[mat_hic > maxV] = maxV      # clip max value
                mat_hic = mat_hic / maxV            # norm to [0,1]
                
            # =======================================================================
            
            if args['aug_type'] == "loop":
                
                mat_hic[mat_hic > maxV] = maxV      # clip max value
                mat_hic = mat_hic / maxV            # norm to [0,1]
        
                resolution = args['resolution']
                assert args['data'] == 'Reed', f"Data should be Reed, not {args['data']}"
                region_range = args['aug_region_range'] #(93_000_000, 96_000_000)
                top_loops = get_true_loops(chr_list=[f'chr{chrid}'], region=region_range, development_mode="all", num_loops=5, loop_loc=args['loop_info'])
                gain_late = get_true_loops(chr_list=[f'chr{chrid}'], region=region_range, development_mode="gain_late", num_loops=False, loop_loc=args['loop_info'])
                
                perturb_info = {
                    '8': {
                        't2': {
                            'add_loop': [gain_late, top_loops],
                        },
                        't5': {
                            'remove_loop': [top_loops, []],
                        },
                    }
                }
                if chrid in perturb_info.keys() and timePoint in perturb_info[chrid].keys():
                    print(f"> Generating anomaly for {chrid} at {timePoint}", flush=True)
                    mat_hic = generate_anomaly_loop(mat_hic, maxV, timePoint, resolution, perturb_info[chrid][timePoint], region_range, args)


        
        
        chr_len = clr.chromsizes[chrid]                     # Length of the chromosome
        n_bins = math.ceil(chr_len/res)
        
        allInds = np.arange(0, n_bins-image_size, step)     # Starting index of the sub-matrices
        lastInd = allInds[len(allInds)-1]                   # Last index of the sub-matrices
        if (lastInd + image_size) < n_bins:                 # If the last index + image_size is less than the number of bins, then add the last index
            allInds = np.append(allInds, n_bins - image_size)

        subMats = []        # Store the sub-matrices
        index   = []        # Store the starting index of the sub-matrix
        
        
        for j in allInds:
            
            # Position of the sub-matrix in row
            idx_sj, idx_ej = j, j + image_size - 1          # idx_sj: starting index of the sub-matrix, idx_ej: ending index of the sub-matrix. Both are inclusive
            psj, pej = idx_sj * res, idx_ej * res + res     # psj: physical starting position of the sub-matrix, pej: physical ending position of the sub-matrix. Both are inclusive. 
            
            # Check if the sub-matrix is going out of the chromosome
            if pej > chr_len:
                pej = chr_len
            
            regionj = (chrid, psj, pej)
            
            # Position of the sub-matrix in column (horizontal): extend_right
            for k in range(extend_right+1):                     # From diagonal j,j to j,j+extend_right*step
                
                idx_sk = j + k*step                             # idx_sk: starting index of the sub-matrix,  
                idx_ek = idx_sk + image_size - 1                # idx_ek: ending index of the sub-matrix. Both are inclusive 
                psk, pek = idx_sk * res, idx_ek * res + res     # psk: physical starting position of the sub-matrix, pek: physical ending position of the sub-matrix. Both are inclusive.
                
                if idx_ek >= n_bins:
                    continue
                
                if pek > chr_len:
                    pek = chr_len
                
                regionk = (chrid, psk, pek)
                
                # Get the sub-matrix
                # sub_mat_hic = clr.matrix(balance=balance).fetch(regionj, regionk) # When using this, need to normalize the matrix repeatedly
                sub_mat_hic = mat_hic[idx_sj:idx_ej+1, idx_sk:idx_ek+1] # datatype: numpy.ndarray
                
                assert sub_mat_hic.shape == (image_size, image_size), f"Sub-matrix shape: {sub_mat_hic.shape}, Expected: ({image_size}, {image_size})"
                
                subMats.append(sub_mat_hic)
                index.append([idx_sj, idx_ej+1, idx_sk, idx_ek+1])
        
        subMats = np.array(subMats)
        index = np.array(index)
        data_timePoints.append(subMats)
        data_index.append(index)

    data_timePoints = np.array(data_timePoints)                     # (number_of_time_points, number_of_sub_matrices, image_size, image_size)
    data_timePoints2 = np.transpose(data_timePoints, (1,0,2,3))     # (number_of_sub_matrices, number_of_time_points, image_size, image_size)
    data_index = np.array(data_index)                               # (number_of_time_points, number_of_sub_matrices, 2)
    data_index2 = np.transpose(data_index, (1,0,2))                 # (number_of_sub_matrices, number_of_time_points, 2)
    
    # Save individual chromosome files
    if verbose: print(f"Saving data_{res}bp_{chrid}.npy and index of shape {data_timePoints2.shape}", flush=True)
    np.save(os.path.join(dir_out, f"data_{res}bp_{chrid}.npy"), data_timePoints2)
    np.save(os.path.join(dir_out, f"data_{res}bp_{chrid}_index.npy"), data_index2)



if __name__ == "__main__":
    
    res = args['resolution']
    image_size = args['sub_matrix_size']
    step = args['step']
    maxV = args['maxV']
    balance = args['cooler_balance']
    num_chr = args['num_chr']
    
    os.makedirs(args['data_output_dir'], exist_ok=True)
    
    if args['ids']:
        ids = args['ids']
        if args['verbose']: print(f"Using provided IDs: {ids}", flush=True)
        assert isinstance(ids, list), f"IDs should be a list, not {type(ids)}"
    else:
        ids = [f"t{i}" for i in range(1, args['num_timepoints']+1)]
        if args['verbose']: print(f"Generated IDs: {ids}", flush=True)

    chrs = [f"{args['chr_prefix']}{i}" for i in range(1, num_chr)]
    print(f"Chromosomes: {chrs}", flush=True)
    if args['chr_prefix']: chrs[-1] = "chrX"    # 22 is X for hg19
    else:   chrs[-1] = "X"

    extend_right = (2_000_000//res - image_size)//step    # calculate the extend_right based on the image_size and step. In total, it should cover 2Mb
    if args['verbose']: print(f"extend_right: {extend_right}", flush=True)

    if not args['augmentation']:
        with ProcessPoolExecutor() as executor:
            futures = []
            for chrid in chrs:
                futures.append(executor.submit(generate_input, chrid, res, image_size, step, extend_right, maxV, balance, ids, num_chr, args['cooler_dir'], args['data_output_dir'], args, args['augmentation'], args['verbose']))  
            for future in futures:
                future.result()
    else:
        generate_input(chrid=f"{args['chr_prefix']}{args['aug_chr']}", res=res, image_size=image_size, step=step, extend_right=extend_right, maxV=maxV, balance=balance, ids=ids, num_chr=num_chr, fcool=args['cooler_dir'], dir_out=args['data_output_dir'], args=args, anomaly=args['augmentation'], verbose=args['verbose'])
    
    print(f"Time: {time.ctime()}, Process ID: {os.getpid()}, Done!", flush=True)
