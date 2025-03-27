# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import numpy as np
import pandas as pd
import cooler

def save_hic_matrix(hics, dir_out, file_name, verbose=False):
    """Save the HiC matrices for multiple timepoints in .matrix format.

    Args:
        hics (np.ndarray): HiC matrices of shape (timepoints, num_bins, num_bins)
        dir_out (str): Directory to save the HiC matrices
        file_name (str): Name of the file to save
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose: print(f"Saving Hi-C matrices as .matrix files for each time point", flush=True)
    dir_out = os.path.join(dir_out, ".matrix")
    os.makedirs(dir_out, exist_ok=True)
    if isinstance(hics, list):  hics = np.array(hics)
    
    num_timepoints = hics.shape[0]
    for t in range(num_timepoints):
        out_file = f"{dir_out}/{file_name}_t{t+1}.matrix"
        # check if the file already exists, if yes, continue
        if os.path.exists(out_file):
            if verbose: 
                print(f"File already exists: {out_file}")
            continue
        np.savetxt(out_file, hics[t], delimiter='\t')
        if verbose: 
            print(f"HiC {out_file} saved.")


def save_hic_matrix_compressed(hics, dir_out, file_name, verbose=False):
    """Save the HiC matrices for multiple timepoints in compressed .npz format.

    Args:
        hics (np.ndarray): HiC matrices of shape (timepoints, num_bins, num_bins)
        dir_out (str): Directory to save the HiC matrices
        file_name (str): Name of the file to save
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose: print(f"Saving Hi-C matrices as .npz files for each time point", flush=True)
    dir_out = os.path.join(dir_out, ".matrix")
    os.makedirs(dir_out, exist_ok=True)

    if isinstance(hics, list):  hics = np.array(hics)
    num_timepoints = hics.shape[0]
    for t in range(num_timepoints):
        out_file = f"{dir_out}/{file_name}_t{t+1}.npz"
        # check if the file already exists, if yes, continue
        if os.path.exists(out_file):
            if verbose: 
                print(f"File already exists: {out_file}")
            continue
        np.savez_compressed(out_file, hics[t])
        if verbose: 
            print(f"HiC {out_file} saved.")


def save_hic_as_cool(hic_matrices, num_bins, bin_size, chromosome, dir_out, output_filename, assembly='hg19', verbose=False):
    """
    Save Hi-C matrices as .cool files for each time point.

    Args:
        hic_matrices (np.ndarray): HiC matrices of shape (timepoints, num_bins, num_bins)
        num_bins (int): Total number of bins in the matrix.
        bin_size (int): Size of each bin in base pairs.
        chromosome (str): Chromosome name (e.g., 'chr1').
        dir_out (str): Directory to output the .cool files.
        output_filename (str): Base output filename for the .cool files.
        assembly (str): Genome assembly version (default: 'hg19').
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose: print(f"Saving Hi-C matrices as .cool files for each time point", flush=True)
    # Ensure the output directory exists
    dir_out = os.path.join(dir_out, ".cool")
    os.makedirs(dir_out, exist_ok=True)
    
    if isinstance(hic_matrices, list):  hic_matrices = np.array(hic_matrices)
    num_timepoints = hic_matrices.shape[0]
    for t in range(num_timepoints):
        hic_matrix = hic_matrices[t]
        out_file = f"{dir_out}/{output_filename}_t{t+1}.cool"
        # Check if the file already exists, if yes, continue
        if os.path.exists(out_file):
            if verbose: 
                print(f"File already exists: {out_file}")
            continue
        
        # Convert the upper triangle of the matrix to a DataFrame with bin1, bin2, count
        rows, cols = np.triu_indices_from(hic_matrix, k=0)  # Get upper triangle indices
        data = pd.DataFrame({
            'bin1_id': rows,
            'bin2_id': cols,
            'count': hic_matrix[rows, cols]
        })

        # Filter out zero counts to reduce file size
        data = data[data['count'] > 0]

        # Create a DataFrame for bins
        bins = pd.DataFrame({
            'chrom': [chromosome] * num_bins,
            'start': np.arange(0, num_bins * bin_size, bin_size),
            'end': np.arange(bin_size, (num_bins + 1) * bin_size, bin_size)
        })

        # Create the .cool file
        cooler.create_cooler(
            cool_uri=out_file,
            bins=bins,
            pixels=data,
            columns=['count'],  # Specify columns if there are additional ones
            dtypes={'count': 'float64'},  # Ensure dtype is float64 for floating-point counts
            metadata={'description': f'Hi-C data for time point {t+1}', 'assembly': assembly},
            assembly=assembly,
            symmetric_upper=True,  # Assume input data is upper triangular
            ensure_sorted=True,    # Ensure data is sorted
            boundscheck=True,      # Check that bin IDs are valid
            dupcheck=True,         # Check for duplicates
            triucheck=True         # Ensure triangle property holds
        )

        if verbose: 
            print(f"Cool file created: {out_file}")
        

def save_cool_as_h5(num_timepoints, dir_out, output_filename, verbose=False):
    """
    Converts .cool files to .h5 format using hicConvertFormat.
    
    Args:
        num_timepoints (int): Number of timepoints
        dir_out (str): Directory where .cool files are located and .h5 files will be saved
        output_filename (str): Base output filename for the converted files
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    import subprocess
    cool_loc_dir = os.path.join(dir_out, ".cool")
    h5_loc_dir = os.path.join(dir_out, ".h5")
    os.makedirs(h5_loc_dir, exist_ok=True)
    
    if verbose: 
        print("Running hicConvertFormat")
    # cool to h5
    for t in range(num_timepoints):
        cool_file = os.path.join(cool_loc_dir, f"{output_filename}_t{t+1}.cool")
        h5_file = os.path.join(h5_loc_dir, f"{output_filename}_t{t+1}.h5")
        
        # check if the file already exists, if yes, continue
        if os.path.exists(h5_file):
            if verbose: 
                print(f"File already exists: {h5_file}")
            continue

        command = f"hicConvertFormat --matrices {cool_file} --outFileName {h5_file} --inputFormat cool --outputFormat h5"
        subprocess.run(command, shell=True)
        if verbose:
            print(f"Converted {cool_file} to {h5_file}")
