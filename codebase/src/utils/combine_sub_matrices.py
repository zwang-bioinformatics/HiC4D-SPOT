# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import gc
import numpy as np
from tqdm import tqdm

def get_combined_hic(file_loc, file_index_loc, num_timepoints, num_bins, sub_mat_n, dir_out, file_name, verbose=False):
    """Combine HiC matrix from sub-matrices(windows) for a chromosome in different time points

    Args:
        file_loc (str): Location of the HiC sub-matrix file (bs, time, num_bins, num_bins)
        file_index_loc (str): Location of the index matrix file (bs, time, 2)
        num_timepoints (int): Number of time points counting from 1
        num_bins (int): Number of bins in the HiC matrix
        sub_mat_n (int): Number of bins of the sub-matrix

    Returns:
        list: List of HiC matrix for each time point
    """

    print(f"\nCombining HiC Matrix from sub-matrices(windows): Source: {file_loc} & Destination: {file_name}", flush=True)
    
    dir_out = os.path.join(dir_out, "combined")
    os.makedirs(dir_out, exist_ok=True)
    
    save_type = "mmap"
    file_path = os.path.join(dir_out, f"{file_name}.{save_type}")
    
    # Check if the file exists
    if os.path.exists(file_path):
        if verbose: print(f"Loading combined HiC matrix from: {file_path}", flush=True)
        # hic = np.load(file_path) # for npy file
        hic = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_timepoints, num_bins, num_bins))
    else: 
    # if True:
        dat_hic = np.load(file_loc, mmap_mode='r')     # (bs, time, num_bins, num_bins)
        dat_index = np.load(file_index_loc, mmap_mode='r') # (bs, time, 4) where 4 is [idx_sj, idx_ej, idx_sk, idx_ek]
        
        # How these indices are calculated in the original code:
        assert dat_hic.shape[1] == num_timepoints, f"Number of time points mismatch: {dat_hic.shape[1]} != {num_timepoints}"
        
        # Check if the hic matrix data are between 0 and 1
        if verbose:
            print(f"Min: {np.min(dat_hic)}, Max: {np.max(dat_hic)}", flush=True)
        
        # Normalize the HiC matrix
        if np.max(dat_hic) > 100:
            # clip to 100
            dat_hic = np.clip(dat_hic, 0, 100)
            dat_hic = np.divide(dat_hic, 100)
        
        hic_min = 0
        # normally the data is between 0 and 1, but if it is between -1 and 1, then normalize it to 0 and 1
        if np.min(dat_hic) < -0.5:
            # from -1 to 1, change to 0 to 1
            print(f"Normalizing HiC Matrix from [-1, 1] to [0, 1]", flush=True)
            dat_hic = np.clip(dat_hic, -1, 1)
            dat_hic = np.divide(dat_hic + 1, 2)
            hic_min = 0
            
        print(f"Min HiC Value: {hic_min}", flush=True)
        
        if save_type == "mmap":
            if os.path.exists(file_path):
                # File exists, open in read/write mode without truncation
                if verbose: 
                    print(f"Resuming from existing file: {file_path}", flush=True)
                hic_mm = np.memmap(file_path, dtype=np.float32, mode='r+', shape=(num_timepoints, num_bins, num_bins))
            else:
                # File does not exist, create new memmap file
                if verbose:
                    print(f"Creating new memmap file: {file_path}", flush=True)
                hic_mm = np.memmap(file_path, dtype=np.float32, mode='w+', shape=(num_timepoints, num_bins, num_bins))
                hic_mm[:, :, :] = hic_min
        else:
            hic = np.full((num_timepoints, num_bins, num_bins), hic_min, dtype=np.float32)  #hic for each time point. The complete HiC matrix is reconstructed from 50x50 bins hic.

        for t in range(dat_hic.shape[1]):   # go through time points
            print(f"Reconstructing HiC for Timepoint: {t+1}/{num_timepoints}", flush=True)
            # Why each bins could have multiple hic? Ans: because of overlapping
            mat_chr = np.full((num_bins, num_bins), hic_min, dtype=np.float32)  # sum of hic for each bin
            mat_n = np.zeros((num_bins, num_bins), dtype=np.float32)  # count number of hic for each bin

            for batch in (range(dat_hic.shape[0])):
                # Check if indices is of len 4 or 2
                if dat_index[batch, t].shape[0] == 4:
                    idx_sj, idx_ej, idx_sk, idx_ek = dat_index[batch, t] # Get the indices of the sub-matrix
                elif dat_index[batch, t].shape[0] == 2:
                    idx_sj, idx_sk = dat_index[batch, t]
                    idx_ej, idx_ek = idx_sj + sub_mat_n, idx_sk + sub_mat_n
                
                # Validate the calculation: So initially, 0 to 0+50 is the first sub-matrix. Since window size is 50, and broadcasting does not include the last element, so 0 to 49 is the first sub-matrix but we use 0 to 50 to include the last element.
                assert idx_ej - idx_sj == sub_mat_n, f"Row sub-matrix size mismatch: {idx_ej - idx_sj} != {sub_mat_n}" # We are adding 1 to include the last element because original code does not include the last element(inclusive like 0 to 49)
                assert idx_ek - idx_sk == sub_mat_n, f"Column sub-matrix size mismatch: {idx_ek - idx_sk} != {sub_mat_n}"
                
                # Check if the sub-matrix is going out of the chromosome
                if idx_ej > num_bins or idx_ek > num_bins:
                    assert False, f"Sub-matrix is going out of the chromosome: {idx_ej} >= {num_bins} or {idx_ek} >= {num_bins}"

                # Add the sub-matrix to the main matrix
                # check if the size of sub-matrix is 54x54, then the last 4 rows and columns are not added to the main matrix
                # sub_mat = dat_hic[batch, t]
                if dat_hic[batch, t].shape[0] > sub_mat_n:
                    mat_chr[idx_sj:idx_ej, idx_sk:idx_ek] += dat_hic[batch, t][:sub_mat_n, :sub_mat_n]
                else:
                    mat_chr[idx_sj:idx_ej, idx_sk:idx_ek] += dat_hic[batch, t]
                mat_n[idx_sj:idx_ej, idx_sk:idx_ek] += 1

            print(f"Normalizing HiC for Timepoint: {t+1}/{num_timepoints}", flush=True)
            # mat_chr2 = np.divide(mat_chr, mat_n, out=np.zeros_like(mat_chr), where=mat_n!=0)    # average of hic for each bin (sum/count) to account for overlapping
            mat_chr2 = np.divide(mat_chr, mat_n, out=np.full_like(mat_chr, hic_min), where=mat_n!=0)    # average of hic for each bin (sum/count) to account for overlapping
            if verbose:
                print(f"Min: {np.min(mat_chr2)}, Max: {np.max(mat_chr2)}, Shape: {mat_chr2.shape}", flush=True)
            
            # mat_chr and mat_n, relieve them from taking memory
            del mat_chr, mat_n
            gc.collect()
            
            # upper_tri = np.triu(mat_chr2)
            # mat_chr2 = upper_tri + upper_tri.T - np.diag(np.diag(upper_tri))
            # mat_chr2 = np.triu(mat_chr2) + np.triu(mat_chr2, 1).T
            
            print(f"Making HiC symmetric for Timepoint: {t+1}/{num_timepoints}", flush=True)
            print(f"Appending HiC for Timepoint: {t+1}/{num_timepoints}", flush=True)
            if save_type == "mmap":
                hic_mm[t] = np.triu(mat_chr2) + np.triu(mat_chr2, 1).T
            else:
                hic[t] = np.triu(mat_chr2) + np.triu(mat_chr2, 1).T
                
            del mat_chr2
            gc.collect()
        
        if save_type == "mmap":
            hic_mm.flush()  # Flush changes to disk
            del hic_mm  # Optionally, delete the memmap object to close the file
            hic = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_timepoints, num_bins, num_bins))
        else:
            # save the combined hic matrix
            if verbose: print(f"Saving combined HiC matrix to: {file_path}", flush=True)
            np.save(file_path, hic)
    
    if verbose:
        print(f"Hic len/Timepoints: {len(hic)}, Output HiC Shape: {hic[0].shape}", flush=True)
    
    return hic


def get_combined_hic_memory_efficient(file_loc, file_index_loc, num_timepoints, num_bins, sub_mat_n, dir_out, file_name, region_bin, verbose=False):

    print(f"\nCombining HiC Matrix from sub-matrices(windows) | Memory Efficient: {file_name}", flush=True)
    dat_hic = np.load(file_loc, mmap_mode='r')  # (bs, time, num_bins, num_bins)
    dat_index = np.load(file_index_loc, mmap_mode='r')  # (bs, time, 4) where 4 is [idx_sj, idx_ej, idx_sk, idx_ek]

    assert dat_hic.shape[1] == num_timepoints, f"Number of time points mismatch: {dat_hic.shape[1]} != {num_timepoints}"

    if verbose:
        print(f"Min: {np.min(dat_hic)}, Max: {np.max(dat_hic)}", flush=True)

    hic_min = 0
    if np.max(dat_hic) > 100:
        dat_hic = np.clip(dat_hic, 0, 100) / 100
    if np.min(dat_hic) < -0.5:
        print(f"Normalizing HiC Matrix from [-1, 1] to [0, 1]", flush=True)
        dat_hic = np.clip(dat_hic, -1, 1) / 2 + 0.5
        hic_min = 0
    print(f"Min: {np.min(dat_hic)}, Max: {np.max(dat_hic)}", flush=True)
    print(f"Min HiC Value: {hic_min}", flush=True)

    num_bins = region_bin[2] - region_bin[1]

    hic = np.full((num_timepoints, num_bins, num_bins), hic_min, dtype=np.float64)

    for t in range(dat_hic.shape[1]):
        print(f"Reconstructing HiC for Timepoint: {t+1}/{num_timepoints}", flush=True)
        mat_chr = np.full((num_bins, num_bins), hic_min, dtype=np.float64)
        mat_n = np.zeros((num_bins, num_bins), dtype=np.float64)

        for batch in (range(dat_hic.shape[0])):
            if dat_index[batch, t].shape[0] == 4:
                idx_sj, idx_ej, idx_sk, idx_ek = dat_index[batch, t]
            elif dat_index[batch, t].shape[0] == 2:
                idx_sj, idx_sk = dat_index[batch, t]
                idx_ej, idx_ek = idx_sj + sub_mat_n, idx_sk + sub_mat_n

            if (region_bin[1] <= idx_sj < region_bin[2] or region_bin[1] <= idx_ej <= region_bin[2]) and \
               (region_bin[1] <= idx_sk < region_bin[2] or region_bin[1] <= idx_ek <= region_bin[2]):
                assert idx_ej - idx_sj == sub_mat_n, f"Row sub-matrix size mismatch: {idx_ej - idx_sj} != {sub_mat_n}"
                assert idx_ek - idx_sk == sub_mat_n, f"Column sub-matrix size mismatch: {idx_ek - idx_sk} != {sub_mat_n}"

                sub_mat = dat_hic[batch, t]

                target_sj = max(region_bin[1], idx_sj)
                target_ej = min(region_bin[2], idx_ej)
                target_sk = max(region_bin[1], idx_sk)
                target_ek = min(region_bin[2], idx_ek)

                sub_sj = target_sj - idx_sj
                sub_ej = target_ej - idx_sj
                sub_sk = target_sk - idx_sk
                sub_ek = target_ek - idx_sk

                region_sj = target_sj - region_bin[1]
                region_ej = target_ej - region_bin[1]
                region_sk = target_sk - region_bin[1]
                region_ek = target_ek - region_bin[1]
        
                mat_chr[region_sj:region_ej, region_sk:region_ek] += sub_mat[sub_sj:sub_ej, sub_sk:sub_ek]
                mat_n[region_sj:region_ej, region_sk:region_ek] += 1

        mat_chr = np.divide(mat_chr, mat_n, where=mat_n > 0)
        mat_chr = np.triu(mat_chr) + np.triu(mat_chr, 1).T
        print(f"Min HiC Value: {np.min(mat_chr)}, Max HiC Value: {np.max(mat_chr)}", flush=True)
        hic[t] = mat_chr

    if verbose:
        print(f"Hic len/Timepoints: {len(hic)}, Output HiC Shape: {hic[0].shape}", flush=True)
    
    return hic
