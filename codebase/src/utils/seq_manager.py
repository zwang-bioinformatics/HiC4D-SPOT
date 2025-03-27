# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import pyBigWig
import numpy as np


def insert_seq_file(mat, bw_loc, t, chrom, resolution, starti, endi, startj, endj, types = ["ATAC-Seq", "ChIP-Seq", "RNA-Seq"]):
    mat_new = np.zeros((mat.shape[0]+ len(types), mat.shape[1]+len(types)))
    mat_new[:mat.shape[0], :mat.shape[1]] = mat
    for idx1, type in enumerate(types):
        
        bw_file = os.path.join(bw_loc, type+"_normalized", f"{t}.bw")   # Getting the CPM normalized bigWig files: mitigates the effect of sequencing depth
        bw = pyBigWig.open(bw_file)
        
        # Get Mean and Standard Deviation of the whole chromosome
        chrom_len = bw.chroms()[chrom]
        mean = bw.stats(chrom, 0, chrom_len, type="mean", exact=True)[0]
        std = bw.stats(chrom, 0, chrom_len, type="std", exact=True)[0]
        # maxV = bw.stats(chrom, 0, chrom_len, type="max", exact=True)[0]  # 0.5, ? Clipping might remove the anomaly in the data. But for training, we need to pass the normal data distribution
        # minV = bw.stats(chrom, 0, chrom_len, type="min", exact=True)[0]  # 0
        maxV = 0.5
        minV = 0
        
        valuesi = np.zeros(mat.shape[0]+ len(types))
        idx = 0
        for i in range(starti, endi, resolution):
            signal = bw.stats(chrom, i, i+resolution, type="mean", exact=True)[0]
            if signal is None:
                signal = 0
            valuesi[idx] = signal
            idx += 1
            
        valuesj = np.zeros(mat.shape[1]+ len(types))
        idx = 0
        for j in range(startj, endj, resolution):
            signal = bw.stats(chrom, j, j+resolution, type="mean", exact=True)[0]
            if signal is None:
                signal = 0
            valuesj[idx] = signal
            idx += 1
        
        values = [valuesi, valuesj]
        # print(f"Values: {values}")
        
        for idx2, value in enumerate(values):
            values[idx2] = np.clip(values[idx2], minV, maxV)               # clip the values to minV and maxV
            values[idx2] = (values[idx2] - minV) / (maxV - minV)           # Perform min-max normalization -> 0 to 1, on the basis of the min and max values of the chromosome
            values[idx2] = values[idx2] * 10                            # Linearly scale to 0-10
            values[idx2] = np.log10(values[idx2] + 1) / np.log10(11)    # a log10 transformation is applied to scale all values to [0, 1]
            values[idx2] = 2 * values[idx2] - 1                         # all values are linearly scaled to [–1, 1].
        
        # Insert the values into the matrix at the bottom column and at the right column. This will increase the size of the matrix by 1. Insert 0 in the excess cell at the bottom right corner 
        mat_new[idx1-len(types),:] = values[1]  # Inserting the seq values of range in horizontal direction at the bottom row
        mat_new[:, idx1-len(types)] = values[0] # Inserting the seq values of range in vertical direction at the right column
        
    return mat_new


def test_dummy():
    mat = np.ones((10, 10))
    bw_loc = "/home/bshrestha/HiC4D-SPOT/data/data_Reed"
    t = "t1"
    chrom = "chr1"
    resolution = 10000
    starti = 100000
    endi = 200000
    startj = 100000
    endj = 200000
    mat_new = insert_seq_file(mat, bw_loc, t, chrom, resolution, starti, endi, startj, endj)
    print(f"Before: {mat.shape}, After: {mat_new.shape}")
    # print(mat_new)

if __name__ == "__main__":
    test_dummy()