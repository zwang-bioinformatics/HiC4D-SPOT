# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .plot_45_degree_heatmap import *

from .visualize.heatmap import *
from .visualize.bar import *

import scienceplots
plt.style.use('science')

def percentile_thresholding(matrix, percentile):
    """ Thresholds the matrix based on the given percentile """
    threshold = np.percentile(matrix, percentile)
    matrix[matrix < threshold] = 0
    return matrix

def show_plot2(hics_true, hics_true_perturbed, hics_pred, hics_ano_refed, rna_seq, region, region_bin, resolution, ids, dir_out, model_name, maxV, show_loops, trianglePlot=True, verbose=False, prefix=""):
    
    print(f"Plotting Hi-C matrices...")
    
    fig_name = f"{dir_out}/{prefix}{model_name}_showplot_triangle_{trianglePlot}_{region[1]}_{region[2]}.png"
    # if os.path.exists(fig_name):
    #     print(f"File {fig_name} already exists. Skipping plot.")
    #     return
    
    # rows = ['true', 'true_perturbed', 'naive', 'pred', 'loss', 'rnaseq', 'gene']   # 'loss',
    rows = ['true_perturbed', 'loss', 'rnaseq', 'gene']   # 'loss',
        
    nCols = len(ids)
    nRows = len(rows)
    fig = plt.figure(figsize=(nCols*8, nRows*3.3))
    # height_ratios = [3, 3, 3, 3, 0.3, 0.3]
    height_ratios = [3] * (nRows-2)
    height_ratios.append(0.3); height_ratios.append(0.3)
    gs = gridspec.GridSpec(figure=fig,ncols=nCols,nrows=nRows,
        height_ratios=height_ratios)
    
    #Draw subplots to fill in
    subPlots = []
    for i in range(nRows):
        row = []
        for j in range(nCols):
            trackax = fig.add_subplot(gs[i,j],frame_on=False)
            row.append(trackax)
        subPlots.append(row)
    
    
    sub_mat_true_all = []    
    sub_mat_true_perturbed_all = []
    sub_mat_temporal_diff_all = []
    sub_mat_pred_all = []
    sub_mat_diff_all = []
    sub_mat_naive_all = []
    sub_rna_seq_all = []
    sub_rna_seq_vec_all = []
    rnamax = 0
    
    for idx, timePoint in enumerate(ids):
        
        mat_chr_true2 = hics_true[idx]
        mat_chr_true_perturbed2 = hics_true_perturbed[idx]
        mat_chr_pred2 = hics_pred[idx]
        mat_chr_difference = np.abs(mat_chr_pred2 - mat_chr_true_perturbed2)
        # clip the difference to 0.5 and scale it to 0-1
        # mat_chr_difference = np.clip(mat_chr_difference, 0, 0.5) / 0.5
        
        # perfrom percentile thresholding to remove smaller noises. I only need significant changes
        # mat_chr_difference = percentile_thresholding(mat_chr_difference, 99)
        
        
        rna_seq_flag = False
        if len(rna_seq) != 0: rna_seq_flag = True
        if rna_seq_flag: rna_seq2 = rna_seq[idx]
        else: rna_seq2 = pd.DataFrame({'Chr': [], 'start': [], 'end': [], 'RPKM': []})
        
        # Get the difference between preceding timepoint in next timepoint. Like t2-t1, t3-t2, t4-t3, ... For the first timepoint, the difference is t1-t1
        if idx == 0:
            mat_chr_naive = np.zeros_like(mat_chr_true_perturbed2)
        else:
            mat_chr_naive = np.abs(hics_true_perturbed[idx-1] - hics_true_perturbed[idx])
        
        # Extracting desired range of submatrix for visualization and scaling it to 0-maxV from 0-1
        sub_mat_true = mat_chr_true2[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV
        sub_mat_true_perturbed = mat_chr_true_perturbed2[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV
        sub_mat_pred = mat_chr_pred2[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV
        sub_mat_diff = mat_chr_difference[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV  # Anomaly Loss
        # sub_mat_diff = percentile_thresholding(sub_mat_diff, 80)
        sub_mat_naive = mat_chr_naive[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV  # t{idx+1}-t{idx}
        sub_rna_seq = rna_seq2[
            (rna_seq2['Chr'] == f'{region[0]}') &
            ((rna_seq2['start'].between(region[1], region[2])) |
             (rna_seq2['end'].between(region[1], region[2])))
        ].copy()
        # convert from bp to bin
        sub_rna_seq['start'] = (sub_rna_seq['start'] - region[1]) // resolution
        sub_rna_seq['end'] = ((sub_rna_seq['end'] - region[1]) // resolution) + 1
        n = int((region[2] - region[1]) / resolution) 
        assert n == sub_mat_true.shape[0], f"n={n}, sub_mat_true.shape[0]={sub_mat_true.shape[0]}"
        sub_rna_seq_vec = np.zeros(n)
        for i, row in sub_rna_seq.iterrows():
            b = max(0,row['start'])
            e = min(n,row['end'])
            for i in range(b,e):
                sub_rna_seq_vec[i] += row['RPKM']
        rnamax = max(rnamax, np.amax(sub_rna_seq_vec))
        
        sub_mat_true_all.append(sub_mat_true)
        sub_mat_true_perturbed_all.append(sub_mat_true_perturbed)
        sub_mat_pred_all.append(sub_mat_pred)
        sub_mat_diff_all.append(sub_mat_diff)
        sub_mat_naive_all.append(sub_mat_naive)
        sub_rna_seq_all.append(sub_rna_seq)
        sub_rna_seq_vec_all.append(sub_rna_seq_vec)
        
    heatmaps = {
        'true': (sub_mat_true_all, maxV, "Original\nHi-C"),
        'true_perturbed': (sub_mat_true_perturbed_all, maxV, "Original\nHi-C"),
        'pred': (sub_mat_pred_all, maxV, "Reconstructed\nHi-C"),
        # 'loss': (sub_mat_diff_all, np.amax(sub_mat_diff_all)),
        # 'loss': (sub_mat_diff_all, maxV),
        'loss': (sub_mat_diff_all, 30, "Anomalies\nDetected"),
        
        'naive': (sub_mat_naive_all, maxV, "Naive\nHi-C"),
    }
    font_size = 24
    timepoints_labels = [f'T{idx+1}' for idx in range(len(ids))]

    # timepoints_labels = [
    #     'Untreated', '+ auxin, 6hr', 'withdraw, 20 min', 'withdraw, 40 min', ' withdraw, 60 min', 'withdraw, 180 min'
    # ]
    for idx, timePoint in enumerate(ids):
    
        # draw HiC matrices
        idx_row = 0
        for row in rows:
            if row in heatmaps.keys():
                ax = subPlots[idx_row][idx]
                drawRotatedHalfHeatmapUp(fig, ax, heatmaps[row][0][idx], heatmaps[row][1])
                # drawNonmalHeatmap(fig, ax, heatmaps[row][0][idx], heatmaps[row][1])
                ax.axvline(x=101,ymin=0,ymax=0.9,c='red',linestyle='--',alpha=0.5,linewidth=2,clip_on=False)
                n = sub_rna_seq_vec_all[idx].shape[0]
                # ax.xaxis.set_ticks(np.arange(0, n, n/4))
                if idx_row == 0:    ax.set_title(timepoints_labels[idx], fontsize=font_size)
                if idx == 0: ax.set_ylabel(f'{heatmaps[row][2]}', fontsize=font_size, rotation=0, labelpad=50) # Adds row labels True, Pred, etc. in the first column
                
                # if last row, add x axis labels
                if row == 'loss':
                    ax.set_xticklabels([int(x) for x in np.arange(region[1],region[2],(region[2]-region[1])/4)],ha='left')
                    ax.tick_params(which='both', direction='out',
                        right=False,left=False,top=False,
                        labelbottom=True,labelleft=False,
                        length=4)
                    ax.xaxis.set_tick_params(labelsize=font_size)
                    

                idx_row += 1

        ax = subPlots[idx_row][idx]
        drawGenes(ax, sub_rna_seq_all[idx], n)
        if idx == 0: ax.set_ylabel('Genes', fontsize=font_size, rotation=0, labelpad=50)
        
        ax = subPlots[idx_row+1][idx]
        drawVectorHeatmap(ax, sub_rna_seq_vec_all[idx], 'Purples', rnamax)
        ax.xaxis.set_tick_params(labelsize=font_size)
        
        if idx == 0: ax.set_ylabel('RNA-Seq', fontsize=font_size, rotation=0, labelpad=50)
        
        ax.set_xticklabels([int(x) for x in np.arange(region[1],region[2],(region[2]-region[1])/4)],ha='left')
        ax.tick_params(which='both', direction='out',
            right=False,left=False,top=False,
            labelbottom=True,labelleft=False,
            length=4)
        
    gs.tight_layout(fig)
    plt.savefig(fig_name, dpi=500, bbox_inches='tight')
    # plt.savefig(f"{dir_out}/{model_name}_showplot_triangle_{trianglePlot}_{region[1]}_{region[2]}.pdf", dpi=500, bbox_inches='tight')
    print(f"Saved plot: {fig_name}")


def show_plot_single_matrix(hic, region, region_bin, resolution, ids, dir_out, model_name, maxV, show_loops, trianglePlot=True, verbose=False, prefix=""):
    
    print(f"Plotting Hi-C matrix: show_plot_single_matrix", flush=True)
    
    fig_name = f"{dir_out}/{prefix}{model_name}_showplot_triangle_{trianglePlot}_{region[1]}_{region[2]}_combined.png"
    if os.path.exists(fig_name):
        print(f"File {fig_name} already exists. Skipping plot.")
        return
    
    rows = ["loss"]
    nCols = 1
    nRows = 1
    fig = plt.figure(figsize=(nCols*8, nRows*3.3))
    # height_ratios = [3, 3, 3, 3, 0.3, 0.3]
    height_ratios = [3] * (nRows)
    gs = gridspec.GridSpec(figure=fig,ncols=nCols,nrows=nRows,
        height_ratios=height_ratios)
    
    #Draw subplots to fill in
    subPlots = []
    for i in range(nRows):
        row = []
        for j in range(nCols):
            trackax = fig.add_subplot(gs[i,j],frame_on=False)
            row.append(trackax)
        subPlots.append(row)

    sub_mats = []
    for idx, timePoint in enumerate(ids):
        
        mat_chr = hic[idx]
        sub_mat = mat_chr[region_bin[1]:region_bin[2], region_bin[1]:region_bin[2]] * maxV
        sub_mats.append(sub_mat)
    
    heatmaps = {
        'loss': (sub_mats, np.amax(sub_mats)),
    }
    
    for idx in range(nRows):
    
        # draw HiC matrices
        idx_row = 0
        for row in rows:
            if row in heatmaps.keys():

                ax = subPlots[idx_row][idx]
                drawRotatedHalfHeatmapUp(fig, ax, heatmaps[row][0][idx], heatmaps[row][1])
                # drawNonmalHeatmap(fig, ax, heatmaps[row][0][idx], heatmaps[row][1])
                if idx_row == 0:    ax.set_title(f'Timepoints Combined', fontsize=18)  # Adds labels T1, T2, etc.
                if idx == 0: ax.set_ylabel(f'{row.capitalize()}', fontsize=18, rotation=0, labelpad=50) # Adds row labels True, Pred, etc. in the first column

                idx_row += 1

        ax.set_xticklabels([int(x) for x in np.arange(region[1],region[2],(region[2]-region[1])/4)],ha='left')
        ax.tick_params(which='both', direction='out',
            right=False,left=False,top=False,
            labelbottom=True,labelleft=False,
            length=4)
        
    gs.tight_layout(fig)
    plt.savefig(fig_name, dpi=500, bbox_inches='tight')
    print(f"Saved plot: {fig_name}")


def plot_matrix(ax, sub_matrix, start, resolution, maxV, cmap, trianglePlot=True):
    """ Plots the Hi-C data based on triangle or square format """
    if trianglePlot:
        pcolormesh_45deg(ax, sub_matrix, start=start, resolution=resolution, vmax=maxV, cmap=cmap)
    else:
        ax.imshow(sub_matrix, cmap=cmap, vmax=maxV)

