# Author: Bishal Shrestha
# Date: 03-24-2025  

import os

def runhicFindTADs(num_timepoints, resolution, dir_out, model_name):
    '''
    usage: hicFindTADs --matrix MATRIX --outPrefix OUTPREFIX
                   --correctForMultipleTesting {fdr,bonferroni,None}
                   [--minDepth INT bp] [--maxDepth INT bp] [--step INT bp]
                   [--TAD_sep_score_prefix TAD_SEP_SCORE_PREFIX]
                   [--thresholdComparisons THRESHOLDCOMPARISONS]
                   [--delta DELTA] [--minBoundaryDistance MINBOUNDARYDISTANCE]
                   [--chromosomes CHROMOSOMES [CHROMOSOMES ...]]
                   [--numberOfProcessors NUMBEROFPROCESSORS] [--help]
                   [--version]
    Required arguments
        --matrix, -m
        Corrected Hi-C matrix to use for the computations.

        --outPrefix
        File prefix to save the resulting files: 1. <prefix>_tad_separation.bm The format of the output file is chrom start end TAD-sep1 TAD-sep2 TAD-sep3 .. etc. We call this format a bedgraph matrix and can be plotted using hicPlotTADs. Each of the TAD-separation scores in the file corresponds to a different window length starting from –minDepth to –maxDepth. 2. <prefix>_zscore_matrix.h5, the z-score matrix used for the computation of the TAD-separation score. 3. < prefix > _boundaries.bed, which contains the positions of boundaries. The genomic coordinates in this file correspond to the resolution used. Thus, for Hi-C bins of 10.000bp the boundary position is 10.000bp long. For restriction fragment matrices the boundary position varies depending on the fragment length at the boundary. 4. <prefix>_domains.bed contains the TADs positions. This is a non-overlapping set of genomic positions. 5. <prefix>_boundaries.gff Similar to the boundaries bed file but with extra information (p-value, delta). 6. <prefix>_score.bedgraph file contains the TAD-separation score measured at each Hi-C bin coordinate. Is useful to visualize in a genome browser. The delta and p-value settings are saved as part of the name.

        --correctForMultipleTesting
        Possible choices: fdr, bonferroni, None
        Select the bonferroni or false discovery rate for a multiple comparison. Bonferroni controls the family-wise error rate (FWER) and needs a p-value. The false discovery rate (FDR) controls the likelyhood of type I errors and needs a q-value. As a third option it is possible to not use a multiple comparison method at all (Default: “fdr”).
        Default: “fdr”

    Optional arguments
        --minDepth
        Minimum window length (in bp) to be considered to the left and to the right of each Hi-C bin. This number should be at least 3 times as large as the bin size of the Hi-C matrix.

        --maxDepth
        Maximum window length to be considered to the left and to the right of the cut point in bp. This number should around 6-10 times as large as the bin size of the Hi-C matrix.

        --step
        Step size when moving from –minDepth to –maxDepth. Note, the step size grows exponentially as minDeph + (step * int(x)**1.5) for x in [0, 1, …] until it reaches maxDepth. For example, selecting step=10,000, minDepth=20,000 and maxDepth=150,000 will compute TAD-scores for window sizes: 20,000, 30,000, 40,000, 70,000 and 100,000

        --TAD_sep_score_prefix
        Sometimes it is useful to change some of the parameters without recomputing the z-score matrix and the TAD-separation score. For this case, the prefix containing the TAD separation score and the z-score matrix can be given. If this option is given, new boundaries will be computed but the values of –minDepth, –maxDepth and –step will not be used.

        --thresholdComparisons
        P-value threshold for the Bonferroni correction / q-value for FDR. The probability of a local minima to be a boundary is estimated by comparing the distribution (Wilcoxon ranksum) of the z-scores between the left and right regions (diamond) at the local minimum with the matrix z-scores for a diamond at –minDepth to the left and a diamond –minDepth to the right. If –correctForMultipleTesting is ‘None’ the threshold is applied on the raw p-values without any multiple testing correction. Set it to ‘1’ if no threshold should be used (Default: 0.01).
        Default: 0.01

        --delta
        Minimum threshold of the difference between the TAD-separation score of a putative boundary and the mean of the TAD-sep. score of surrounding bins. The delta value reduces spurious boundaries that are shallow, which usually occur at the center of large TADs when the TAD-sep. score is flat. Higher delta threshold values produce more conservative boundary estimations (Default: 0.01).
        Default: 0.01

        --minBoundaryDistance
        Minimum distance between boundaries (in bp). This parameter can be used to reduce spurious boundaries caused by noise.

        --chromosomes
        Chromosomes and order in which the chromosomes should be plotted. This option overrides –region and –region2.

        --numberOfProcessors, -p
        Number of processors to use (Default: 1).
        Default: 1

        --version
        show program's version number and exit
    '''
    print("Running hicFindTADs")
    dir_out
    for data_label in ["true", "true_perturbed", "pred", "anomaly"]:    # "anomaly_refed"
        for t in range(1, num_timepoints+1):
            
            matrix_file = f'{dir_out}/.h5/{model_name}_{data_label}_t{t}.h5'
            os.makedirs(f'{dir_out}/hicFindTADs', exist_ok=True)
            outprefix = f'{dir_out}/hicFindTADs/{model_name}_{data_label}_t{t}'
            
            # check if the outprefix_domains.bed file exists and is empty, if empty delete all the file with prefix outprefix
            # if os.path.exists(f"{outprefix}_domains.bed"):
            #     if os.stat(f"{outprefix}_domains.bed").st_size == 0:
            #         os.system(f"rm {outprefix}*")
            #     else:
            #         print(f"File already exists: {outprefix}_domains.bed")
            #         continue
            
            command = f'hicFindTADs --matrix {matrix_file} \
                --outPrefix {outprefix} \
                --correctForMultipleTesting fdr \
                --minDepth {5*resolution} \
                --maxDepth {10*resolution} \
                --step {resolution} \
                --thresholdComparisons 0.05 \
                --delta 0.01 \
                --minBoundaryDistance {resolution} \
                --numberOfProcessors 50'
            os.system(command)
            
            
# Run OnTAD to get TADs
def runOnTAD(num_timepoints, chr, chr_len, resolution, dir_out, output_filename):
    import subprocess
    # for data_label in ["true", "true_perturbed", "pred", "anomaly", "anomaly_refed"]:
    for data_label in ["true", "true_perturbed", "pred", "anomaly"]:
        for t in range(1, num_timepoints+1):
            matrix_file = f'{dir_out}/.matrix/{output_filename}_{data_label}_t{t}.matrix'
            os.makedirs(f'{dir_out}/ontad', exist_ok=True)
            output_file = f'{dir_out}/ontad/{output_filename}_{data_label}_t{t}_OnTAD'
            
            # Check if file with same prefix exists, if yes, continue
            # if os.path.exists(f"{output_file}.bed"):
            #     print(f"File already exists: {output_file}")
            #     continue

            OnTAD_path = '/home/bshrestha/tools/OnTAD/src/OnTAD'

            # OnTAD <Hi-C matrix> [-penalty <float>] [-maxsz <int>] [-minsz <int>] [-ldiff <float>] [-lsize <int>] [-bedout <chrnum int> <chrlength int> <int>] [-log2] [-o output_file] [-hic_norm <NONE/VC/VC_SQRT/KR>]
            # OnTAD chr18_KR.matrix -penalty 0.1 -maxsz 200 -o OnTAD_KRnorm_pen0.1_max200_chr18 -bedout chr18 78077248 10000
            # Output: startpos  endpos  TADlevel  TADmean  TADscore

            command = f"LD_LIBRARY_PATH=$HOME/local/gcc-9.3.0/lib64:$LD_LIBRARY_PATH {OnTAD_path} {matrix_file} -penalty 0.2 -minsz 3 -maxsz 500 -ldiff 1.96 -lsize 5 -o {output_file} -bedout {chr} {chr_len} {resolution}"

            subprocess.run(command, shell=True)