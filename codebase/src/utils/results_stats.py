# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np
from scipy.stats import spearmanr, pearsonr, entropy
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from pyemd import emd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scienceplots
from .plot_metrics_bars import plot_metrics_bars

def mask_zeros(hic1, hic2):
    """
    Mask out zero entries in both matrices to ensure only valid values contribute to metrics.
    
    Returns:
        Filtered 1D arrays of valid (non-zero) values from both matrices.
    """
    # not zero
    mask = (hic1 != 0) & (hic2 != 0)    # mask out zero entries    
    return hic1[mask], hic2[mask]

def calculate_mse(hic1, hic2, filter_zeros=True):
    """
    Calculate Mean Squared Error (MSE) between two Hi-C matrices.
    
    - **Purpose**: Measures the squared difference between predicted and true values.
    - **Range**: [0, ∞) (lower is better).
    - **Ideal Value**: 0 (perfect prediction).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    return np.mean((filtered_hic1 - filtered_hic2) ** 2)

def calculate_l1_loss(hic1, hic2, filter_zeros=True):
    """
    Calculate Mean Absolute Error (L1 Loss) between two Hi-C matrices.
    
    - **Purpose**: Measures absolute differences between predicted and true values.
    - **Range**: [0, ∞) (lower is better).
    - **Ideal Value**: 0 (perfect prediction).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    return np.mean(np.abs(filtered_hic1 - filtered_hic2))

def calculate_ssi(hic1, hic2):
    """
    Calculate Structural Similarity Index (SSI) between two Hi-C matrices.
    
    - **Purpose**: Measures structural similarity in terms of contrast, luminance, and structure.
    - **Range**: [-1, 1] (higher is better).
    - **Ideal Value**: 1 (identical structure).
    """
    return ssim(hic1, hic2, data_range=hic1.max() - hic1.min())

def calculate_pcc(hic1, hic2, filter_zeros=True):
    """
    Calculate Pearson Correlation Coefficient (PCC) between two Hi-C matrices.
    
    - **Purpose**: Measures linear correlation between two matrices.
    - **Range**: [-1, 1] (higher is better).
    - **Ideal Value**: 1 (perfect correlation).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    return pearsonr(filtered_hic1, filtered_hic2)[0]

def calculate_scc(hic1, hic2, filter_zeros=True):
    """
    Calculate Spearman Correlation Coefficient (SCC) between two Hi-C matrices.
    
    - **Purpose**: Measures monotonic correlation (rank-based similarity).
    - **Range**: [-1, 1] (higher is better).
    - **Ideal Value**: 1 (perfect rank correlation).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    return spearmanr(filtered_hic1, filtered_hic2)[0]

def calculate_psnr(hic1, hic2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two Hi-C matrices.
    
    - **Purpose**: Measures the ratio of signal strength to noise.
    - **Range**: [0, ∞) dB (higher is better).
    - **Ideal Value**: >30 dB (low noise, good prediction quality).
    """
    return psnr(hic1, hic2, data_range=hic1.max() - hic1.min())

def calculate_nrmse(hic1, hic2):
    """
    Calculate Normalized Root Mean Squared Error (NRMSE).
    
    - **Purpose**: Measures RMSE relative to the data range for better comparability.
    - **Range**: [0, ∞) (lower is better).
    - **Ideal Value**: 0 (perfect reconstruction).
    """
    mse = calculate_mse(hic1, hic2)
    return np.sqrt(mse) / (hic1.max() - hic1.min())


def calculate_kl_divergence(hic1, hic2, filter_zeros=True, epsilon=1e-10):
    """
    Calculate Kullback-Leibler (KL) Divergence.
    
    - **Purpose**: Measures how different the predicted distribution is from the true one.
    - **Range**: [0, ∞) (lower is better).
    - **Ideal Value**: 0 (identical distributions).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())

    p = (filtered_hic1 + epsilon) / (filtered_hic1.sum() + epsilon)
    q = (filtered_hic2 + epsilon) / (filtered_hic2.sum() + epsilon)

    return entropy(p, q)


def calculate_js_divergence(hic1, hic2, filter_zeros=True):
    """
    Calculate Jensen-Shannon (JS) Divergence.
    
    - **Purpose**: Measures symmetric divergence between two distributions.
    - **Range**: [0, 1] (lower is better).
    - **Ideal Value**: 0 (identical distributions).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    m = 0.5 * (filtered_hic1 + filtered_hic2)
    return 0.5 * (entropy(filtered_hic1, m) + entropy(filtered_hic2, m))

def calculate_cosine_similarity(hic1, hic2, filter_zeros=True):
    """
    Calculate Cosine Similarity between two Hi-C matrices.
    
    - **Purpose**: Measures similarity in vector space.
    - **Range**: [-1, 1] (higher is better).
    - **Ideal Value**: 1 (identical vectors).
    """
    filtered_hic1, filtered_hic2 = mask_zeros(hic1, hic2) if filter_zeros else (hic1.flatten(), hic2.flatten())
    return 1 - cosine(filtered_hic1, filtered_hic2)

def calculate_frobenius_norm_diff(hic1, hic2):
    """
    Calculate Frobenius Norm Difference.
    
    - **Purpose**: Measures overall matrix structure difference.
    - **Range**: [0, ∞) (lower is better).
    - **Ideal Value**: 0 (identical matrices).
    """
    return norm(hic1 - hic2, 'fro')


def compute_metrics(hics_a, hics_b, comparison_name, filter_zeros=True, verbose=True):
    """
    Compute all evaluation metrics between two lists of Hi-C matrices.
    
    Includes:
    - Traditional reconstruction metrics (MSE, L1, SSI, PCC, SCC, PSNR, NRMSE)
    - Probability-based metrics (KL Divergence, JS Divergence)
    - Structural similarity metrics (Cosine Similarity, Frobenius Norm, EMD)
    
    Returns:
        A DataFrame containing per-timepoint and average values for all metrics.
    """
    num_timepoints = len(hics_a)
    
    metrics = {
        "Timepoint": list(range(1, num_timepoints + 1)),
        "MSE": [],
        "L1_Loss": [],
        "SSI": [],
        "PCC": [],
        "SCC": [],
        "PSNR": [],
        "NRMSE": [],
        "KL_Divergence": [],
        "JS_Divergence": [],
        "Cosine_Similarity": [],
        # "Frobenius_Norm": [],
    }

    for i in range(num_timepoints):
        metrics["MSE"].append(calculate_mse(hics_a[i], hics_b[i], filter_zeros))
        metrics["L1_Loss"].append(calculate_l1_loss(hics_a[i], hics_b[i], filter_zeros))
        metrics["SSI"].append(calculate_ssi(hics_a[i], hics_b[i]))
        metrics["PCC"].append(calculate_pcc(hics_a[i], hics_b[i], filter_zeros))
        metrics["SCC"].append(calculate_scc(hics_a[i], hics_b[i], filter_zeros))
        metrics["PSNR"].append(calculate_psnr(hics_a[i], hics_b[i]))
        metrics["NRMSE"].append(calculate_nrmse(hics_a[i], hics_b[i]))
        metrics["KL_Divergence"].append(calculate_kl_divergence(hics_a[i], hics_b[i], filter_zeros))
        metrics["JS_Divergence"].append(calculate_js_divergence(hics_a[i], hics_b[i], filter_zeros))
        metrics["Cosine_Similarity"].append(calculate_cosine_similarity(hics_a[i], hics_b[i], filter_zeros))


    # Compute average metrics
    avg_metrics = {key: np.mean(val) for key, val in metrics.items() if key != "Timepoint"}
    avg_metrics["Timepoint"] = "Average"

    if verbose:
        print(f"\n{comparison_name} Metrics:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value}")

    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    df = pd.concat([df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    # Till third decimal point
    df = df.round(3)
    
    return df

def generate_stats(hics_true, hics_true_perturbed, hics_pred, hics_ano, hics_ano_refed, hic_ano_combined_array, dir_out, model_name, filter_zeros=True, verbose=True):
    """
    Generate statistics for Hi-C matrix evaluation and export results to CSV.

    Comparisons:
    - **True vs Predicted**: Measures reconstruction accuracy.
    - **True vs Perturbed**: Measures the effect of perturbation.
    - **Perturbed vs Predicted**: Measures how well perturbations are corrected.

    Outputs:
    - Saves CSV files in `dir_out` containing all per-timepoint metrics.

    Args:
        hics_true: List of ground truth Hi-C matrices.
        hics_true_perturbed: List of perturbed Hi-C matrices.
        hics_pred: List of predicted Hi-C matrices.
        dir_out: Output directory for CSV files.
        model_name: Name used for file naming.
        verbose: Whether to print results.
    """
    print(f"Generating statistics for {model_name}...")
    dir_out = os.path.join(dir_out, "metrics")
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
        
    # Compute all metrics
    df_true_perturbed = compute_metrics(hics_true, hics_true_perturbed, "True vs Perturbed", filter_zeros, verbose)
    df_perturbed_pred = compute_metrics(hics_true_perturbed, hics_pred, "Perturbed vs Predicted", filter_zeros, verbose)

    # Save results to CSV
    df_true_perturbed.to_csv(os.path.join(dir_out, f"{model_name}_filter{filter_zeros}_true_vs_perturbed.csv"), index=False)
    df_perturbed_pred.to_csv(os.path.join(dir_out, f"{model_name}_filter{filter_zeros}_perturbed_vs_pred.csv"), index=False)
    
    # plot the metrics
    plot_metrics_bars(df_true_perturbed, model_name, filter_zeros, dir_out, "Original vs Time-swapped")
    plot_metrics_bars(df_perturbed_pred, model_name, filter_zeros, dir_out, "Time-swapped vs Reconstructed")
    
    print(f"\n✅ Metrics saved in {dir_out}")
