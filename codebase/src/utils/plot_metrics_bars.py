import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use('science')

def plot_metrics_bars(df_metrics, model_name, filter_zeros, dir_out, title):
    
    # Drop metrics that are not needed
    df_metrics = df_metrics.drop(columns=['PSNR', 'NRMSE'], errors='ignore')

    # Adjust metrics where higher is better
    df_metrics['SSI'] = 1 - df_metrics['SSI']  # Structural Similarity Index
    df_metrics['Cosine_Similarity'] = 1 - df_metrics['Cosine_Similarity']  # Cosine Similarity

    # Adjust metrics that range from -1 to 1
    df_metrics['PCC'] = (1 - df_metrics['PCC']) / 2  # Pearson Correlation Coefficient
    df_metrics['SCC'] = (1 - df_metrics['SCC']) / 2  # Spearman Correlation Coefficient

    # Rename columns for better LaTeX-style formatting with multi-line labels
    col = {
        "SSI": r"$1 - \text{SSI}$",
        "PCC": r"$\frac{1 - \text{PCC}}{2}$",
        "SCC": r"$\frac{1 - \text{SCC}}{2}$",
        "Cosine_Similarity": r"$1 - \text{Cosine}$" + "\n" + r"$\text{Similarity}$",
        "MSE": "MSE",
        "L1_Loss": "L1\nLoss",
        "KL_Divergence": "KL\nDivergence",
        "JS_Divergence": "JS\nDivergence"
    }
    df_metrics = df_metrics.rename(columns=col)

    # Set 'Timepoint' as the index and transpose for grouped bar chart
    df_metrics = df_metrics.set_index('Timepoint').T

    # Plot with increased figure width
    fig, ax = plt.subplots(figsize=(14, 7))  # Wider for better spacing

    # Define bar width and position
    num_timepoints = len(df_metrics.columns)
    num_metrics = len(df_metrics.index)
    bar_width = 0.8 / num_timepoints  # Ensure bars are well-spaced

    # Set positions for bars
    x = np.arange(num_metrics)  # X positions for each metric
    for i, column in enumerate(df_metrics.columns):
        ax.bar(x + i * bar_width, df_metrics[column], width=bar_width, label=f"Timepoint {column}")

    # Center xticks in the middle of each group
    ax.set_xticks(x + (num_timepoints - 1) * bar_width / 2)
    ax.set_xticklabels(df_metrics.index, ha="center", fontsize=14)

    # Set title and labels with LaTeX formatting
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel("Metric Values (Lower is better)", fontsize=16)
    ax.set_xlabel("Metrics", fontsize=16)

    # Improve legend
    ax.legend(title="Timepoints", loc='upper right', fontsize=14, title_fontsize=16)

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tight layout for better spacing
    plt.tight_layout()
    
    
    # save the figure
    # get image name from title
    image_name = title.replace(" ", "_").lower()
    fig.savefig(os.path.join(dir_out, f"{model_name}_filter{filter_zeros}_{image_name}.png"), bbox_inches='tight')
    print(f"Saved the figure: {model_name}_filter{filter_zeros}_{image_name}.png")
    
    