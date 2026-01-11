#!/usr/bin/env python3
"""
Compute Hurst exponent of Goldbach deviations using R/S analysis.

The Hurst exponent H characterizes long-range dependence:
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trending)
- H < 0.5: Anti-persistent (mean-reverting)

Usage:
    python compute_hurst.py --input data/goldbach_deviations.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def rescaled_range(x):
    """
    Compute the rescaled range R/S for a time series.
    
    Parameters
    ----------
    x : array-like
        Time series data.
    
    Returns
    -------
    float
        R/S statistic.
    """
    n = len(x)
    if n < 2:
        return np.nan
    
    # Mean-adjusted series
    mean_x = np.mean(x)
    y = x - mean_x
    
    # Cumulative deviation
    z = np.cumsum(y)
    
    # Range
    R = np.max(z) - np.min(z)
    
    # Standard deviation
    S = np.std(x, ddof=1)
    
    if S == 0:
        return np.nan
    
    return R / S


def compute_hurst_rs(data, min_window=10, max_window=None, num_points=50):
    """
    Compute Hurst exponent using R/S analysis.
    
    Parameters
    ----------
    data : array-like
        Time series data.
    min_window : int
        Minimum window size.
    max_window : int, optional
        Maximum window size (default: len(data)//4).
    num_points : int
        Number of window sizes to test.
    
    Returns
    -------
    dict
        Dictionary with 'H' (Hurst exponent), 'H_std' (standard error),
        'window_sizes', 'rs_values', and regression statistics.
    """
    data = np.asarray(data)
    n = len(data)
    
    if max_window is None:
        max_window = n // 4
    
    # Generate logarithmically spaced window sizes
    window_sizes = np.unique(np.logspace(
        np.log10(min_window), 
        np.log10(max_window), 
        num_points
    ).astype(int))
    
    rs_values = []
    
    for w in window_sizes:
        # Number of non-overlapping windows
        num_windows = n // w
        if num_windows < 1:
            continue
        
        rs_list = []
        for i in range(num_windows):
            segment = data[i*w:(i+1)*w]
            rs = rescaled_range(segment)
            if not np.isnan(rs):
                rs_list.append(rs)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(rs_values)
    window_sizes = window_sizes[valid_mask]
    rs_values = np.array(rs_values)[valid_mask]
    
    # Linear regression in log-log space
    log_w = np.log(window_sizes)
    log_rs = np.log(rs_values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_w, log_rs)
    
    return {
        'H': slope,
        'H_std': std_err,
        'r_squared': r_value**2,
        'p_value': p_value,
        'window_sizes': window_sizes,
        'rs_values': rs_values,
        'intercept': intercept
    }


def plot_hurst_analysis(result, output_file=None):
    """
    Plot R/S analysis results.
    
    Parameters
    ----------
    result : dict
        Output from compute_hurst_rs().
    output_file : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points
    ax.scatter(
        result['window_sizes'], 
        result['rs_values'],
        alpha=0.7, 
        label='R/S values'
    )
    
    # Fitted line
    w_fit = np.linspace(
        result['window_sizes'].min(), 
        result['window_sizes'].max(), 
        100
    )
    rs_fit = np.exp(result['intercept']) * w_fit**result['H']
    ax.plot(w_fit, rs_fit, 'r-', linewidth=2, 
            label=f'H = {result["H"]:.3f} ± {result["H_std"]:.3f}')
    
    # Reference lines
    ax.plot(w_fit, w_fit**0.5, 'k--', alpha=0.5, label='H = 0.5 (random walk)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Window size n', fontsize=12)
    ax.set_ylabel('R/S statistic', fontsize=12)
    ax.set_title('Hurst Exponent Analysis of Goldbach Deviations', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'H = {result["H"]:.4f} ± {result["H_std"]:.4f}\n'
    textstr += f'R² = {result["r_squared"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compute Hurst exponent of Goldbach deviations'
    )
    parser.add_argument(
        '--input', type=str, default='data/goldbach_deviations.csv',
        help='Input CSV file with columns N, delta_N'
    )
    parser.add_argument(
        '--output', type=str, default='figures/hurst_analysis.png',
        help='Output figure file'
    )
    parser.add_argument(
        '--min-window', type=int, default=10,
        help='Minimum window size (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    delta = df['delta_N'].values
    
    print(f"Loaded {len(delta)} data points")
    
    # Compute Hurst exponent
    print("\nComputing Hurst exponent using R/S analysis...")
    result = compute_hurst_rs(delta, min_window=args.min_window)
    
    # Print results
    print("\n" + "="*50)
    print("Hurst Exponent Analysis Results")
    print("="*50)
    print(f"Hurst exponent H = {result['H']:.4f} ± {result['H_std']:.4f}")
    print(f"R² = {result['r_squared']:.4f}")
    print(f"p-value = {result['p_value']:.2e}")
    print("\nInterpretation:")
    if result['H'] > 0.5:
        print(f"  H = {result['H']:.2f} > 0.5 indicates PERSISTENT behavior")
        print("  (long-range positive correlations, trending)")
    elif result['H'] < 0.5:
        print(f"  H = {result['H']:.2f} < 0.5 indicates ANTI-PERSISTENT behavior")
        print("  (mean-reverting)")
    else:
        print("  H ≈ 0.5 indicates random walk behavior")
    
    print(f"\nTheoretical prediction: H ≈ 0.84")
    print(f"Observed value:         H = {result['H']:.4f}")
    
    # Plot results
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_hurst_analysis(result, args.output)


if __name__ == "__main__":
    main()
