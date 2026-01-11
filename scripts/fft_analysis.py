#!/usr/bin/env python3
"""
FFT spectral analysis of Goldbach deviations.

Identifies spectral peaks corresponding to L-function zeros.

Usage:
    python fft_analysis.py --input data/goldbach_deviations.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Known L-function zero imaginary parts
L_FUNCTION_ZEROS = {
    'L(s, χ_4)': 6.021,
    'L(s, χ_3)': 8.039,
    'ζ(s)': 14.135,
}


def compute_power_spectrum(data, sampling_rate=1.0):
    """
    Compute power spectrum using FFT.
    
    Parameters
    ----------
    data : array-like
        Time series data.
    sampling_rate : float
        Sampling rate (default: 1.0).
    
    Returns
    -------
    tuple
        (frequencies, power_spectrum)
    """
    n = len(data)
    
    # Remove mean
    data = data - np.mean(data)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(n)
    data_windowed = data * window
    
    # Compute FFT
    fft_result = np.fft.fft(data_windowed)
    
    # Power spectrum (one-sided)
    power = np.abs(fft_result[:n//2])**2 / n
    
    # Frequency axis
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)[:n//2]
    
    return freqs, power


def find_spectral_peaks(freqs, power, height_percentile=90, distance=10):
    """
    Find significant peaks in power spectrum.
    
    Parameters
    ----------
    freqs : array-like
        Frequency values.
    power : array-like
        Power spectrum values.
    height_percentile : float
        Minimum height as percentile of power values.
    distance : int
        Minimum distance between peaks.
    
    Returns
    -------
    tuple
        (peak_frequencies, peak_powers, peak_indices)
    """
    height_threshold = np.percentile(power, height_percentile)
    
    peak_indices, properties = find_peaks(
        power, 
        height=height_threshold,
        distance=distance
    )
    
    peak_freqs = freqs[peak_indices]
    peak_powers = power[peak_indices]
    
    # Sort by power (descending)
    sort_idx = np.argsort(peak_powers)[::-1]
    
    return peak_freqs[sort_idx], peak_powers[sort_idx], peak_indices[sort_idx]


def match_zeros(peak_freqs, known_zeros, tolerance=0.5):
    """
    Match observed peaks to known L-function zeros.
    
    Parameters
    ----------
    peak_freqs : array-like
        Observed peak frequencies (as γ values).
    known_zeros : dict
        Dictionary of {label: gamma_value}.
    tolerance : float
        Maximum allowed deviation for matching.
    
    Returns
    -------
    list
        List of (peak_freq, matched_label, known_gamma, deviation) tuples.
    """
    matches = []
    
    for peak_f in peak_freqs:
        # Convert frequency to gamma (assuming freq = gamma / (2*pi))
        peak_gamma = peak_f * 2 * np.pi
        
        best_match = None
        best_dev = float('inf')
        
        for label, gamma in known_zeros.items():
            dev = abs(peak_gamma - gamma)
            if dev < best_dev and dev < tolerance:
                best_dev = dev
                best_match = (label, gamma)
        
        if best_match:
            matches.append((peak_gamma, best_match[0], best_match[1], best_dev))
        else:
            matches.append((peak_gamma, None, None, None))
    
    return matches


def plot_spectrum(freqs, power, peak_freqs, peak_powers, output_file=None):
    """
    Plot power spectrum with identified peaks.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Full spectrum
    ax1.semilogy(freqs * 2 * np.pi, power, 'b-', alpha=0.7, linewidth=0.5)
    ax1.scatter(peak_freqs * 2 * np.pi, peak_powers, color='red', s=50, zorder=5)
    
    # Mark known zeros
    for label, gamma in L_FUNCTION_ZEROS.items():
        ax1.axvline(gamma, color='green', linestyle='--', alpha=0.7, label=f'{label}: γ={gamma}')
    
    ax1.set_xlabel('γ (imaginary part of zero)', fontsize=12)
    ax1.set_ylabel('Power (log scale)', fontsize=12)
    ax1.set_title('FFT Power Spectrum of Goldbach Deviations', fontsize=14)
    ax1.set_xlim(0, 30)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view of low frequencies
    mask = freqs * 2 * np.pi < 20
    ax2.plot(freqs[mask] * 2 * np.pi, power[mask], 'b-', linewidth=1)
    
    peak_mask = peak_freqs * 2 * np.pi < 20
    ax2.scatter(peak_freqs[peak_mask] * 2 * np.pi, peak_powers[peak_mask], 
                color='red', s=100, zorder=5, label='Detected peaks')
    
    for label, gamma in L_FUNCTION_ZEROS.items():
        if gamma < 20:
            ax2.axvline(gamma, color='green', linestyle='--', alpha=0.7, linewidth=2)
            ax2.annotate(f'{label}\nγ={gamma}', xy=(gamma, ax2.get_ylim()[1]*0.8),
                        fontsize=9, ha='center')
    
    ax2.set_xlabel('γ (imaginary part of zero)', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('Zoomed View: γ < 20', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='FFT spectral analysis of Goldbach deviations'
    )
    parser.add_argument(
        '--input', type=str, default='data/goldbach_deviations.csv',
        help='Input CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='figures/fft_spectrum.png',
        help='Output figure file'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Use ln(N) as the "time" variable for spectral analysis
    # This linearizes the oscillation γ*ln(N)
    ln_N = np.log(df['N'].values)
    delta = df['delta_N'].values
    
    # Interpolate to uniform spacing in ln(N)
    ln_N_uniform = np.linspace(ln_N.min(), ln_N.max(), len(ln_N))
    delta_interp = np.interp(ln_N_uniform, ln_N, delta)
    
    # Sampling rate in ln(N) space
    d_ln_N = ln_N_uniform[1] - ln_N_uniform[0]
    sampling_rate = 1 / d_ln_N
    
    print(f"Data points: {len(delta)}")
    print(f"ln(N) range: {ln_N.min():.2f} to {ln_N.max():.2f}")
    
    # Compute power spectrum
    print("\nComputing FFT power spectrum...")
    freqs, power = compute_power_spectrum(delta_interp, sampling_rate)
    
    # Find peaks
    print("Identifying spectral peaks...")
    peak_freqs, peak_powers, _ = find_spectral_peaks(freqs, power)
    
    # Match to known zeros
    matches = match_zeros(peak_freqs[:10], L_FUNCTION_ZEROS)
    
    # Print results
    print("\n" + "="*60)
    print("FFT Spectral Analysis Results")
    print("="*60)
    print("\nTop spectral peaks (as γ values):")
    print("-"*60)
    print(f"{'Observed γ':>12} | {'Matched Zero':>15} | {'Known γ':>10} | {'Deviation':>10}")
    print("-"*60)
    
    for peak_gamma, label, known_gamma, dev in matches[:10]:
        if label:
            print(f"{peak_gamma:12.3f} | {label:>15} | {known_gamma:10.3f} | {dev:10.3f}")
        else:
            print(f"{peak_gamma:12.3f} | {'(no match)':>15} | {'-':>10} | {'-':>10}")
    
    print("\n" + "="*60)
    print("Known L-function zeros for reference:")
    for label, gamma in L_FUNCTION_ZEROS.items():
        print(f"  {label}: γ = {gamma}")
    
    # Plot
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_spectrum(freqs, power, peak_freqs, peak_powers, args.output)


if __name__ == "__main__":
    main()
