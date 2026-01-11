#!/usr/bin/env python3
"""
Fit and validate the 1/24 formula for Dirichlet character coefficients.

The formula: c_p = (1/24) * L(1, χ_p) / (p - 2)

Usage:
    python coefficient_fitting.py --input data/goldbach_deviations.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from utils import (
    dirichlet_character_mod3,
    dirichlet_character_mod5,
    L_value_chi3,
    L_value_chi5,
    L_value_chi7,
    theoretical_coefficient,
    KAPPA_LOCAL,
    C2
)


# Theoretical coefficients using 1/24 formula
THEORETICAL_COEFFICIENTS = {
    3: theoretical_coefficient(3, L_value_chi3()),
    5: theoretical_coefficient(5, L_value_chi5()),
    7: theoretical_coefficient(7, L_value_chi7()),
}

# L-values for reference
L_VALUES = {
    3: L_value_chi3(),
    5: L_value_chi5(),
    7: L_value_chi7(),
}


def fit_character_coefficient(N_values, delta_values, p, character_func):
    """
    Fit coefficient c_p by regression on character values.
    
    Model: δ(N) ≈ c_p * χ_p(N) + noise
    
    Parameters
    ----------
    N_values : array-like
        Even integers.
    delta_values : array-like
        Deviation values δ(N).
    p : int
        Prime modulus for character.
    character_func : callable
        Dirichlet character function.
    
    Returns
    -------
    dict
        Fitted coefficient and statistics.
    """
    # Compute character values
    chi_values = np.array([character_func(N) for N in N_values])
    
    # Filter out N where χ(N) = 0
    mask = chi_values != 0
    chi_filtered = chi_values[mask]
    delta_filtered = delta_values[mask]
    
    # Simple regression: δ = c * χ + ε
    # Estimate c by correlation
    correlation = np.corrcoef(chi_filtered, delta_filtered)[0, 1]
    
    # Fit by least squares: c = Σ(χ*δ) / Σ(χ²)
    c_fitted = np.sum(chi_filtered * delta_filtered) / np.sum(chi_filtered**2)
    
    # Standard error
    residuals = delta_filtered - c_fitted * chi_filtered
    n = len(delta_filtered)
    se = np.sqrt(np.sum(residuals**2) / (n - 1)) / np.sqrt(np.sum(chi_filtered**2))
    
    return {
        'p': p,
        'c_fitted': c_fitted,
        'c_std': se,
        'correlation': correlation,
        'n_points': n,
        'c_theoretical': THEORETICAL_COEFFICIENTS.get(p, None),
        'L_value': L_VALUES.get(p, None)
    }


def fit_global_kappa(N_values, delta_values):
    """
    Fit the global coupling constant κ from envelope.
    
    Model: |δ(N)|_max ≈ κ / ln(N)
    
    Returns
    -------
    dict
        Fitted κ and comparison with C_2.
    """
    ln_N = np.log(N_values)
    
    # Compute running maximum envelope
    window_size = max(len(delta_values) // 100, 100)
    envelope = []
    ln_N_env = []
    
    for i in range(0, len(delta_values) - window_size, window_size // 2):
        window = delta_values[i:i + window_size]
        envelope.append(np.max(np.abs(window)))
        ln_N_env.append(np.mean(ln_N[i:i + window_size]))
    
    envelope = np.array(envelope)
    ln_N_env = np.array(ln_N_env)
    
    # Fit: envelope = κ / ln(N)
    # Transform: envelope * ln(N) = κ
    kappa_estimates = envelope * ln_N_env
    kappa_fitted = np.mean(kappa_estimates)
    kappa_std = np.std(kappa_estimates)
    
    return {
        'kappa_fitted': kappa_fitted,
        'kappa_std': kappa_std,
        'C2': C2,
        'ratio': kappa_fitted / C2,
        'ln_N_env': ln_N_env,
        'envelope': envelope
    }


def create_comparison_table(results):
    """Create formatted comparison table."""
    print("\n" + "="*70)
    print("Coefficient Comparison: Theoretical (1/24 formula) vs Fitted")
    print("="*70)
    print(f"{'Prime p':>8} | {'L(1,χ_p)':>10} | {'1/24(p-2)':>10} | "
          f"{'Theory c_p':>12} | {'Fitted c_p':>12} | {'Error %':>8}")
    print("-"*70)
    
    for r in results:
        p = r['p']
        L_val = r['L_value'] if r['L_value'] else 0
        geom = 1/(24*(p-2)) if p > 2 else 0
        c_th = r['c_theoretical'] if r['c_theoretical'] else 0
        c_fit = r['c_fitted']
        
        if c_th != 0:
            error = abs(c_fit - c_th) / abs(c_th) * 100
            print(f"{p:>8} | {L_val:>10.4f} | {geom:>10.4f} | "
                  f"{c_th:>12.4f} | {c_fit:>12.4f} | {error:>7.1f}%")
        else:
            print(f"{p:>8} | {L_val:>10.4f} | {geom:>10.4f} | "
                  f"{'N/A':>12} | {c_fit:>12.4f} | {'N/A':>8}")
    
    print("-"*70)
    print(f"\nUniversal coupling constant: κ_local = 1/24 = {KAPPA_LOCAL:.6f}")


def plot_coefficient_analysis(results, envelope_result, output_file=None):
    """Plot coefficient analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Coefficient comparison
    ax1 = axes[0]
    primes = [r['p'] for r in results]
    c_theory = [r['c_theoretical'] for r in results]
    c_fitted = [r['c_fitted'] for r in results]
    c_std = [r['c_std'] for r in results]
    
    x = np.arange(len(primes))
    width = 0.35
    
    ax1.bar(x - width/2, c_theory, width, label='Theory: (1/24)·L(1,χ)/(p-2)', 
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, c_fitted, width, label='Fitted', 
            color='coral', alpha=0.8, yerr=c_std, capsize=5)
    
    ax1.set_xlabel('Prime p', fontsize=12)
    ax1.set_ylabel('Coefficient c_p', fontsize=12)
    ax1.set_title('Validation of 1/24 Formula', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'p={p}' for p in primes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right panel: Envelope scaling
    ax2 = axes[1]
    ax2.scatter(envelope_result['ln_N_env'], envelope_result['envelope'], 
                alpha=0.6, label='Observed envelope')
    
    # Theoretical curve: κ/ln(N)
    ln_N_fit = np.linspace(min(envelope_result['ln_N_env']), 
                           max(envelope_result['ln_N_env']), 100)
    ax2.plot(ln_N_fit, C2/ln_N_fit, 'r-', linewidth=2, 
             label=f'$C_2$/ln(N), $C_2$={C2:.4f}')
    ax2.plot(ln_N_fit, envelope_result['kappa_fitted']/ln_N_fit, 'g--', linewidth=2,
             label=f'Fitted κ/ln(N), κ={envelope_result["kappa_fitted"]:.4f}')
    
    ax2.set_xlabel('ln(N)', fontsize=12)
    ax2.set_ylabel('|δ(N)|_max (envelope)', fontsize=12)
    ax2.set_title(f'Global Envelope: κ/C₂ = {envelope_result["ratio"]:.3f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Fit and validate 1/24 formula'
    )
    parser.add_argument(
        '--input', type=str, default='data/goldbach_deviations.csv',
        help='Input CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='figures/coefficient_analysis.png',
        help='Output figure file'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    N_values = df['N'].values
    delta_values = df['delta_N'].values
    
    print(f"Loaded {len(df)} data points")
    
    # Fit coefficients for each prime
    print("\nFitting Dirichlet character coefficients...")
    
    results = []
    
    # Mod 3
    r3 = fit_character_coefficient(N_values, delta_values, 3, dirichlet_character_mod3)
    results.append(r3)
    
    # Mod 5
    r5 = fit_character_coefficient(N_values, delta_values, 5, dirichlet_character_mod5)
    results.append(r5)
    
    # Print comparison table
    create_comparison_table(results)
    
    # Fit global envelope
    print("\nFitting global envelope...")
    envelope_result = fit_global_kappa(N_values, delta_values)
    
    print("\n" + "="*50)
    print("Global Envelope Analysis")
    print("="*50)
    print(f"Fitted κ_global = {envelope_result['kappa_fitted']:.4f} ± {envelope_result['kappa_std']:.4f}")
    print(f"Twin prime constant C₂ = {C2:.4f}")
    print(f"Ratio κ/C₂ = {envelope_result['ratio']:.4f}")
    
    if abs(envelope_result['ratio'] - 1) < 0.1:
        print("\n✓ Good agreement: κ_global ≈ C₂")
    
    # Plot
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_coefficient_analysis(results, envelope_result, args.output)


if __name__ == "__main__":
    main()
