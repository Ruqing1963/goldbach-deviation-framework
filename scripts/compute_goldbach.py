#!/usr/bin/env python3
"""
Compute Goldbach counts and deviations for a range of even integers.

Usage:
    python compute_goldbach.py --nmax 100000 --output data/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils import (
    sieve_of_eratosthenes, 
    goldbach_count, 
    hardy_littlewood, 
    singular_series,
    relative_deviation,
    C2
)


def compute_goldbach_data(nmax, step=2, verbose=True):
    """
    Compute Goldbach counts and related quantities.
    
    Parameters
    ----------
    nmax : int
        Maximum even integer to analyze.
    step : int
        Step size (default 2 for all even integers).
    verbose : bool
        Print progress updates.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: N, G_N, HL_N, S_N, delta_N
    """
    if verbose:
        print(f"Generating primes up to {nmax}...")
    
    primes = sieve_of_eratosthenes(nmax)
    prime_set = set(primes)
    
    if verbose:
        print(f"Found {len(primes)} primes")
        print(f"Computing Goldbach counts for N = 4 to {nmax}...")
    
    results = []
    
    for i, N in enumerate(range(4, nmax + 1, step)):
        G_N = goldbach_count(N, primes, prime_set)
        S_N = singular_series(N)
        HL_N = hardy_littlewood(N, S_N)
        delta_N = relative_deviation(G_N, HL_N)
        
        results.append({
            'N': N,
            'G_N': G_N,
            'HL_N': HL_N,
            'S_N': S_N,
            'delta_N': delta_N
        })
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} values...")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compute Goldbach counts and deviations'
    )
    parser.add_argument(
        '--nmax', type=int, default=100000,
        help='Maximum even integer (default: 100000)'
    )
    parser.add_argument(
        '--output', type=str, default='data/',
        help='Output directory (default: data/)'
    )
    parser.add_argument(
        '--step', type=int, default=2,
        help='Step size for even integers (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Compute data
    df = compute_goldbach_data(args.nmax, args.step)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full data
    counts_file = output_dir / 'goldbach_counts.csv'
    df.to_csv(counts_file, index=False)
    print(f"\nSaved Goldbach counts to {counts_file}")
    
    # Save deviations only (for FFT/Hurst analysis)
    deviations_file = output_dir / 'goldbach_deviations.csv'
    df[['N', 'delta_N']].to_csv(deviations_file, index=False)
    print(f"Saved deviations to {deviations_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    print(f"N range: {df['N'].min()} to {df['N'].max()}")
    print(f"Number of points: {len(df)}")
    print(f"\nDeviation δ(N):")
    print(f"  Mean:   {df['delta_N'].mean():.6f}")
    print(f"  Std:    {df['delta_N'].std():.6f}")
    print(f"  Min:    {df['delta_N'].min():.6f}")
    print(f"  Max:    {df['delta_N'].max():.6f}")
    print(f"\nTwin prime constant C_2 = {C2:.10f}")
    print(f"Envelope estimate κ/ln(N_max) = {C2/np.log(args.nmax):.6f}")


if __name__ == "__main__":
    main()
