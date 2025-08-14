#!/usr/bin/env python3
"""
Analysis of the distribution of Ω(n) modulo 3.

This module provides functions to compute and analyze the distribution
of the prime omega function Ω(n) modulo 3 for ranges of integers.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple
from numba import jit


@jit(nopython=True)
def compute_omega(n: int) -> int:
    """
    Compute Ω(n), the total number of prime factors with multiplicity.
    
    Args:
        n: A positive integer
        
    Returns:
        The value of Ω(n)
    """
    if n <= 1:
        return 0
    
    omega_count = 0
    
    # Handle factors of 2
    while n % 2 == 0:
        omega_count += 1
        n //= 2
    
    # Handle odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            omega_count += 1
            n //= i
        i += 2
    
    # If n is still > 1, it's prime
    if n > 1:
        omega_count += 1
    
    return omega_count


@jit(nopython=True)
def count_omega_mod3_range(start: int, end: int) -> np.ndarray:
    """
    Count the distribution of Ω(n) mod 3 for n in [start, end].
    
    Args:
        start: Starting value (inclusive)
        end: Ending value (inclusive)
        
    Returns:
        Array of counts for residue classes 0, 1, 2 modulo 3
    """
    counts = np.zeros(3, dtype=np.int64)
    
    for n in range(start, end + 1):
        omega_val = compute_omega(n)
        counts[omega_val % 3] += 1
    
    return counts


def analyze_distribution(max_n: int, checkpoints: List[int] = None) -> Dict:
    """
    Analyze the distribution of Ω(n) mod 3 up to max_n.
    
    Args:
        max_n: Maximum value to analyze
        checkpoints: List of values at which to report results
        
    Returns:
        Dictionary containing results at each checkpoint
    """
    if checkpoints is None:
        # Default checkpoints at powers of 10
        checkpoints = []
        power = 3
        while 10**power <= max_n:
            checkpoints.append(10**power)
            power += 1
        if max_n not in checkpoints:
            checkpoints.append(max_n)
    
    print(f"Analyzing Ω(n) mod 3 distribution up to {max_n:,}")
    print("=" * 60)
    
    results = {}
    start_time = time.time()
    
    # Warm up JIT compiler
    _ = count_omega_mod3_range(1, 100)
    
    for checkpoint in checkpoints:
        if checkpoint > max_n:
            break
            
        # Count distribution
        counts = count_omega_mod3_range(1, checkpoint)
        total = counts.sum()
        fractions = counts / total
        
        # Calculate statistics
        expected = total / 3
        chi_squared = sum((counts[i] - expected)**2 / expected for i in range(3))
        max_deviation = max(abs(fractions[i] - 1/3) for i in range(3))
        
        elapsed = time.time() - start_time
        
        results[checkpoint] = {
            'counts': counts.tolist(),
            'fractions': fractions.tolist(),
            'chi_squared': float(chi_squared),
            'max_deviation': float(max_deviation),
            'elapsed_time': elapsed
        }
        
        # Report progress
        print(f"\nn = {checkpoint:,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Counts: {counts[0]:,} | {counts[1]:,} | {counts[2]:,}")
        print(f"Fractions: {fractions[0]:.6f} | {fractions[1]:.6f} | {fractions[2]:.6f}")
        print(f"Chi-squared: {chi_squared:.2f}")
        print(f"Max deviation from 1/3: {max_deviation:.6f}")
    
    return results


def plot_results(results: Dict, output_file: str = "omega_mod3_distribution.png"):
    """
    Create visualization of the distribution analysis.
    
    Args:
        results: Dictionary of results from analyze_distribution
        output_file: Filename for the output plot
    """
    ns = sorted(results.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Fraction evolution
    for i in range(3):
        fractions = [results[n]['fractions'][i] for n in ns]
        ax1.semilogx(ns, fractions, 'o-', label=f'Ω ≡ {i} (mod 3)', markersize=8)
    ax1.axhline(y=1/3, color='k', linestyle='--', alpha=0.5, label='Expected (1/3)')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Fraction')
    ax1.set_title('Distribution of Ω(n) mod 3')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Deviations from uniform
    deviations_0 = [(results[n]['fractions'][0] - 1/3) * 100 for n in ns]
    deviations_1 = [(results[n]['fractions'][1] - 1/3) * 100 for n in ns]
    deviations_2 = [(results[n]['fractions'][2] - 1/3) * 100 for n in ns]
    
    ax2.semilogx(ns, deviations_0, 'o-', label='Ω ≡ 0 (mod 3)')
    ax2.semilogx(ns, deviations_1, 's-', label='Ω ≡ 1 (mod 3)')
    ax2.semilogx(ns, deviations_2, '^-', label='Ω ≡ 2 (mod 3)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('n')
    ax2.set_ylabel('Deviation from 33.33% (%)')
    ax2.set_title('Bias in Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Chi-squared statistic
    chi_squares = [results[n]['chi_squared'] for n in ns]
    ax3.loglog(ns, chi_squares, 'go-', markersize=8)
    ax3.set_xlabel('n')
    ax3.set_ylabel('Chi-squared statistic')
    ax3.set_title('Statistical Significance of Non-uniformity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Maximum deviation
    max_devs = [results[n]['max_deviation'] for n in ns]
    ax4.loglog(ns, max_devs, 'ro-', markersize=8)
    ax4.set_xlabel('n')
    ax4.set_ylabel('Maximum deviation from 1/3')
    ax4.set_title('Convergence Analysis')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Analysis of Ω(n) mod 3 Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_file}")


def save_results(results: Dict, output_file: str = "omega_mod3_results.json"):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results from analyze_distribution
        output_file: Filename for the output JSON
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def main():
    """
    Main function to run the analysis.
    """
    # Analyze up to 10^8 with checkpoints at each power of 10
    max_n = 10**8
    checkpoints = [10**i for i in range(3, 9)]
    
    results = analyze_distribution(max_n, checkpoints)
    
    # Save results and create visualizations
    save_results(results)
    plot_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    final_n = max(results.keys())
    final_result = results[final_n]
    
    print(f"\nAt n = {final_n:,}:")
    for i in range(3):
        fraction = final_result['fractions'][i]
        deviation = (fraction - 1/3) * 100
        print(f"  Ω ≡ {i} (mod 3): {fraction:.6f} ({deviation:+.2f}% from expected)")
    
    print(f"\nChi-squared: {final_result['chi_squared']:.2f}")
    print(f"This indicates a highly significant deviation from uniform distribution.")


if __name__ == "__main__":
    main()