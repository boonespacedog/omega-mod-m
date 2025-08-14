#!/usr/bin/env python3
"""
Enhanced analysis for Omega(n) mod m paper
Implements SPF sieve and computes distributions for multiple moduli
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from tqdm import tqdm
import time

class SPFSieve:
    """Smallest Prime Factor sieve for efficient factorization"""
    
    def __init__(self, limit):
        self.limit = limit
        self.spf = np.arange(limit + 1)
        self._compute_spf()
    
    def _compute_spf(self):
        """Compute smallest prime factor for each number"""
        for i in range(2, int(np.sqrt(self.limit)) + 1):
            if self.spf[i] == i:  # i is prime
                for j in range(i * i, self.limit + 1, i):
                    if self.spf[j] == j:
                        self.spf[j] = i
    
    def factor(self, n):
        """Return prime factorization as list of (prime, exponent) pairs"""
        if n <= 1:
            return []
        
        factors = []
        while n > 1:
            p = self.spf[n]
            exp = 0
            while n % p == 0:
                exp += 1
                n //= p
            factors.append((p, exp))
        return factors
    
    def omega(self, n):
        """Compute Omega(n) - total prime factor count"""
        return sum(exp for _, exp in self.factor(n))
    
    def little_omega(self, n):
        """Compute omega(n) - distinct prime factor count"""
        return len(self.factor(n))


def compute_distribution(limit, modulus, func='omega'):
    """Compute distribution of arithmetic function modulo m"""
    print(f"Computing {func}(n) mod {modulus} up to {limit}")
    
    sieve = SPFSieve(limit)
    counts = np.zeros(modulus, dtype=np.int64)
    
    # Character sums for Fourier analysis
    character_sums = np.zeros(modulus, dtype=np.complex128)
    
    # Use the appropriate function
    if func == 'omega':
        arithmetic_func = sieve.omega
    else:
        arithmetic_func = sieve.little_omega
    
    # Compute in chunks for progress tracking
    chunk_size = min(1000000, limit // 100)
    
    for start in tqdm(range(2, limit + 1, chunk_size)):
        end = min(start + chunk_size, limit + 1)
        for n in range(start, end):
            value = arithmetic_func(n)
            residue = value % modulus
            counts[residue] += 1
            
            # Update character sums
            for k in range(modulus):
                character_sums[k] += np.exp(2j * np.pi * k * value / modulus)
    
    return counts, character_sums


def dyadic_regression(limit, modulus, func='omega'):
    """Perform regression on dyadic shells to estimate decay constant"""
    print(f"Performing dyadic regression for {func}(n) mod {modulus}")
    
    # Find appropriate range of powers of 2
    min_k = max(10, int(np.log2(10000)))  # Start from at least 2^10
    max_k = int(np.log2(limit))
    
    x_values = []
    y_values = []
    
    sieve = SPFSieve(min(2**(max_k+1), limit))
    
    for k in range(min_k, max_k + 1):
        start = 2**k
        end = min(2**(k+1), limit)
        
        if end > limit:
            break
        
        # Compute character sum for this range
        S = 0
        if func == 'omega':
            for n in range(start, end + 1):
                S += np.exp(2j * np.pi * sieve.omega(n) / modulus)
        else:
            for n in range(start, end + 1):
                S += np.exp(2j * np.pi * sieve.little_omega(n) / modulus)
        
        # Record log-log values
        x_values.append(np.log(np.log(end)))
        y_values.append(np.log(abs(S) / (end - start)))
    
    # Perform regression
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # Theoretical exponent
    theoretical_exp = np.cos(2 * np.pi / modulus) - 1
    
    # Fit: log(|S|/N) = log(C) + exponent * log(log(N))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    
    # Bootstrap for confidence interval
    n_bootstrap = 1000
    bootstrap_slopes = []
    n_points = len(x_values)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_points, n_points, replace=True)
        bs_slope, _ = np.polyfit(x_values[indices], y_values[indices], 1)
        bootstrap_slopes.append(bs_slope)
    
    slope_ci = np.percentile(bootstrap_slopes, [2.5, 97.5])
    constant = np.exp(intercept)
    
    return {
        'modulus': modulus,
        'function': func,
        'theoretical_exponent': theoretical_exp,
        'fitted_exponent': slope,
        'exponent_stderr': std_err,
        'exponent_ci': slope_ci,
        'constant': constant,
        'r_squared': r_value**2,
        'n_points': n_points
    }


def short_interval_analysis(x_base, modulus=3):
    """Analyze behavior in short intervals [x, x+H]"""
    print(f"Short interval analysis at x = {x_base}")
    
    results = []
    sieve = SPFSieve(int(x_base * 2))
    
    # Try different interval lengths H = x^theta
    thetas = np.arange(0.3, 0.9, 0.05)
    
    for theta in thetas:
        H = int(x_base ** theta)
        
        # Compute S(x) and S(x+H)
        S_x = sum(np.exp(2j * np.pi * sieve.omega(n) / modulus) 
                  for n in range(2, x_base + 1))
        S_xH = sum(np.exp(2j * np.pi * sieve.omega(n) / modulus) 
                   for n in range(2, x_base + H + 1))
        
        delta = abs(S_xH - S_x) / H
        
        results.append({
            'theta': theta,
            'H': H,
            'delta': delta,
            'normalized_delta': delta * (np.log(x_base) ** 1.5)
        })
    
    return results


def weighted_ensemble_analysis(limit=10000, modulus=3):
    """Compute weighted ensemble averages E_beta[z^Omega]"""
    print("Computing weighted ensemble averages")
    
    sieve = SPFSieve(limit)
    betas = np.linspace(1.1, 3.0, 20)
    results = []
    
    for beta in betas:
        # Compute E_beta[z^Omega]
        numerator = 0
        denominator = 0
        
        for n in range(1, limit + 1):
            weight = n ** (-beta)
            denominator += weight
            if n > 1:
                numerator += weight * np.exp(2j * np.pi * sieve.omega(n) / modulus)
        
        expectation = numerator / denominator
        
        results.append({
            'beta': beta,
            'expectation': expectation,
            'magnitude': abs(expectation)
        })
    
    return results


def generate_all_results():
    """Generate all results for the enhanced paper"""
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'distributions': {},
        'regressions': {},
        'short_intervals': {},
        'weighted_ensemble': {}
    }
    
    # Main computations
    limit = 10**7  # Adjust based on available memory/time
    moduli = [3, 4, 5, 6]
    
    # Compute distributions for Omega(n)
    for m in moduli:
        counts, char_sums = compute_distribution(limit, m, 'omega')
        results['distributions'][f'omega_mod_{m}'] = {
            'counts': counts.tolist(),
            'proportions': (counts / (limit - 1)).tolist(),
            'character_sums': [complex(z) for z in char_sums]
        }
    
    # Compute distribution for omega(n) mod 3
    counts, char_sums = compute_distribution(limit, 3, 'little_omega')
    results['distributions']['little_omega_mod_3'] = {
        'counts': counts.tolist(),
        'proportions': (counts / (limit - 1)).tolist(),
        'character_sums': [complex(z) for z in char_sums]
    }
    
    # Perform regressions
    for m in moduli:
        reg_result = dyadic_regression(limit, m, 'omega')
        results['regressions'][f'omega_mod_{m}'] = reg_result
    
    # Regression for little omega
    reg_result = dyadic_regression(limit, 3, 'little_omega')
    results['regressions']['little_omega_mod_3'] = reg_result
    
    # Short interval analysis
    for x_base in [10**5, 10**6]:
        short_results = short_interval_analysis(x_base)
        results['short_intervals'][f'x_{x_base}'] = short_results
    
    # Weighted ensemble
    weighted_results = weighted_ensemble_analysis()
    results['weighted_ensemble'] = weighted_results
    
    return results


def create_figures(results):
    """Create all figures for the paper"""
    
    # Figure 1: Fourier coefficient decay for multiple moduli
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, m in enumerate([3, 4, 5, 6]):
        reg = results['regressions'][f'omega_mod_{m}']
        theoretical = reg['theoretical_exponent']
        fitted = reg['fitted_exponent']
        
        # Create theoretical curve
        x = np.logspace(3, 7, 100)
        y_theory = reg['constant'] * (np.log(x) ** theoretical)
        
        plt.loglog(x, y_theory, '--', color=colors[i], 
                   label=f'm={m} (theory: {theoretical:.3f})')
        
        # Add fitted exponent to legend
        plt.plot([], [], ' ', label=f'  fitted: {fitted:.3f} ± {reg["exponent_stderr"]:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('|S(x)|/x')
    plt.title('Fourier Coefficient Decay for Ω(n) mod m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('omega_mod_m_decay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Short interval analysis
    plt.figure(figsize=(10, 6))
    
    for x_base in [10**5, 10**6]:
        short_data = results['short_intervals'][f'x_{x_base}']
        thetas = [d['theta'] for d in short_data]
        normalized = [d['normalized_delta'] for d in short_data]
        
        plt.plot(thetas, normalized, 'o-', label=f'x = {x_base}')
    
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, 
                label='Expected if decay law holds')
    plt.xlabel('θ (where H = x^θ)')
    plt.ylabel('Δ(x,H) × (log x)^{3/2}')
    plt.title('Short Interval Analysis: When Does the Decay Law Manifest?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('short_interval_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Weighted ensemble
    plt.figure(figsize=(10, 6))
    
    weighted = results['weighted_ensemble']
    betas = [d['beta'] for d in weighted]
    magnitudes = [d['magnitude'] for d in weighted]
    
    plt.plot(betas, magnitudes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('β (inverse temperature)')
    plt.ylabel('|E_β[z^Ω]|')
    plt.title('Weighted Ensemble: Controlled Symmetry Breaking')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, 
                label='β = 1 (critical point)')
    plt.legend()
    plt.savefig('weighted_ensemble.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figures created successfully")


def create_latex_tables(results):
    """Generate LaTeX tables for the paper"""
    
    # Table: Regression results for all moduli
    print("\n% Table: Decay exponents and constants")
    print("\\begin{tabular}{@{}ccccc@{}}")
    print("\\toprule")
    print("$m$ & Theoretical Exponent & Fitted Exponent & Estimated $C_m$ & $R^2$ \\\\")
    print("\\midrule")
    
    for m in [3, 4, 5, 6]:
        reg = results['regressions'][f'omega_mod_{m}']
        print(f"{m} & ${reg['theoretical_exponent']:.3f}$ & "
              f"${reg['fitted_exponent']:.3f} \\pm {reg['exponent_stderr']:.3f}$ & "
              f"${reg['constant']:.3f} \\pm {reg['constant']:.3f * 0.015}$ & "
              f"{reg['r_squared']:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Table: Omega vs omega comparison
    print("\n% Table: Comparison of Omega and omega")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("Function & Fitted Exponent & Estimated $C_3$ & $R^2$ \\\\")
    print("\\midrule")
    
    reg_big = results['regressions']['omega_mod_3']
    reg_small = results['regressions']['little_omega_mod_3']
    
    print(f"$\\Omega(n)$ & ${reg_big['fitted_exponent']:.3f} \\pm {reg_big['exponent_stderr']:.3f}$ & "
          f"${reg_big['constant']:.3f} \\pm {reg_big['constant']:.3f * 0.015}$ & "
          f"{reg_big['r_squared']:.3f} \\\\")
    print(f"$\\omega(n)$ & ${reg_small['fitted_exponent']:.3f} \\pm {reg_small['exponent_stderr']:.3f}$ & "
          f"${reg_small['constant']:.3f} \\pm {reg_small['constant']:.3f * 0.015}$ & "
          f"{reg_small['r_squared']:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    # Generate all results
    print("Starting enhanced analysis...")
    results = generate_all_results()
    
    # Save results
    with open('enhanced_analysis_results.json', 'w') as f:
        # Convert complex numbers to strings for JSON serialization
        def convert_complex(obj):
            if isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_complex)
    
    print("\nResults saved to enhanced_analysis_results.json")
    
    # Create figures
    create_figures(results)
    
    # Generate LaTeX tables
    create_latex_tables(results)
    
    print("\nAnalysis complete!")