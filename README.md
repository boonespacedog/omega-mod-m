# Omega Modulo m: Finite-Size Equidistribution Analysis

This repository contains the code, data, and paper for our investigation of the finite-size distribution of Ω(n) modulo m.

## Paper

**Finite-Size Equidistribution of Ω(n) Modulo m: Theory and Computation**  
Oksana Sudoma & Claude (Anthropic)

We study the finite-size distribution of the additive prime factor count Ω(n) modulo m, revealing structured deviations that match classical Selberg–Delange/Halász predictions with explicit constant estimation.

## Key Results

- **Main Finding**: The first Fourier coefficient decays as |S(x)|/x ~ C_m(log x)^(cos(2π/m)-1)
- **For m=3**: Verified decay |S(x)|/x ~ 1.708(log x)^(-3/2) up to x = 10^8  
- **Universal Law**: Extended to m = 4, 5, 6 and to ω(n), confirming universal exponent
- **Short Intervals**: Decay law requires H ≳ x^0.6 to manifest locally
- **Constants**: C_m values determined both theoretically (Euler product) and empirically (regression)

## Repository Structure

```
omega-mod-m/
├── src/                    # Core analysis code
│   └── omega_analysis_final.py  # Complete, corrected implementation
├── data/                   # Computational results
│   └── [JSON result files]
├── figures/                # All paper figures
│   └── [8 PNG files with correct constants]
├── paper/                  # LaTeX source and PDF
│   ├── omega_mod3_paper_enhanced.tex
│   ├── omega_mod3_paper_enhanced.pdf
│   └── omega_mod3_references.bib
└── requirements.txt        # Python dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/boonespacedog/omega-mod-m.git
cd omega-mod-m

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run complete analysis
```python
python src/omega_analysis_final.py
```

This will:
- Compute theoretical constants C_m using correct Euler product
- Generate empirical verification data
- Create figures with correct constants
- Output LaTeX table with results

## Methods

1. **SPF Sieve**: Efficient O(N log log N) factorization using smallest prime factors
2. **Dyadic Shell Regression**: Log-log regression on intervals [2^k, 2^(k+1)]
3. **Bootstrap Confidence**: 1000-sample bootstrap for uncertainty quantification
4. **Fourier Framework**: Complete characterization via single complex coefficient

## Key Algorithms

### Computing Ω(n) Distribution
- Precompute smallest prime factors (SPF) up to N
- Factor each n using SPF table in O(log n) time
- Track character sums S(x) = Σ_{n≤x} z^{Ω(n)}

### Constant Estimation
- Dyadic shell regression: fit |S(x)|/x vs (log x)^α
- Bootstrap resampling for confidence intervals
- Theoretical validation via truncated Euler products

## Results Summary

| Modulus m | Theoretical Exponent | Constant C_m (Theory) | Constant C_m (Empirical) | Agreement |
|-----------|---------------------|--------------------|------------------------|-----------|
| 3 | -1.500 | 1.708456 | 1.708 ± 0.025 | Excellent |
| 4 | -1.000 | 1.555237 | 1.555 ± 0.020 | Excellent |
| 5 | -0.691 | 1.273375 | 1.273 ± 0.015 | Excellent |
| 6 | -0.500 | 1.117734 | 1.118 ± 0.012 | Excellent |

## Citation

If you use this code or results, please cite:
```bibtex
@article{sudoma2025omega,
  title={Finite-Size Equidistribution of Omega(n) Modulo m: Theory and Computation},
  author={Sudoma, Oksana and Claude (Anthropic)},
  year={2025},
  journal={arXiv preprint}
}
```

## Reproducibility

- **Hardware**: Intel Core i9-12900K, 64GB RAM
- **Software**: Python 3.11, NumPy 1.24
- **Random Seed**: 42 for all bootstrap samples
- **Data**: All computational results in `data/` directory

## License

This work is released under the MIT License.

## Contact

Oksana Sudoma - boonespacedog@gmail.com

## Acknowledgments

We thank the mathematical community for foundational work that made this discovery possible. This represents a novel human-AI collaboration in mathematical discovery.