# Omega Modulo m: Finite-Size Equidistribution Analysis

This repository contains the code, data, and paper for our investigation of the finite-size distribution of Ω(n) modulo m.

## Paper

**Finite-Size Equidistribution of Ω(n) Modulo m: Theory and Computation**  
Oksana Sudoma & Claude (Anthropic)

We study the finite-size distribution of the additive prime factor count Ω(n) modulo m, revealing structured deviations that match classical Selberg–Delange/Halász predictions with explicit constant estimation.

## Key Results

- **Main Finding**: The first Fourier coefficient decays as |S(x)|/x ~ C_m(log x)^(cos(2π/m)-1)
- **For m=3**: Verified decay |S(x)|/x ~ 0.523(log x)^(-3/2) up to x = 10^8  
- **Universal Law**: Extended to m = 4, 5, 6 and to ω(n), confirming universal exponent
- **Short Intervals**: Decay law requires H ≳ x^0.6 to manifest locally

## Repository Structure

```
omega-mod-m/
├── src/                    # Core analysis code
│   ├── omega_mod3_analysis.py
│   ├── compute_enhanced_analysis.py
│   ├── omega_mod3_visualizations.py
│   └── generate_new_figures.py
├── data/                   # Computational results
│   ├── enhanced_analysis_results.json
│   ├── final_results.json
│   └── omega_modm_V2_report.json
├── figures/                # All paper figures
│   └── [8 PNG files]
├── paper/                  # LaTeX source and bibliography
│   ├── omega_mod3_paper_enhanced.tex
│   └── omega_mod3_references.bib
├── notebooks/              # Interactive explorations
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

### Compute Ω(n) mod m distribution
```python
python src/omega_mod3_analysis.py --modulus 3 --limit 10000000
```

### Generate paper figures
```python
python src/generate_new_figures.py
```

### Run enhanced analysis with bootstrap
```python
python src/compute_enhanced_analysis.py
```

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

| Modulus m | Theoretical Exponent | Fitted Exponent | Constant C_m | R² |
|-----------|---------------------|-----------------|--------------|-----|
| 3 | -1.500 | -1.497 ± 0.012 | 0.523 ± 0.008 | 0.994 |
| 4 | -1.000 | -0.998 ± 0.009 | 0.412 ± 0.006 | 0.997 |
| 5 | -0.691 | -0.688 ± 0.008 | 0.387 ± 0.005 | 0.996 |
| 6 | -0.500 | -0.502 ± 0.007 | 0.298 ± 0.004 | 0.998 |

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