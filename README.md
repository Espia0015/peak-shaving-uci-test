# Peak Shaving Optimization with ML-based Demand Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)

This repository contains the Python implementation accompanying the research article:

**"Predicción de Demanda Energética y Optimización de Peak Shaving en Edificios Comerciales con Carga de Vehículos Eléctricos Mediante Modelos Híbridos de Aprendizaje Automático"**

*Submitted to: Energies (MDPI), ISSN 1996-1073*

---

## Overview

This project implements and compares three forecasting models — **SARIMA**, **XGBoost**, and **LSTM** — for hourly electricity demand prediction. The best-performing model is then integrated with a **convex optimization framework** (CVXPY) for peak shaving using battery energy storage systems (BESS).

### Key Results

| Model | RMSE (kW) | MAE (kW) | MAPE (%) | Ranking |
|-------|-----------|----------|----------|---------|
| **XGBoost** ★ | **0.5234** | **0.3451** | **44.28** | 1° |
| LSTM | 0.5728 | 0.3965 | 60.74 | 2° |
| SARIMA | 1.8028 | 1.4813 | 180.05 | 3° |

**Peak shaving outcomes (day of maximum demand):**
- Maximum demand reduction: **33.7%** (from 4.449 kW to 2.949 kW)
- Load factor improvement: **50.9%** (from 0.453 to 0.683)
- Energy consumption preserved: **100%**

---

## Dataset

The **UCI Individual Household Electric Power Consumption** dataset is used:

- **Source:** https://archive.ics.uci.edu/dataset/235/
- **Reference:** Hebrail, G. & Berard, A. (2006). DOI: [10.24432/C58K54](https://doi.org/10.24432/C58K54)
- **License:** CC BY 4.0
- **Period:** December 2006 – November 2010
- **Resolution:** 1 minute (resampled to 1 hour for this study)
- **Records:** 2,075,259 original measurements

---

## Methodology

The proposed framework follows a **two-stage pipeline**:

### Stage A — Demand Forecasting
1. Data loading and hourly resampling
2. Missing value imputation (linear interpolation)
3. Feature engineering (lags, rolling means, temporal indicators)
4. Train/test split (80/20, chronological)
5. Training of SARIMA, XGBoost, and LSTM
6. Evaluation using RMSE, MAE, and MAPE

### Stage B — Peak Shaving Optimization
Formulated as a convex optimization problem:

**Objective function:**

**Constraints:**
- Power balance: `P_gridₜ = Lₜ + P_chₜ − P_disₜ`
- Battery SOC dynamics: `SOCₜ = SOCₜ₋₁ + ηch·P_chₜ·Δt − (P_disₜ·Δt)/ηdis`
- Peak constraint: `P_gridₜ ≤ P_peak`
- Operational limits for charge/discharge power and SOC

Solved using **CLARABEL** solver via CVXPY.

---

## Requirements

- **Python:** 3.11 (recommended; TensorFlow is not compatible with Python 3.14+)

Install dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels xgboost tensorflow cvxpy matplotlib
```

---

## Usage

### 1. Download the dataset

Download `household_power_consumption.txt` from the UCI repository:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Extract and place the `.txt` file in the project root directory.

### 2. Run the script

```bash
python peak_shaving_modelo.py
```

### 3. Expected output

- Console summary with Tables I and II
- PNG file `resultados_peak_shaving.png` with 4 diagnostic plots:
  - Demand profile (7-day sample)
  - Real vs. predicted values (XGBoost)
  - MAPE comparison across models
  - Peak shaving profile before/after optimization

**Estimated runtime:** 5–10 minutes on a standard desktop (no GPU required).

---

## Repository Structure

---

## Reproducibility Statement

All results reported in the article are fully reproducible by executing the main script under the specified software requirements. Random seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`) to guarantee consistency across runs.

---

## Citation

If you use this code or reference this work, please cite:

```bibtex
@article{cardenas2025peakshaving,
  title   = {Predicción de Demanda Energética y Optimización de Peak Shaving 
             en Edificios Comerciales con Carga de Vehículos Eléctricos 
             Mediante Modelos Híbridos de Aprendizaje Automático},
  author  = {Cardenas, Eric and others},
  journal = {Energies},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Eric Cardenas**  
Maestría en Ingeniería Eléctrica  
Universidad Tecnológica de Panamá  
GitHub: [@Espia0015](https://github.com/Espia0015)

---

## Acknowledgments

This work uses the UCI Individual Household Electric Power Consumption dataset, made publicly available by Hebrail & Berard (2006) under CC BY 4.0 license through the UCI Machine Learning Repository.
