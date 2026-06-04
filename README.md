# Replication Package: 企业数据要素应用能力与供应链韧性

This repository provides a replication package for 巫强 (2026) "企业数据要素应用能力与供应链韧性：广度与深度的差异化机制".

## Overview

The project constructs a firm-level supply chain resilience measure (`res_v2`) using an autoencoder-based dimensionality reduction combined with TOPSIS, then replicates the key empirical analyses from the paper.

## Project Structure

```
.
├── replication.do              # Main Stata replication script
├── src/
│   ├── 24_construct_res_v2.py  # Construct supply chain resilience (res_v2)
│   ├── 18_prepare_stata_data.py # Merge data for Stata analysis
│   ├── 01_mechanism_interaction.py
│   ├── 02_heterogeneity_enhanced.py
│   ├── 03_placebo_test.py
│   ├── 04_psm_matching.py
│   ├── 06_iv_regression.py
│   └── 20_descriptive_stats.py
├── data/                       # Input data files
├── Y 的复现数据/                # Raw financial and patent data
└── output/                     # Regression output tables
```

## Requirements

- **Stata 16+** with `reghdfe`, `ivreghdfe`, `ppmlhdfe`, `outreg2`, `bdiff` packages
- **Python 3.9+** with `pandas`, `numpy`, `scikit-learn`, `pyreadstat`, `scipy`

Install Python dependencies:
```bash
pip install pandas numpy scikit-learn pyreadstat scipy openpyxl
```

## Quick Start

### Step 1: Construct supply chain resilience (res_v2)

```bash
python src/24_construct_res_v2.py
```

This reads raw financial data from `Y 的复现数据/` and produces `output/中间结果/res_v2_panel.dta`.

### Step 2: Prepare the analysis dataset

```bash
python src/18_prepare_stata_data.py
```

This merges `res_v2`, DEA variables, controls, and auxiliary data into `output/replication_panel_own.dta`.

### Step 3: Run Stata replication

```stata
do replication.do
```

Outputs tables to `output/table*.txt` and `output/table*.xls`.

## Data Sources

The analysis uses the following data sources:

- **DEA variables**: Firm-level data element application capability (text-mining based)
- **Financial data**: CSMAR balance sheets and income statements
- **Control variables**: Standard firm-level controls (size, leverage, age, etc.)
- **Auxiliary data**: Internal control index, transaction costs, PageRank, geographic distance, lightning frequency, policy pilots

Due to data license restrictions, raw data files are not included. Please contact the authors for data access.

## License

MIT License. See [LICENSE](LICENSE) for details.
