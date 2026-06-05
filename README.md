# Replication Package: Data Element Application and Supply Chain Resilience

Replication code for 巫强 (2026) "企业数据要素应用能力与供应链韧性：广度与深度的差异化机制".

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── code/
│   ├── 01_construct_resilience.py   # Construct supply chain resilience (res)
│   ├── 02_prepare_panel.py          # Merge all variables into analysis panel
│   ├── 03_main_analysis.do          # Main Stata replication (Tables 1–5, Appendix)
│   ├── 04_placebo.py                # Placebo test (permutation)
│   ├── 05_psm.py                    # Propensity score matching
│   ├── 06_iv.py                     # Instrumental variable regression
│   ├── 07_mechanism.py              # Mechanism analysis (interaction terms)
│   ├── 08_heterogeneity.py           # Heterogeneity analysis
│   ├── 09_confounding.py            # Confounding policy checks
│   ├── 10_robustness.py             # Additional robustness checks
│   └── 11_tables.py                 # Publication-format tables
├── output/
│   └── tables/                      # Regression output (.txt / .xls)
├── data/                            # Input data (not included)
└── raw_data/                        # Raw financial & patent data (not included)
```

## Requirements

- **Stata 16+** with `reghdfe`, `ivreghdfe`, `ppmlhdfe`, `outreg2`, `bdiff`
- **Python 3.9+** with dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Construct resilience measure

```bash
python code/01_construct_resilience.py
```

Reads raw financial data from `raw_data/`, produces `output/中间结果/res_v2_panel.dta`.

### Step 2: Prepare analysis panel

```bash
python code/02_prepare_panel.py
```

Merges resilience measure, DEA variables, controls, and auxiliary data into `output/replication_panel_own.dta`.

### Step 3: Run Stata replication

```stata
do code/03_main_analysis.do
```

Outputs tables to `output/tables/`.

## Data

Due to license restrictions, input data files (`data/`, `raw_data/`) are not included in this repository. Place the required data files in the respective directories before running the pipeline.

## License

MIT License. See [LICENSE](LICENSE).
