# Replication Package: Data Element Application and Supply Chain Resilience

[![Release](https://img.shields.io/badge/release-2026.06-blue.svg)](.)

Replication package for 巫强 (2026) "企业数据要素应用能力与供应链韧性：广度与深度的差异化机制" (*Data Element Application Capability and Supply Chain Resilience: Differentiated Mechanisms of Breadth and Depth*).

The original paper and appendix are available in the [`paper/`](paper/) directory.

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── paper/                             # Original paper & appendix
│   ├── 企业数据要素应用能力与供应链韧性_巫强.pdf
│   ├── 附录.pdf
│   ├── 原始程序.do
│   └── CIE数据使用说明.pdf
├── code/
│   ├── 01_construct_resilience.py     # Construct supply chain resilience
│   ├── 02_prepare_panel.py            # Merge variables into analysis panel
│   ├── 03_main_analysis.do            # Main Stata replication
│   ├── 04_placebo.py                  # Placebo test
│   ├── 05_psm.py                      # Propensity score matching
│   ├── 06_iv.py                       # Instrumental variable regression
│   ├── 07_mechanism.py                # Mechanism analysis
│   ├── 08_heterogeneity.py            # Heterogeneity analysis
│   ├── 09_confounding.py              # Confounding checks
│   ├── 10_robustness.py               # Additional robustness
│   └── 11_tables.py                   # Publication-format tables
├── output/
│   └── tables/                        # Regression output
├── data/                              # Input data (see below)
└── raw_data/                          # Raw financial & patent data
```

## Requirements

- **Stata 16+** with `reghdfe`, `ivreghdfe`, `ppmlhdfe`, `outreg2`, `bdiff`
- **Python 3.9+**

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Construct resilience measure

```bash
python code/01_construct_resilience.py
```

### Step 2: Prepare analysis panel

```bash
python code/02_prepare_panel.py
```

### Step 3: Run Stata replication

```stata
do code/03_main_analysis.do
```

Results are written to `output/tables/`.

## Data Requirements

Due to license restrictions, input data are not included. Place the following files before running:

**`data/` directory:**
| File | Description |
|------|-------------|
| `dea_original.dta` | Firm-level DEA variables |
| `常用控制变量2000_2024_Ver3.1.dta` | Standard firm-level controls |
| `内部控制指数2000-2024.dta` | Internal control index |
| `交易成本2000-2024.dta` | Transaction cost data |
| `pagerank.xlsx` | Supply chain PageRank |
| `上市公司供应链地理距离（2001-2024年）.dta` | Geographic distance |
| `数据要素市场化配置2000-2024.dta` | Data market policy DID |
| `地级市数据交易所 did.dta` | Data exchange pilot DID |
| `lightning.dta` | Lightning frequency (IV) |
| `变量_被引证.xlsx` | Patent citation data (optional) |

**`raw_data/` directory:**
| File | Description |
|------|-------------|
| `资产负债表.xlsx` | Balance sheet (CSMAR format) |
| `利润表.xlsx` | Income statement (CSMAR format) |
| `申请专利质量.dta` | Patent application quality |
| `授权专利质量.dta` | Patent grant quality |

## License

MIT License. See [LICENSE](LICENSE).

## Reference

巫强, 等. (2026). 企业数据要素应用能力与供应链韧性：广度与深度的差异化机制.
