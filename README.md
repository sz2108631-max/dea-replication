# Replication Package: Data Element Application and Supply Chain Resilience

[![Release](https://img.shields.io/badge/release-2026.06-blue.svg)](.)

复现 巫强 (2026) "企业数据要素应用能力与供应链韧性：广度与深度的差异化机制"。

原论文及附录见 [`paper/`](paper/) 目录。

## 关键数据文件

| 文件 | 位置 | 说明 |
|------|------|------|
| DEA 变量（Dea, Breadth, Depth） | `data/dea_original.dta` | 企业数据要素应用能力，文本挖掘构建 |
| 企业供应链韧性（res_v2） | `output/中间结果/res_v2_panel.dta` | 自编码器 + TOPSIS 构建，运行 Step 1 生成 |
| 完整分析面板 | `output/replication_panel_own.dta` | DEA + res_v2 + 控制变量 + 附加数据全量合并，运行 Step 2 生成 |

## 复现流程

### 环境

- **Stata 16+**：`reghdfe`, `ivreghdfe`, `ppmlhdfe`, `outreg2`, `bdiff`
- **Python 3.9+**

```bash
pip install -r requirements.txt
```

### Step 1：构建供应链韧性（res_v2）

```bash
python code/01_construct_resilience.py
```

**输入：** `raw_data/` 下 4 个文件（资产负债表.xlsx, 利润表.xlsx, 申请专利质量.dta, 授权专利质量.dta）+ `data/` 下控制变量

**输出：** `output/中间结果/res_v2_panel.dta`（30,070 obs）

### Step 2：合并分析面板

```bash
python code/02_prepare_panel.py
```

**输入：** Step 1 输出的 res_v2 + `data/dea_original.dta` + 控制变量 + IC、交易成本、PageRank、地理距离、政策 DID、IV 等附加数据

**输出：** `output/replication_panel_own.dta`（最终分析面板）

### Step 3：Stata 回归

```stata
do code/03_main_analysis.do
```

**输入：** `output/replication_panel_own.dta`

**输出：** `output/tables/` 下 22 个表格（表1–5 + 附录A6–A11）

## 项目结构

```
.
├── paper/                            原论文、附录、原始程序
├── code/
│   ├── 01_construct_resilience.py    构建 res_v2
│   ├── 02_prepare_panel.py           合并分析面板
│   ├── 03_main_analysis.do           Stata 主回归
│   ├── 04_placebo.py                 安慰剂检验
│   ├── 05_psm.py                     倾向得分匹配
│   ├── 06_iv.py                      工具变量
│   ├── 07_mechanism.py               机制分析
│   ├── 08_heterogeneity.py           异质性分析
│   ├── 09_confounding.py             混淆因素
│   ├── 10_robustness.py              附加稳健性
│   └── 11_tables.py                  发表格式表格
├── output/
│   ├── tables/                       回归结果（.txt / .xls）
│   ├── 中间结果/                     res_v2 构建中间产物
│   └── replication_panel_own.dta     最终分析面板
├── data/                             输入数据
├── raw_data/                         原始财务及专利数据
├── requirements.txt
└── .gitignore
```

## 数据文件清单

### `data/`（管道必需）

| 文件 | 大小 | 用途 |
|------|------|------|
| `dea_original.dta` | 4.3 MB | DEA / Breadth / Depth 变量 |
| `常用控制变量2000_2024_Ver3.1.dta` | 61 MB | 企业层面控制变量 |
| `内部控制指数2000-2024.dta` | 10 MB | IC 机制变量 |
| `交易成本2000-2024.dta` | 680 KB | cost 机制变量 |
| `客户&供应商&供应链集中度（未剔除）.dta` | 6.2 MB | res_v2 构建 + 供应链分析 |
| `上市公司供应链地理距离（2001-2024年）.dta` | 2.1 MB | 表5 地理距离 |
| `pagerank.xlsx` | 56 KB | 表4 PageRank |
| `数据要素市场化配置2000-2024.dta` | 20 MB | 政策 DID |
| `地级市数据交易所 did.dta` | 340 KB | 数交所试点 |
| `lightning.dta` | 44 KB | IV 工具变量（雷电频率） |
| `变量_被引证.xlsx` | 11 MB | res_v2 构建（可选） |

### `raw_data/`（管道必需）

| 文件 | 大小 | 用途 |
|------|------|------|
| `资产负债表.xlsx` | 40 MB | res_v2 构建 |
| `利润表.xlsx` | 26 MB | res_v2 构建 |
| `申请专利质量.dta` | 778 KB | res_v2 构建 |
| `授权专利质量.dta` | 755 KB | res_v2 构建 |

## License

MIT License. See [LICENSE](LICENSE).

## Reference

巫强, 等. (2026). 企业数据要素应用能力与供应链韧性：广度与深度的差异化机制.
