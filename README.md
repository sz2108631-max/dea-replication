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

## 数据清洗流程

本项目涉及三个阶段的完整数据处理管线。

### Phase 1：构建企业供应链韧性指标 `res_v2`

**脚本：** `code/01_construct_resilience.py`

#### 1.0 原始数据加载

| 数据 | 来源 | 关键字段 |
|------|------|----------|
| 资产负债表 | `raw_data/资产负债表.xlsx` | 现金、应收票据、应收账款、预付账款、存货 |
| 利润表 | `raw_data/利润表.xlsx` | 营业总收入、营业收入、营业成本 |
| 供应链集中度 | `data/客户&供应商&供应链集中度（未剔除）.dta` | 前五大客户销售额、前五大供应商采购额 |
| 企业控制变量 | `data/常用控制变量2000_2024_Ver3.1.dta` | ROA、营业收入、总资产 |
| 申请专利质量 | `raw_data/申请专利质量.dta` | 申请专利知识宽度 |
| 授权专利质量 | `raw_data/授权专利质量.dta` | 授权专利知识宽度 |
| 专利被引证 | `data/变量_被引证.xlsx` | 外部专利被引次数（可选，缺失则 ln_citation=0） |

- 股票代码统一转换为 6 位字符串（`str.zfill(6)`）
- 财务数据按 `stkcode` + `year` 合并为企业-年度面板

#### 1.1 构建 11 个三级指标

| # | 指标 | 计算方式 | 方向 |
|---|------|----------|------|
| ind1 | 库存调整幅度 | ln(\|Inventory_t − Inventory_{t-1}\| + 1) | + |
| ind2 | 供需偏离度 | \|CV(COGS) − CV(Revenue)\|，滚动 3 年 | − |
| ind3 | 供应链集中度 | (供应商集中度 + 客户集中度) / 2 | − |
| ind4 | 供应商稳定性 | 1 − \|供应商比例变动\| / 上年供应商比例 | + |
| ind5 | 客户稳定性 | 1 − \|客户比例变动\| / 上年客户比例 | + |
| ind6 | 盈利能力 | ROA | + |
| ind7 | 现金周转率 | 营业总收入 / 现金（截尾 P99） | + |
| ind8 | 供需协同 | (应收票据+应收账款+预付账款) / 营业收入（截尾 P99） | − |
| ind9 | 申请专利知识宽度 | 企业申请专利 IPC 知识宽度 | + |
| ind10 | 授权专利知识宽度 | 企业授权专利 IPC 知识宽度 | + |
| ind11 | 外部知识引用 | ln(外部专利被引次数 + 1) | + |

#### 1.2 数据预处理

- **缩尾处理：** 每个指标在 1% 和 99% 分位数处缩尾（winsorize）
- **正向化：** 负向指标取反（ind2 供需偏离度、ind8 应收占比取负值）
- **标准化：** Z-score 标准化（`StandardScaler`，均值为 0，标准差为 1）

#### 1.3 深度自编码器降维

通过栈式自编码器将 11 维指标降维至 4 维潜在特征空间：

```
11D → 8D → 6D → 4D (瓶颈层)
```

- **Layer 1：** MLPRegressor(8, ReLU)，11→8，max_iter=200
- **Layer 2：** MLPRegressor(6, ReLU)，8→6，max_iter=200
- **Layer 3：** MLPRegressor(4, Identity)，6→4，max_iter=300（瓶颈层）
- 重构 MSE 作为降维质量检验
- 随机种子固定（`random_state=42`）以确保可复现性

#### 1.4 Sigma 原则地标中心选择

- **子采样：** 从全样本中随机抽取 5,000 个观测值用于地标选择
- **截断距离：** 距离矩阵上三角的 P2 分位数（`dc_percentile=2.0`）
- **局部密度 ρ_i：** 距离小于 dc 的点数
- **最小距离 δ_i：** 到更高密度点的最小距离
- **地标得分 γ_i = ρ_i × δ_i：** 同时考虑密度和分离度
- **阈值：** γ ≥ μ + 2σ（`sigma_multiplier=2.0`）
- **去冗余：** 若两个地标间距 < dc，仅保留密度更高的那个

#### 1.5 异常值剔除

- 计算每个样本到最近地标中心的距离
- 剔除距离 > μ + 2.5σ 的样本（`outlier_sigma=2.5`）

#### 1.6 TOPSIS 评分

- **模式：** 逐年动态（`dynamic`）—— 每年独立运行 TOPSIS
- **权重：** 熵权法（entropy weights），方差为零的维度权重为零
- **决策矩阵归一化：** 向量归一化（`R / sqrt(sum(R²))`）
- **计算：** 正理想解距离 S⁺、负理想解距离 S⁻ → res_v2 = S⁻ / (S⁺ + S⁻)
- 年份观测值 < 5 时，回退为等权重

#### 1.7 输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `output/中间结果/res_v2_panel.csv` | CSV | 企业-年度面板（stkcode, year, res_v2） |
| `output/中间结果/res_v2_panel.dta` | Stata | 同上，Stata 格式 |

---

### Phase 2：合并全量分析面板

**脚本：** `code/02_prepare_panel.py`

#### 2.1 加载 DEA 变量

从 `data/dea_original.dta` 读取核心解释变量：
- `Dea`：企业数据要素应用能力综合得分
- `Breadth`：应用广度得分
- `Depth`：应用深度得分
- `Dea_count` / `Breadth_count` / `Depth_count`：词频计数版本
- `_b_raw` / `_d_raw`：原始广度/深度词数
- `_eff_chars`：有效字符数

#### 2.2 顺序合并数据集

| 步骤 | 数据 | 合并键 | 连接方式 | 关键变量 |
|------|------|--------|----------|----------|
| 1 | `常用控制变量2000_2024_Ver3.1.dta` | stkcode, year | left | lnage, lnsize, klr, lev, bsize, dual, lnrd, indrate, own, ind, ind_str, citycode, soe, area_1, hhi_d, hightech, STPT |
| 2 | `res_v2_panel.csv` | stkcode, year | **inner** | res_v2 → 重命名为 `res` |
| 3 | `内部控制指数2000-2024.dta` | stkcode, year | left | IC1 → 重命名为 `IC` |
| 4 | `交易成本2000-2024.dta` | stkcode, year | left | 交易成本 → 重命名为 `cost` |
| 5 | `pagerank.xlsx` | stkcode, year | left | PageRank_C（客户）, PageRank_P（供应商） |
| 6 | `上市公司供应链地理距离（2001-2024年）.dta` | stkcode, year | left | Disw_c（客户距离）, Disw_s（供应商距离） |
| 7 | `数据要素市场化配置2000-2024.dta` | stkcode, year | left | Treat, Post, DID |
| 8 | `lightning.dta` | citycode | left | 雷击频率 → 重命名为 `shandian`（IV） |
| 9 | `地级市数据交易所 did.dta` | year, city_x | left | exchange_treat, exchange_market |

> **注意：** 步骤 2 使用 inner join —— 仅有 res_v2 的企业-年度才会进入最终面板。

#### 2.3 样本筛选

- **剔除 ST/PT 企业：** `STPT != 1`
- **剔除金融与房地产行业：** `ind_str` 包含「金融、保险、银行、证券、货币、其他金融、建筑、房地产」

#### 2.4 变量处理

- **缩尾处理（1% / 99%）：** `lnrd`（研发投入）、`lev`（杠杆率）、`klr`（资本劳动比）
- **Stata 编码变量生成：**
  - `id`：企业数值编码（从 stkcode 类别编码 + 1）
  - `city`：城市数值编码（从 citycode 类别编码 + 1）
  - `ind_num`：行业数值编码
- **PageRank 中心化交互项：**
  - 对 PageRank_C 和 PageRank_P 分别进行均值中心化 → PageRank_C1, PageRank_P1
  - 生成 Dea / Breadth / Depth 与中心化 PageRank 的交互项（去中心化乘积）

#### 2.5 输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `output/replication_panel_own.dta` | Stata | 全量分析面板，包含所有回归所需变量 |

---

### Phase 3：Stata 回归分析

**脚本：** `code/03_main_analysis.do`

- 读入 `output/replication_panel_own.dta`
- 全局控制变量：`lnage klr lnsize bsize dual lnrd indrate own lev`
- 固定效应：企业（`id`）+ 年份（`year`），聚类标准误：城市（`city`）
- 生成 22 个回归表格（表 1–5 + 附录 A6–A11），输出至 `output/tables/`

### Step 4：Python 稳健性检验

```bash
python code/12_dfbeta.py
```

**DFBETA 影响统计量分析** — 检验 Dea 系数对个别观测的敏感性：
1. Pooled OLS 估计基准模型，计算每个观测对 Dea 系数的 DFBETA 影响统计量
2. 按 DFBETA 绝对值从大到小排序，在 **P2 / P3 / P5 / P10** 四个阈值下（即剔除最有影响力的 2% / 3% / 5% / 10% 样本）重新估计 PanelOLS
3. 若仅剔除 3% 样本即可翻转系数符号或显著性，则基准结果不稳健

**关键结果：**

| 阈值 | 剔除样本 | Dea 系数 | p 值 |
|------|----------|----------|------|
| 全样本 | 0 | −0.0010** | 0.017 |
| DFBETA > P3 | 805 (3%) | +0.0008** | 0.048 |
| DFBETA > P5 | 1,341 (5%) | +0.0019*** | <0.001 |

输出：`output/dfbeta_regression_report.csv`

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
│   ├── 11_tables.py                  发表格式表格
│   └── 12_dfbeta.py                  DFBETA 影响统计量分析
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
