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

## 变量构建方法

### 被解释变量：企业供应链韧性 `res`（即 `res_v2`）

供应链韧性采用**深度自编码器 + 密度峰聚类 + TOPSIS**三步法从企业财务与专利数据中无监督构建：

```
11个三级指标 → 自编码器降维(11→8→6→4) → 密度峰聚类(γ=ρ×δ≥μ+2σ) → 异常值剔除 → TOPSIS评分 → res_v2
```

**11 个三级指标**覆盖 5 个维度：

| 维度 | 指标 | 数据来源 |
|------|------|----------|
| 库存调整 | 库存调整幅度 = ln(\|Inventory_t − Inventory_{t-1}\| + 1) | 资产负债表 |
| 供需匹配 | 供需偏离度 = \|CV(COGS) − CV(Revenue)\|（3年滚动） | 利润表 |
| 供应链关系 | 集中度、供应商稳定性、客户稳定性 | 供应链集中度数据 |
| 财务韧性 | ROA、现金周转率、供需协同（应收/收入） | 利润表 + 资产负债表 |
| 知识基础 | 申请/授权专利知识宽度、外部专利引用 | 专利质量 + 被引数据 |

**关键设计选择：**
- **自编码器**：三层栈式 MLP（11→8 ReLU →6 ReLU →4 Identity），随机种子 42 锁定
- **密度峰聚类**：γ = ρ × δ ≥ μ + 2σ，同时考虑密度和分离度，去冗余后剔除异常值（> μ + 2.5σ）
- **TOPSIS**：逐年动态模式 + 熵权法，输出 [0,1] 连续得分

### 解释变量：企业数据要素应用能力 `Dea` / `Breadth` / `Depth`

数据要素应用能力采用**BERT 零样本分类 + 正则规则 + 场景约束**从上市公司 MD&A 年报文本中构建：

```
MD&A文本 → 分句 → 关键词预筛 → [正则预筛 → BERT二分类(阈值0.60)] → 场景约束计数 → ln(count+1)
```

**三个指标：**
| 变量 | 定义 | 统计口径 |
|------|------|----------|
| `Dea` | 数据要素应用能力综合得分 | ln(广度词频 + 深度词频 + 1) |
| `Breadth` | 应用广度 | ln(广度关键词词频 + 1) |
| `Depth` | 应用深度 | ln(深度关键词词频 + 1) |

**关键设计选择：**
- **关键词词典**：按论文附录 2 构建，分广度 5 类（基础设施/处理分析/存储管理/安全治理/业务应用）和深度 5 类（战略定位/组织保障/治理体系/技术深度/业务融合），每类区分核心词与辅助词
- **BERT 否定过滤**：中文 NLI 模型 `Erlangshen-Roberta-110M-NLI`，零样本分类判断语句是否描述「实际部署/应用」关系，阈值 0.60
- **正则预筛**：明确否定句（尚未部署…）、未来计划句（计划/拟/将…）、早期探索句（尚处于研究阶段…）直接排除
- **场景约束**：仅有辅助词无核心词的句子，若为口号式笼统表述（「持续推进数字化转型」等）则不计入
- **稳健性指标**：`Dea_count` = 词频 / MD&A 有效字数 × 1000（剔除停用词后）

## 复现流程

### 环境

- **Stata 16+**：`reghdfe`, `ivreghdfe`, `ppmlhdfe`, `outreg2`, `bdiff`
- **Python 3.9+**

```bash
pip install -r requirements.txt
```

### Step 0：构建 DEA 变量（可选，已有 `data/dea_original.dta`）

```bash
# 设置 MD&A 文本目录（CMDA 数据库下载的年报文本）
export CMDA_DIR="/path/to/CMDA_管理层讨论与分析_ALL"

# 0a. 扩展关键词词典（Word2Vec，可选，已提供预扩展词典）
python code/00_dea_expand_keywords.py

# 0b. 运行 DEA 构建管线（BERT + 规则）
python code/00_dea_construct.py
```

**输入：** CMDA 数据库 MD&A 年报文本（`<CMDA_DIR>/<年份>/文本/<stkcode>_<年份>-12-31.txt`）

**输出：** `output/dea_bert_zero-shot.csv`（可替代 `data/dea_original.dta`）

> **注意：** MD&A 文本来自 CMDA（中国上市公司管理层讨论与分析）数据库，需用户自行获取。项目已提供预计算的 `data/dea_original.dta`，无需运行此步即可复现全部回归。

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

本项目涉及四个阶段的完整数据处理管线。

### Phase 0：构建企业数据要素应用能力（DEA / Breadth / Depth）

**脚本：** `code/00_dea_construct.py`，依赖 `code/dea_module_bert.py` + `code/dea_module_rules.py` + `config/keywords_expanded.json`

#### 0.0 文本来源

从 CMDA 数据库获取上市公司 MD&A 年报文本（2011–2023），仅取 12-31 年报。目录结构：

```
<CMDA_DIR>/<年份>/文本/<stkcode>_<年份>-12-31.txt
```

#### 0.1 分句与预处理（Module C）

- 按句末标点（。！？；）和换行切分句子
- 基于哈工大停用词表计算实词密度，过滤财务表格行（实词密度 < 0.28）
- 剔除表格标志行（含「单位：元」「√适用」等）
- 超长句（>480 字）在逗号处二次切分，BERT 512 token 上限保护

#### 0.2 关键词预筛

对每句用关键词词典快速扫描（`quick_has_keyword`），过滤不含任何关键词的句子。关键词词典见 `config/keywords_expanded.json`：

| 维度 | 类别 | 核心词（core） | 辅助词（auxiliary） |
|------|------|---------------|---------------------|
| **广度** | 数据基础设施 | 服务器、数据中心、GPU、算力、云化… | — |
| | 数据处理与分析 | 并行计算、数据分析、数据挖掘… | — |
| | 数据存储与管理 | 数据仓库、数据湖、分布式存储… | — |
| | 数据安全与治理 | 数据加密、访问控制、数据治理… | — |
| | 业务应用场景 | 智能制造、智慧城市、数字营销… | 数字化、智能化、信息化… |
| **深度** | 战略定位 | 数据驱动、数据中台、数据资产化… | 数字化转型、数据战略… |
| | 组织保障 | 首席数据官、数据委员会… | — |
| | 数据治理体系 | 数据标准、数据质量、主数据管理… | 数据管理、信息管理… |
| | 技术能力深度 | 深度学习、NLP、知识图谱、数字孪生… | AI、人工智能、大数据… |
| | 业务融合深度 | 精准营销、智能风控、数据决策… | 技术赋能、互联网技术… |

> 关键词词典可通过 `code/00_dea_expand_keywords.py` 从种子词（`keywords.json`）用 Word2Vec Skip-Gram（dim=200, window=10, cos≥0.75）自动扩展。

#### 0.3 BERT 否定过滤（Module B）

两阶段精判，对应论文阈值 0.60：

**阶段一：正则预筛（`regex_prefilter`）**
- 明确否定（尚未部署、并未引入、不具备条件…）→ 直接排除
- 明确未来计划（计划/拟/将 + 建立/引入/构建…）→ 直接排除
- 处于探索/起步阶段（尚处于研究阶段…）→ 直接排除
- 含模糊否定词（未/不/没/无）→ 进入 BERT 精判
- 其余 → 直接保留

**阶段二：BERT 零样本 NLI 分类（`BertDeploymentClassifier`）**
- 模型：`IDEA-CCNL/Erlangshen-Roberta-110M-NLI`（中文 NLI，本地 `models/` 目录）
- 假设模板：
  - "该公司已实际部署或应用了相关数据技术"
  - "该描述仅为方向性表达或否定性陈述"
- 阈值：P(部署) ≥ 0.60 → 保留，否则过滤

#### 0.4 场景约束计数（Module C）

对通过 BERT 的句子，匹配关键词并分类计数：

1. 找句子中所有广度词/深度词，区分核心词（core）和辅助词（auxiliary）
2. **有核心词：** core + aux 全部计入
3. **仅有辅助词：** 检查是否为口号式笼统表述（「持续推进数字化转型」「不断提升智能化水平」等）→ 是则全部不计入；否则正常计入

#### 0.5 量化与输出

- **主指标（对数形式）：**
  - `Breadth = ln(广度词频 + 1)`
  - `Depth = ln(深度词频 + 1)`
  - `Dea = ln(总词频 + 1)`
- **稳健性指标（密度形式）：**
  - `Dea_count = 总词频 / 有效字数 × 1000`
  - 有效字数 = MD&A 文本剔除停用词、英文、数字、标点后的汉字数

输出：`output/dea_bert_zero-shot.csv`（已预计算为 `data/dea_original.dta`，无需重新运行）

---

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

#### 1.4 Sigma 密度峰聚类

- **子采样：** 从全样本中随机抽取 5,000 个观测值用于密度峰筛选
- **截断距离：** 距离矩阵上三角的 P2 分位数（`dc_percentile=2.0`）
- **局部密度 ρ_i：** 距离小于 dc 的点数
- **最小距离 δ_i：** 到更高密度点的最小距离
- **聚类得分 γ_i = ρ_i × δ_i：** 同时考虑密度和分离度
- **阈值：** γ ≥ μ + 2σ（`sigma_multiplier=2.0`）
- **去冗余：** 若两个聚类中心间距 < dc，仅保留密度更高的那个

#### 1.5 异常值剔除

- 计算每个样本到最近聚类中心的距离
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

### 额外稳健性 & 机制检验

**脚本：** `code/13_additional_tests.py`

在基准回归基础上进行 3 项额外检验：

| # | 检验 | 方法 | 关键结果 |
|---|------|------|----------|
| 1 | 融资约束缓解 | Dea → 债券/股权/商业信用融资 + 信贷可得性 | Dea ↑ 债券(+), 商业信用(+), 股权(-) |
| 2 | 现金流波动降低 | Dea → 现金流波动 | 不显著 (p=0.190) |
| 3 | 倒U型检验 | Dea + Dea² → res | 二次项显著负 (p=0.006), 转折点 Dea=0.69, 在样本内 |

所有回归均使用 PanelOLS 双向固定效应（企业 + 年份），聚类标准误。
输出：`output/additional_tests_results.csv`

## 项目结构

```
.
├── paper/                            原论文、附录、原始程序
├── config/
│   ├── keywords.json                 种子关键词（附录2）
│   ├── keywords_expanded.json        Word2Vec扩展词典
│   └── hit_stopwords.txt             哈工大停用词表
├── code/
│   ├── 00_dea_construct.py           DEA 变量构建管线（BERT + 规则）
│   ├── 00_dea_expand_keywords.py     关键词 Word2Vec 扩展
│   ├── dea_module_bert.py            Module B：BERT 否定过滤器
│   ├── dea_module_rules.py           Module C：场景约束 + 中文分句
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
│   └── 13_additional_tests.py         额外稳健性 & 机制检验
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
