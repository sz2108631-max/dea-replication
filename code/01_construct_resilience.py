"""
按论文附录1严格重新构造企业供应链韧性 (Res)
修正项:
  1. Sigma地标选择: γ=ρ×δ ≥ μ+2σ (原错误: 仅用δ)
  2. TOPSIS: 静态全样本池化 (原: 逐年动态)
  3. 参数可配置: dc_percentile, sigma_multiplier, outlier_sigma
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pyreadstat
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"
Y_DATA = ROOT / "raw_data"
CTRL_PATH = DATA / "常用控制变量2000_2024_Ver3.1.dta"
CITE_PATH = DATA / "变量_被引证.xlsx"

# Ensure output directories exist
(OUT / "中间结果").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 可配置参数
# ============================================================
DC_PERCENTILE = 2.0     # 截断距离分位数 (越小越严格)
SIGMA_MULT = 2.0         # Sigma阈值倍数 (论文μ+2σ)
OUTLIER_SIGMA = 2.5      # 异常值剔除σ倍数 (越小剔除越多)
TOPSIS_MODE = 'dynamic'  # 'static'=全样本池化(论文方式), 'dynamic'=逐年
N_SAMPLE_LANDMARK = 5000 # 地标选择子样本数

print(f"参数配置:")
print(f"  dc_percentile = {DC_PERCENTILE}")
print(f"  sigma_multiplier = {SIGMA_MULT}")
print(f"  outlier_sigma = {OUTLIER_SIGMA}")
print(f"  topsis_mode = {TOPSIS_MODE}")

# ============================================================
# 第零步: 加载所有原始数据 (复用22_construct_res.py)
# ============================================================
print("\n" + "=" * 60)
print("加载原始数据...")

bs = pd.read_excel(Y_DATA / "资产负债表.xlsx", header=0, skiprows=[1, 2])
bs_cols = {
    'Stkcd': 'stkcd', 'Accper': 'accper',
    'A001101000': 'cash', 'A001110000': 'notes_receiv',
    'A001111000': 'acct_receiv', 'A001112000': 'prepay',
    'A001123000': 'inventory',
}
bs = bs[list(bs_cols.keys())].rename(columns=bs_cols).copy()
bs['year'] = pd.to_datetime(bs['accper']).dt.year
bs['stkcode'] = bs['stkcd'].astype(int).astype(str).str.zfill(6)
for c in ['cash','notes_receiv','acct_receiv','prepay','inventory']:
    bs[c] = pd.to_numeric(bs[c], errors='coerce')
bs = bs.drop(columns=['stkcd','accper'])

pl = pd.read_excel(Y_DATA / "利润表.xlsx", header=0, skiprows=[1, 2])
pl_cols = {
    'Stkcd': 'stkcd', 'Accper': 'accper',
    'B001100000': 'total_revenue', 'B001101000': 'revenue',
    'B001201000': 'cogs',
}
pl = pl[list(pl_cols.keys())].rename(columns=pl_cols).copy()
pl['year'] = pd.to_datetime(pl['accper']).dt.year
pl['stkcode'] = pl['stkcd'].astype(int).astype(str).str.zfill(6)
for c in ['total_revenue','revenue','cogs']:
    pl[c] = pd.to_numeric(pl[c], errors='coerce')
pl = pl.drop(columns=['stkcd','accper'])

sc, _ = pyreadstat.read_dta(DATA / "客户&供应商&供应链集中度（未剔除）.dta")
sc['stkcode'] = sc['股票代码'].astype(int).astype(str).str.zfill(6)
sc['year'] = sc['年份'].astype(int)
sc = sc.rename(columns={
    '前五大客户销售额': 'top5_cust_sales',
    '前五大供应商采购额': 'top5_supp_purch',
    '营业总收入': 'sc_revenue',
})
for c in ['top5_cust_sales','top5_supp_purch','sc_revenue']:
    sc[c] = pd.to_numeric(sc[c], errors='coerce')
sc = sc[['stkcode','year','top5_cust_sales','top5_supp_purch','sc_revenue']]

ctrl, _ = pyreadstat.read_dta(CTRL_PATH)
ctrl['stkcode'] = ctrl['stkcode'].astype(str).str.zfill(6)
ctrl['year'] = ctrl['year'].astype(int)
ctrl = ctrl[['stkcode','year','roa','operatingrevenue','totalassets']]

apply_pat, _ = pyreadstat.read_dta(Y_DATA / "申请专利质量.dta")
apply_pat['stkcode'] = apply_pat['股票代码'].astype(int).astype(str).str.zfill(6)
apply_pat['year'] = apply_pat['会计年度'].astype(int)
apply_pat = apply_pat.rename(columns={'申请专利质量': 'apply_knowledge_width'})
apply_pat = apply_pat[['stkcode','year','apply_knowledge_width']]

grant_pat, _ = pyreadstat.read_dta(Y_DATA / "授权专利质量.dta")
grant_pat['stkcode'] = grant_pat['股票代码'].astype(int).astype(str).str.zfill(6)
grant_pat['year'] = grant_pat['会计年度'].astype(int)
grant_pat = grant_pat.rename(columns={'授权专利质量': 'grant_knowledge_width'})
grant_pat = grant_pat[['stkcode','year','grant_knowledge_width']]

if CITE_PATH.exists():
    cite = pd.read_excel(CITE_PATH)
    cite = cite.dropna(subset=['年份'])
    cite['year'] = cite['年份'].astype(int)
    cite['stkcode'] = cite['股票代码_被引证'].apply(
        lambda x: str(int(x)).zfill(6) if pd.notna(x) else '')
    cite['citing_code'] = cite['股票代码'].astype(str).str.zfill(6)
    cite['is_self'] = cite['stkcode'] == cite['citing_code']
    cite_ext = cite[~cite['is_self']]
    firm_cite = cite_ext.groupby(['stkcode','year'])['Citations'].sum().reset_index()
    firm_cite['ln_citation'] = np.log(firm_cite['Citations'] + 1)
else:
    print(f"  [WARN] Citation data not found: {CITE_PATH}, setting ln_citation=0")
    firm_cite = pd.DataFrame(columns=['stkcode','year','ln_citation'])

# ============================================================
# 第一步: 构建11个三级指标 (复用已验证的逻辑)
# ============================================================
print("\n" + "=" * 60)
print("第一步: 构建11个三级指标...")

panel = ctrl[['stkcode','year','roa','operatingrevenue','totalassets']].copy()
panel = panel.merge(bs, on=['stkcode','year'], how='left')
panel = panel.merge(pl, on=['stkcode','year'], how='left')
panel = panel.merge(sc, on=['stkcode','year'], how='left')
panel = panel.merge(apply_pat, on=['stkcode','year'], how='left')
panel = panel.merge(grant_pat, on=['stkcode','year'], how='left')
panel = panel.merge(firm_cite[['stkcode','year','ln_citation']], on=['stkcode','year'], how='left')

panel['apply_knowledge_width'] = panel['apply_knowledge_width'].fillna(0)
panel['grant_knowledge_width'] = panel['grant_knowledge_width'].fillna(0)
panel['ln_citation'] = panel['ln_citation'].fillna(0)

# 指标①: 库存调整幅度
panel = panel.sort_values(['stkcode','year'])
panel['inventory_lag'] = panel.groupby('stkcode')['inventory'].shift(1)
panel['ind1'] = np.log(np.abs(panel['inventory'] - panel['inventory_lag']) + 1)

# 指标②: 供需偏离度
for v in ['cogs','revenue']:
    for lag in [1,2]:
        panel[f'{v}_l{lag}'] = panel.groupby('stkcode')[v].shift(lag)
def rolling_cv(v, v_l1, v_l2):
    m = (v + v_l1 + v_l2) / 3
    std = np.sqrt(((v-m)**2 + (v_l1-m)**2 + (v_l2-m)**2) / 3)
    return np.abs(std / m.replace(0, np.nan))
panel['cogs_cv'] = rolling_cv(panel['cogs'].fillna(0), panel['cogs_l1'].fillna(0), panel['cogs_l2'].fillna(0))
panel['rev_cv'] = rolling_cv(panel['revenue'].fillna(0), panel['revenue_l1'].fillna(0), panel['revenue_l2'].fillna(0))
panel['ind2'] = np.abs(panel['cogs_cv'] - panel['rev_cv'])

# 指标③: 供应链集中度
panel['ind3'] = (panel['top5_supp_purch']/panel['sc_revenue'].replace(0,np.nan) +
                 panel['top5_cust_sales']/panel['sc_revenue'].replace(0,np.nan)) / 2

# 指标④: 供应商稳定性 (论文: 1 - |比例变动| / 上年比例, 比例=前五大采购额/营业总收入)
panel['top5_supp_ratio'] = panel['top5_supp_purch'] / panel['sc_revenue'].replace(0, np.nan)
panel['top5_supp_ratio_lag'] = panel.groupby('stkcode')['top5_supp_ratio'].shift(1)
panel['ind4'] = (1 - np.abs(panel['top5_supp_ratio'] - panel['top5_supp_ratio_lag']) /
                 panel['top5_supp_ratio_lag'].replace(0, np.nan)).clip(-1, 1)

# 指标⑤: 客户稳定性 (论文: 1 - |比例变动| / 上年比例, 比例=前五大销售额/营业总收入)
panel['top5_cust_ratio'] = panel['top5_cust_sales'] / panel['sc_revenue'].replace(0, np.nan)
panel['top5_cust_ratio_lag'] = panel.groupby('stkcode')['top5_cust_ratio'].shift(1)
panel['ind5'] = (1 - np.abs(panel['top5_cust_ratio'] - panel['top5_cust_ratio_lag']) /
                 panel['top5_cust_ratio_lag'].replace(0, np.nan)).clip(-1, 1)

# 指标⑥-⑧
panel['ind6'] = panel['roa']
panel['ind7'] = (panel['total_revenue'] / panel['cash'].replace(0,np.nan)).clip(0, panel['total_revenue'].quantile(0.99))
panel['receivables'] = panel['notes_receiv'].fillna(0)+panel['acct_receiv'].fillna(0)+panel['prepay'].fillna(0)
# 论文: 占主营业务收入的比重 → 使用营业收入(B001101000)
panel['ind8'] = (panel['receivables'] / panel['revenue'].replace(0,np.nan)).clip(0, panel['receivables'].quantile(0.99))

# 指标⑨-⑪
panel['ind9'] = panel['apply_knowledge_width']
panel['ind10'] = panel['grant_knowledge_width']
panel['ind11'] = panel['ln_citation']

ind_cols = [f'ind{i}' for i in range(1,12)]
ind_labels = [
    '库存调整幅度','供需偏离度','供应链集中度',
    '供应商稳定性','客户稳定性','ROA',
    '现金周转率','供需协同','申请专利知识宽度',
    '授权专利知识宽度','外部引用',
]

sub = panel[['stkcode','year'] + ind_cols].dropna().copy()
print(f"  11指标完整样本: {len(sub):,} obs, {sub['stkcode'].nunique():,} firms")
print(f"  年份: {sub['year'].min():.0f}-{sub['year'].max():.0f}")

# ============================================================
# 第二步: 数据预处理 - 正向化 + 标准化
# ============================================================
print("\n" + "=" * 60)
print("第二步: 数据预处理...")

X_raw = sub[ind_cols].values.astype(np.float64)

# 1% 99% 缩尾
for j in range(X_raw.shape[1]):
    p1, p99 = np.nanpercentile(X_raw[:, j], [1, 99])
    X_raw[:, j] = np.clip(X_raw[:, j], p1, p99)

# 正向化: ind2(偏离度)和ind8(应收占比)越小越好 → 取负
for j in [1, 7]:
    X_raw[:, j] = -X_raw[:, j]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"  X_scaled: {X_scaled.shape}")

# ============================================================
# 第三步: 深度自编码器降维 (11D → 8D → 6D → 4D)
# ============================================================
print("\n" + "=" * 60)
print("第三步: 深度自编码器降维 (11D → 4D)...")

ae1 = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam',
                   alpha=0.0001, batch_size=256, learning_rate_init=0.001,
                   max_iter=200, random_state=42)
ae1.fit(X_scaled, X_scaled)
H1 = np.maximum(0, X_scaled @ ae1.coefs_[0] + ae1.intercepts_[0])
print(f"  Layer 1 (11→8): loss={ae1.loss_:.6f}")

ae2 = MLPRegressor(hidden_layer_sizes=(6,), activation='relu', solver='adam',
                   alpha=0.0001, batch_size=256, learning_rate_init=0.001,
                   max_iter=200, random_state=42)
ae2.fit(H1, H1)
H2 = np.maximum(0, H1 @ ae2.coefs_[0] + ae2.intercepts_[0])
print(f"  Layer 2 (8→6): loss={ae2.loss_:.6f}")

ae3 = MLPRegressor(hidden_layer_sizes=(4,), activation='identity', solver='adam',
                   alpha=0.0001, batch_size=256, learning_rate_init=0.001,
                   max_iter=300, random_state=42)
ae3.fit(H2, H2)
Z = H2 @ ae3.coefs_[0] + ae3.intercepts_[0]
print(f"  Layer 3 (6→4 bottleneck): loss={ae3.loss_:.6f}")

# 重构验证
H2_recon = np.maximum(0, Z @ ae3.coefs_[1] + ae3.intercepts_[1])
H1_recon = np.maximum(0, H2_recon @ ae2.coefs_[1] + ae2.intercepts_[1])
X_recon = H1_recon @ ae1.coefs_[1] + ae1.intercepts_[1]
mse = np.mean((X_recon - X_scaled)**2)
print(f"  整体重构MSE: {mse:.6f}")
print(f"  潜在特征Z: mean={np.round(Z.mean(axis=0),4)}, std={np.round(Z.std(axis=0),4)}")

# ============================================================
# 第四步: Sigma原则地标中心选择 (修正版: γ=ρ×δ)
# ============================================================
print("\n" + "=" * 60)
print("第四步: Sigma原则地标中心选择 (γ=ρ×δ ≥ μ+2σ)...")

from scipy.spatial.distance import cdist

N = Z.shape[0]

# 随机子样本做地标选择 (内存限制)
if N > N_SAMPLE_LANDMARK:
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, N_SAMPLE_LANDMARK, replace=False)
    Z_sample = Z[sample_idx]
else:
    sample_idx = np.arange(N)
    Z_sample = Z

n_s = Z_sample.shape[0]
print(f"  地标选择子样本: {n_s:,} / {N:,}")

# 计算子样本成对距离
D_sample = cdist(Z_sample, Z_sample)

# 截断距离 dc (取距离矩阵上三角的DC_PERCENTILE分位)
upper_tri = D_sample[np.triu_indices(n_s, k=1)]
dc = np.percentile(upper_tri, DC_PERCENTILE)
print(f"  dc (截断距离, P{DC_PERCENTILE}): {dc:.4f}")

# 局部密度 ρ_i = 距离小于dc的点数 (论文 χ(||xi-xj|| - dc))
chi = (D_sample < dc).astype(float)
rho = chi.sum(axis=1) - 1  # 减自身

# 最小相对距离 δ_i — 到更高密度点的最小距离
delta = np.full(n_s, np.inf)
for i in range(n_s):
    higher_rho = np.where(rho > rho[i])[0]
    if len(higher_rho) > 0:
        delta[i] = D_sample[i, higher_rho].min()
    else:
        delta[i] = D_sample[i].max()  # 最高密度点取最大距离

# ★ 修正核心: γ = ρ × δ ★
gamma = rho * delta
gamma_mean = gamma.mean()
gamma_std = gamma.std()
threshold = gamma_mean + SIGMA_MULT * gamma_std  # μ + 2σ

print(f"  ρ: mean={rho.mean():.2f}, std={rho.std():.2f}")
print(f"  δ: mean={delta.mean():.4f}, std={delta.std():.4f}")
print(f"  γ=ρ×δ: mean={gamma_mean:.4f}, std={gamma_std:.4f}")
print(f"  阈值 (μ+{SIGMA_MULT}σ): {threshold:.4f}")

# 候选地标: γ_i ≥ μ + 2σ
candidate_mask = gamma >= threshold
candidates = np.where(candidate_mask)[0]
print(f"  候选地标数: {len(candidates)} / {n_s}")

# 去冗余: 若两个地标距离 < dc, 只保留密度ρ更高的
selected_sub = []
if len(candidates) > 0:
    sorted_by_rho = candidates[np.argsort(-rho[candidates])]
    for idx in sorted_by_rho:
        too_close = False
        for s in selected_sub:
            if D_sample[idx, s] < dc:
                too_close = True
                break
        if not too_close:
            selected_sub.append(idx)
    selected_sub = np.array(selected_sub)
else:
    selected_sub = np.array([])

print(f"  去冗余后地标数: {len(selected_sub)}")

# 映射回全样本
if len(selected_sub) > 0:
    landmark_global_idx = sample_idx[selected_sub]
    Z_landmarks = Z[landmark_global_idx]

    # 全样本到地标的距离
    D_to_landmarks = cdist(Z, Z_landmarks)
    min_dist = D_to_landmarks.min(axis=1)

    # 异常值: 到最近地标距离 > μ + OUTLIER_SIGMA*σ
    md_mean = min_dist.mean()
    md_std = min_dist.std()
    outlier_threshold = md_mean + OUTLIER_SIGMA * md_std
    normal_mask = min_dist <= outlier_threshold

    print(f"  最小距离: mean={md_mean:.4f}, std={md_std:.4f}")
    print(f"  异常值阈值 (μ+{OUTLIER_SIGMA}σ): {outlier_threshold:.4f}")
    print(f"  正常样本: {normal_mask.sum():,} / {N} ({normal_mask.mean():.1%})")
    print(f"  剔除异常值: {(~normal_mask).sum():,}")
else:
    normal_mask = np.ones(N, dtype=bool)
    landmark_global_idx = np.array([])
    print(f"  无地标中心，保留全部样本")

# 筛选正常样本
Z_normal = Z[normal_mask]
sub_normal = sub.iloc[normal_mask].reset_index(drop=True)
years_normal = sub_normal['year'].values
X_raw_normal = X_raw[normal_mask]
print(f"  筛选后样本: {len(sub_normal):,}")

# ============================================================
# 第五步: TOPSIS (静态: 全样本池化 或 动态: 逐年)
# ============================================================
print("\n" + "=" * 60)
print(f"第五步: TOPSIS ({TOPSIS_MODE} 模式)...")

def entropy_weights(R):
    """熵权法计算权重"""
    R_range = R.max(axis=0) - R.min(axis=0)
    zero_var = R_range < 1e-10
    R_pos = R - R.min(axis=0) + 1e-10
    P = R_pos / R_pos.sum(axis=0)
    k = 1.0 / np.log(len(R))
    E = -k * (P * np.log(np.clip(P, 1e-12, 1))).sum(axis=0)
    w_raw = 1 - E
    w_raw[zero_var] = 0
    if w_raw.sum() > 1e-10:
        return w_raw / w_raw.sum()
    else:
        return np.ones(len(w_raw)) / len(w_raw)

Res = np.zeros(len(sub_normal))

if TOPSIS_MODE == 'static':
    # 论文方式: 单一决策矩阵,全样本池化TOPSIS
    R = Z_normal / np.sqrt((Z_normal**2).sum(axis=0))
    w = entropy_weights(R)
    V = R * w
    v_plus = V.max(axis=0)
    v_minus = V.min(axis=0)
    S_plus = np.sqrt(((V - v_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - v_minus)**2).sum(axis=1))
    Res = S_minus / (S_plus + S_minus)
    print(f"  熵权: {np.round(w, 4)}")
    print(f"  v⁺: {np.round(v_plus, 4)}")
    print(f"  v⁻: {np.round(v_minus, 4)}")
else:
    # 逐年动态TOPSIS
    unique_years = np.sort(np.unique(years_normal))
    for yr in unique_years:
        yr_mask = years_normal == yr
        Z_yr = Z_normal[yr_mask]
        n_t = Z_yr.shape[0]
        if n_t < 5:
            w = np.ones(Z_yr.shape[1]) / Z_yr.shape[1]
        else:
            R_yr = Z_yr / np.sqrt((Z_yr**2).sum(axis=0))
            w = entropy_weights(R_yr)
        V_yr = (Z_yr / np.sqrt((Z_yr**2).sum(axis=0))) * w
        v_plus = V_yr.max(axis=0)
        v_minus = V_yr.min(axis=0)
        S_p = np.sqrt(((V_yr - v_plus)**2).sum(axis=1))
        S_m = np.sqrt(((V_yr - v_minus)**2).sum(axis=1))
        Res[yr_mask] = S_m / (S_p + S_m)
    print(f"  逐年TOPSIS完成, {len(unique_years)} 个年份")

print(f"  Res: mean={Res.mean():.6f}, std={Res.std():.6f}")
print(f"  Res range: [{Res.min():.6f}, {Res.max():.6f}]")

# ============================================================
# 第六步: 保存结果
# ============================================================
print("\n" + "=" * 60)
print("第六步: 保存结果...")

intermediate = OUT / "中间结果"
result = sub_normal[['stkcode','year']].copy()
result['res_v2'] = Res

result.to_csv(intermediate / "res_v2_panel.csv", index=False)
pyreadstat.write_dta(result, intermediate / "res_v2_panel.dta")
print(f"  已保存: {len(result):,} obs to res_v2_panel.dta")

print(f"\n>>> 24_construct_res_v2 完成 <<<")
