"""
任务4 — PSM 倾向得分匹配
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

做法：
  1. 按 Dea 中位数将企业分为「高DEA组」和「低DEA组」
  2. Logit 模型估计企业被归入高DEA组的倾向得分
  3. 1:1 最近邻匹配（卡尺=0.05）
  4. 匹配平衡性检验（标准化偏差 < 10%）
  5. 匹配后样本重新估计基准回归

注意：
  PSM 处理的是选择性的截面差异，不是面板内的因果效应。
  论文使用 PSM 要回答的是：高DEA企业的供应链韧性是否显著高于
  具有相似特征但DEA较低的企业？
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

OLD_OUT = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "output"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(OLD_OUT / "panel_merged.csv")
df_raw["stkcode"] = df_raw["stkcode"].astype(str).str.zfill(6)

df = df_raw[df_raw["STPT"] != 1].copy()
excl = df["ind_str"].str.contains(
    "金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
df = df[~excl].copy()

for col in ["lnrd", "lev", "klr"]:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual",
            "lnrd", "indrate", "own", "lev"]
OUTCOME  = "res"
CORE     = "Dea"

# ── 构建 PSM 样本 ──────────────────────────────────────────────────────────────
def make_psm_sample(data):
    """以企业-年平均值为PSM的截面单位"""
    cols = [CORE] + CONTROLS + [OUTCOME, "stkcode", "year", "city_x", "ind_str"]
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})

    # 对企业取时间平均（PSM在截面做）
    firm_avg = d.groupby("stkcode").agg({
        CORE: "mean",
        OUTCOME: "mean",
        **{c: "mean" for c in CONTROLS if c in d.columns},
        "city": "first",
        "ind_str": "first",
    }).reset_index()
    return firm_avg.dropna()


psm_df = make_psm_sample(df)
print(f"PSM截面样本: {len(psm_df):,} 家企业")

# ── 处理组/控制组划分 ────────────────────────────────────────────────────────
median_dea = psm_df[CORE].median()
psm_df["treated"] = (psm_df[CORE] > median_dea).astype(int)
print(f"中位数Dea: {median_dea:.4f}")
print(f"处理组(高DEA): {psm_df['treated'].sum():,}")
print(f"控制组(低DEA): {(1-psm_df['treated']).sum():,}")

# ── Logit 倾向得分 ───────────────────────────────────────────────────────────
X_vars = [c for c in CONTROLS if c in psm_df.columns]
X = psm_df[X_vars].copy()
y = psm_df["treated"].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X_vars, index=X.index)

# Logit
logit = LogisticRegression(penalty=None, max_iter=1000)
logit.fit(X_scaled, y)
psm_df["pscore"] = logit.predict_proba(X_scaled)[:, 1]
psm_df = psm_df.reset_index(drop=True)  # 重置索引，方便后续匹配

print(f"\n倾向得分分布: min={psm_df['pscore'].min():.3f}, "
      f"max={psm_df['pscore'].max():.3f}, "
      f"mean={psm_df['pscore'].mean():.3f}")

# ── 1:1 最近邻匹配（带卡尺） ────────────────────────────────────────────────
def nearest_neighbor_matching(df, caliper=0.05):
    treated = df[df["treated"] == 1].copy()
    control = df[df["treated"] == 0].copy()

    matches = []
    control_used = set()

    # 随机排序避免顺序偏差
    treated = treated.sample(frac=1, random_state=42)

    for t_idx, t_row in treated.iterrows():
        t_ps = t_row["pscore"]

        # 只考虑未使用的控制组
        available = control[~control.index.isin(control_used)]
        if len(available) == 0:
            break

        # 计算距离
        available = available.copy()
        available["distance"] = np.abs(available["pscore"] - t_ps)

        # 卡尺内最近邻
        in_caliper = available[available["distance"] <= caliper]
        if len(in_caliper) == 0:
            continue  # 无合适匹配

        best_match = in_caliper.nsmallest(1, "distance").iloc[0]

        matches.append({
            "treated_idx": int(t_idx),
            "control_idx": int(best_match.name),
            "treated_ps": t_ps,
            "control_ps": best_match["pscore"],
            "distance": best_match["distance"]
        })
        control_used.add(best_match.name)

    return pd.DataFrame(matches)


matches = nearest_neighbor_matching(psm_df, caliper=0.05)
print(f"\n成功匹配: {len(matches):,} 对")
print(f"卡尺外(未匹配): {(psm_df['treated'] == 1).sum() - len(matches):,} 家企业")

# ── 构建匹配后样本 ───────────────────────────────────────────────────────────
matched_indices = (list(matches["treated_idx"]) +
                   list(matches["control_idx"]))

matched_psm = psm_df.loc[matched_indices].copy()
matched_stkcodes = matched_psm["stkcode"].unique()
print(f"匹配后截面: {len(matched_stkcodes):,} 家企业")

# 回到面板数据
matched_df = df[df["stkcode"].isin(matched_stkcodes)].copy()
print(f"匹配后面板: {len(matched_df):,} obs, {matched_df['stkcode'].nunique():,} 家企业")

# ── 平衡性检验 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("匹配平衡性检验（标准化偏差 %）")
print("=" * 80)
print(f"{'变量':12s} {'处理组均值':>12s} {'控制组均值':>12s} {'标准化偏差%':>12s} {'t检验p':>10s}")
print("-" * 65)

balance = []
for var in X_vars:
    t_mean = matched_psm[matched_psm["treated"] == 1][var].mean()
    c_mean = matched_psm[matched_psm["treated"] == 0][var].mean()
    t_std = matched_psm[matched_psm["treated"] == 1][var].std()
    c_std = matched_psm[matched_psm["treated"] == 0][var].std()
    pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
    bias = 100 * (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0

    t_stat, p_val = ttest_ind(
        matched_psm[matched_psm["treated"] == 1][var].dropna(),
        matched_psm[matched_psm["treated"] == 0][var].dropna()
    )

    flag = "⚠️" if abs(bias) > 10 else "✅"
    print(f"{flag} {var:<10s} {t_mean:>12.4f} {c_mean:>12.4f} {bias:>12.2f} {p_val:>10.4f}")

    # 也计算原始样本的偏差
    t_mean_raw = psm_df[psm_df["treated"] == 1][var].mean()
    c_mean_raw = psm_df[psm_df["treated"] == 0][var].mean()
    t_std_raw = psm_df[psm_df["treated"] == 1][var].std()
    c_std_raw = psm_df[psm_df["treated"] == 0][var].std()
    pooled_std_raw = np.sqrt((t_std_raw**2 + c_std_raw**2) / 2)
    bias_raw = 100 * (t_mean_raw - c_mean_raw) / pooled_std_raw if pooled_std_raw > 0 else 0

    balance.append({"变量": var,
                    "原始偏差%": bias_raw, "匹配后偏差%": bias,
                    "偏差降低%": 100 * (abs(bias_raw) - abs(bias)) / max(abs(bias_raw), 0.01)})

balance_df = pd.DataFrame(balance)
print(f"\n平均偏差降低: {balance_df['偏差降低%'].mean():.1f}%")
print(f"匹配后|偏差|<10%的变量: {(balance_df['匹配后偏差%'].abs() < 10).sum()}/{len(balance_df)}")

# ── 匹配后回归 ───────────────────────────────────────────────────────────────
def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clusters = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clusters)


def fmt(coef, se, pval):
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars} ({se:.4f})"


def make_reg_sample(data, core_var="Dea"):
    cols = [core_var] + CONTROLS + [OUTCOME, "stkcode", "year", "city_x"]
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})
    return d.dropna(subset=[core_var] + CONTROLS + [OUTCOME])


print("\n" + "=" * 80)
print("匹配后回归结果")
print("=" * 80)

for label, data, desc in [
    ("全样本(匹配前)", df, "匹配前全样本"),
    ("PSM匹配后", matched_df, "PSM 1:1卡尺匹配后"),
]:
    s = make_reg_sample(data)
    s = s.dropna(subset=["city"])
    print(f"\n{desc}: {len(s):,} obs")
    r = run_fe(s, OUTCOME, [CORE] + CONTROLS)
    print(f"  Dea: {fmt(r.params[CORE], r.std_errors[CORE], r.pvalues[CORE])}  "
          f"N={r.nobs:,}  R²(within)={r.rsquared_within:.4f}")

# ── 保存 ──────────────────────────────────────────────────────────────────────
balance_df.to_csv(OUT / "psm_balance.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已保存到 {OUT}/psm_balance.csv")
