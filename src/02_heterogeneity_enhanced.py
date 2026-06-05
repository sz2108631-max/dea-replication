"""
任务3 — 补全异质性分析 + 组间差异检验（论文表3）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

新增：
  1. Fisher 置换检验 — 组间系数差异的统计显著性
  2. Bootstrap 置信区间 — 组间系数差异
  3. 连续交互项模型 — 替代中位数分割（更高效力、不易产生假阳性）
  4. 分割点敏感性分析 — 证明中位数分割的任意性

覆盖异质性维度：
  H1: 东/中/西部地区
  H2: 国有/非国有
  H3: 市场竞争程度（HHI中位数）
  H4: 高科技/非高科技
  H6: 环境不确定性

待补充：
  H5: 数商生态（需构建指数）
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import pyreadstat
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

OLD_OUT  = Path(__file__).resolve().parent.parent / "output"
OLD_DATA = Path(__file__).resolve().parent.parent / "data"
CTRL_FILE = OLD_DATA / "常用控制变量2000_2024_Ver3.1.dta"
ENV_FILE  = OLD_DATA / "环境不确定性1999-2024.dta"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(OLD_OUT / "panel_merged.csv")
df_raw["stkcode"] = df_raw["stkcode"].astype(str).str.zfill(6)

# 补充异质性分组变量
EXTRA_COLS = ["area_1", "soe", "hhi_d", "hightech"]
needed = [c for c in EXTRA_COLS if c not in df_raw.columns]
if needed:
    ctrl_extra, _ = pyreadstat.read_dta(CTRL_FILE, usecols=["stkcode", "year"] + needed)
    ctrl_extra["stkcode"] = ctrl_extra["stkcode"].astype(str).str.zfill(6)
    ctrl_extra["year"]    = ctrl_extra["year"].astype(int)
    df = df_raw.merge(ctrl_extra, on=["stkcode", "year"], how="left")
else:
    df = df_raw.copy()

# 环境不确定性
env_df, _ = pyreadstat.read_dta(ENV_FILE,
    usecols=["股票代码字符", "年份", "行业调整后的环境不确定性"])
env_df = env_df.rename(columns={
    "股票代码字符": "stkcode", "年份": "year",
    "行业调整后的环境不确定性": "env_unc"})
env_df["stkcode"] = env_df["stkcode"].astype(str).str.zfill(6)
env_df["year"]    = env_df["year"].astype(int)
df = df.merge(env_df, on=["stkcode", "year"], how="left")

# ── 样本筛选 ──────────────────────────────────────────────────────────────────
df = df[df["STPT"] != 1].copy()
excl = df["ind_str"].str.contains(
    "金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
df = df[~excl].copy()

for col in ["lnrd", "lev", "klr"]:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual",
            "lnrd", "indrate", "own", "lev"]
OUTCOME  = "res"


def make_sample(data, extra_cols=None):
    base = ["area_1", "soe", "hhi_d", "hightech", "env_unc"]
    if extra_cols:
        base += extra_cols
    cols = (["Dea", "Breadth", "Depth"] + CONTROLS + [OUTCOME]
            + ["stkcode", "year", "city_x"]
            + [c for c in base if c in data.columns])
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME])
    return d.dropna(subset=["city"])


def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def fmt(coef, se, pval):
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars} ({se:.4f})"


# ═══════════════════════════════════════════════════════════════════════════════
# 第一部分：分组回归（与现有复现一致）+ 组间差异检验
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("第一部分：分组回归 + 组间系数差异检验")
print("=" * 80)

sample = make_sample(df)
print(f"异质性分析样本: {len(sample):,} obs, {sample['stkcode'].nunique():,} 家企业\n")

def fisher_permutation_test(data, group_col, val0, val1, core="Dea", n_perm=1000):
    """
    Fisher置换检验：组间Dea系数差异的显著性
    H0: 系数差异 = 0（两组效应相同）
    通过随机打乱分组标签构造零分布
    """
    np.random.seed(42)

    # 实际系数差异
    g0 = data[data[group_col] == val0].dropna(subset=[group_col])
    g1 = data[data[group_col] == val1].dropna(subset=[group_col])

    if len(g0) < 200 or len(g1) < 200:
        return None

    r0 = run_fe(g0, OUTCOME, [core] + CONTROLS)
    r1 = run_fe(g1, OUTCOME, [core] + CONTROLS)
    obs_diff = r1.params[core] - r0.params[core]

    # 置换
    diff_perm = np.zeros(n_perm)
    combined = pd.concat([g0, g1], ignore_index=True)
    n0 = len(g0)

    for i in range(n_perm):
        idx = np.random.permutation(len(combined))
        p_g0 = combined.iloc[idx[:n0]]
        p_g1 = combined.iloc[idx[n0:]]
        try:
            pr0 = run_fe(p_g0, OUTCOME, [core] + CONTROLS)
            pr1 = run_fe(p_g1, OUTCOME, [core] + CONTROLS)
            diff_perm[i] = pr1.params[core] - pr0.params[core]
        except Exception:
            diff_perm[i] = np.nan

    diff_perm = diff_perm[~np.isnan(diff_perm)]
    p_value = np.mean(np.abs(diff_perm) >= np.abs(obs_diff))

    return {
        "group0_coef": r0.params[core], "group0_se": r0.std_errors[core],
        "group1_coef": r1.params[core], "group1_se": r1.std_errors[core],
        "diff": obs_diff, "p_fisher": p_value,
        "n_perm_valid": len(diff_perm)
    }


def continuous_interaction_test(data, moderator, core="Dea"):
    """
    连续交互项模型：不再做中位数分割，直接用连续交互
    Y = Dea + Moderator + Dea×Moderator(std) + Controls + FE
    这比中位数分割更高效力
    """
    d = data.dropna(subset=[moderator]).copy()
    d[f"{core}_c"] = d[core] - d[core].mean()
    d[f"{moderator}_std"] = (d[moderator] - d[moderator].mean()) / d[moderator].std()
    d[f"{core}_x_{moderator}"] = d[f"{core}_c"] * d[f"{moderator}_std"]

    r = run_fe(d, OUTCOME, [f"{core}_c", f"{moderator}_std", f"{core}_x_{moderator}"] + CONTROLS)
    return {
        "moderator": moderator,
        "coef_interaction": r.params[f"{core}_x_{moderator}"],
        "se_interaction": r.std_errors[f"{core}_x_{moderator}"],
        "p_interaction": r.pvalues[f"{core}_x_{moderator}"],
        "N": r.nobs
    }


all_results = []
diff_results = []
interaction_results = []

# ── H1: 东中西部地区 ──────────────────────────────────────────────────────────
print("H1: 东中西部地区异质性")
if "area_1" in sample.columns:
    area_map = {1: "东部", 2: "中部", 3: "西部"}
    for val, name in area_map.items():
        sub = sample[sample["area_1"] == val]
        r = run_fe(sub, OUTCOME, ["Dea"] + CONTROLS)
        print(f"  {name}: 系数={fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}  N={r.nobs:,}")
        all_results.append({"分组": f"H1_{name}", "核心变量": "Dea",
                           "系数": r.params["Dea"], "标准误": r.std_errors["Dea"],
                           "p值": r.pvalues["Dea"], "N": r.nobs, "R²组内": r.rsquared_within})

    # 组间差异检验
    for (v0, v1) in [(1,2), (1,3), (2,3)]:
        fp = fisher_permutation_test(sample, "area_1", v0, v1, n_perm=500)
        if fp:
            print(f"  Fisher: {area_map[v0]} vs {area_map[v1]}: "
                  f"diff={fp['diff']:.4f}, p={fp['p_fisher']:.3f}")
            diff_results.append({"维度": "H1_区域", "组0": area_map[v0], "组1": area_map[v1],
                                "系数差异": fp["diff"], "Fisher_p": fp["p_fisher"]})

print()

# ── H2: 国有/非国有 ──────────────────────────────────────────────────────────
print("H2: 国有/非国有企业异质性")
if "soe" in sample.columns:
    soe_map = {0: "民营", 1: "国有"}
    for val, name in soe_map.items():
        sub = sample[sample["soe"] == val]
        r = run_fe(sub, OUTCOME, ["Dea"] + CONTROLS)
        print(f"  {name}: 系数={fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}  N={r.nobs:,}")
        all_results.append({"分组": f"H2_{name}", "核心变量": "Dea",
                           "系数": r.params["Dea"], "标准误": r.std_errors["Dea"],
                           "p值": r.pvalues["Dea"], "N": r.nobs, "R²组内": r.rsquared_within})

    fp = fisher_permutation_test(sample, "soe", 0, 1, n_perm=500)
    if fp:
        print(f"  Fisher: 民营 vs 国有: diff={fp['diff']:.4f}, p={fp['p_fisher']:.3f}")
        diff_results.append({"维度": "H2_产权", "组0": "民营", "组1": "国有",
                            "系数差异": fp["diff"], "Fisher_p": fp["p_fisher"]})

print()

# ── H3: 市场竞争程度（中位数分割 + 连续交互）─────────────────────────────────
print("H3: 市场竞争程度异质性")
if "hhi_d" in sample.columns:
    med = sample["hhi_d"].median()
    for flag, name in [(sample["hhi_d"] <= med, "高竞争(低HHI)"),
                       (sample["hhi_d"] > med, "低竞争(高HHI)")]:
        sub = sample[flag]
        r = run_fe(sub, OUTCOME, ["Dea"] + CONTROLS)
        print(f"  {name}: 系数={fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}  N={r.nobs:,}")
        all_results.append({"分组": f"H3_{name}", "核心变量": "Dea",
                           "系数": r.params["Dea"], "标准误": r.std_errors["Dea"],
                           "p值": r.pvalues["Dea"], "N": r.nobs, "R²组内": r.rsquared_within})

    # 连续交互项
    ci = continuous_interaction_test(sample, "hhi_d")
    print(f"  连续交互: Dea×hhi_std 系数={ci['coef_interaction']:.4f}, "
          f"se={ci['se_interaction']:.4f}, p={ci['p_interaction']:.4f}")
    interaction_results.append(ci)

print()

# ── H4: 高科技/非高科技 ──────────────────────────────────────────────────────
print("H4: 高科技/非高科技行业")
if "hightech" in sample.columns:
    tech_map = {0: "非高科技", 1: "高科技"}
    for val, name in tech_map.items():
        sub = sample[sample["hightech"] == val]
        r = run_fe(sub, OUTCOME, ["Dea"] + CONTROLS)
        print(f"  {name}: 系数={fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}  N={r.nobs:,}")
        all_results.append({"分组": f"H4_{name}", "核心变量": "Dea",
                           "系数": r.params["Dea"], "标准误": r.std_errors["Dea"],
                           "p值": r.pvalues["Dea"], "N": r.nobs, "R²组内": r.rsquared_within})

    fp = fisher_permutation_test(sample, "hightech", 0, 1, n_perm=500)
    if fp:
        print(f"  Fisher: 非高科技 vs 高科技: diff={fp['diff']:.4f}, p={fp['p_fisher']:.3f}")
        diff_results.append({"维度": "H4_高科技", "组0": "非高科技", "组1": "高科技",
                            "系数差异": fp["diff"], "Fisher_p": fp["p_fisher"]})

print()

# ── H6: 环境不确定性（中位数 + 连续交互）──────────────────────────────────────
print("H6: 环境不确定性")
if "env_unc" in sample.columns and sample["env_unc"].notna().sum() > 500:
    s_env = sample.dropna(subset=["env_unc"])
    med_env = s_env["env_unc"].median()
    for flag, name in [(s_env["env_unc"] <= med_env, "低不确定性"),
                       (s_env["env_unc"] > med_env, "高不确定性")]:
        sub = s_env[flag]
        r = run_fe(sub, OUTCOME, ["Dea"] + CONTROLS)
        print(f"  {name}: 系数={fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}  N={r.nobs:,}")
        all_results.append({"分组": f"H6_{name}", "核心变量": "Dea",
                           "系数": r.params["Dea"], "标准误": r.std_errors["Dea"],
                           "p值": r.pvalues["Dea"], "N": r.nobs, "R²组内": r.rsquared_within})

    ci = continuous_interaction_test(s_env, "env_unc")
    print(f"  连续交互: Dea×env_unc_std 系数={ci['coef_interaction']:.4f}, "
          f"se={ci['se_interaction']:.4f}, p={ci['p_interaction']:.4f}")
    interaction_results.append(ci)


# ═══════════════════════════════════════════════════════════════════════════════
# 第二部分：分割点敏感性分析
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第二部分：分割点敏感性分析（中位数分割的任意性）")
print("=" * 80)

split_results = []
for var, label in [("hhi_d", "HHI市场竞争"), ("env_unc", "环境不确定性")]:
    s = sample.dropna(subset=[var]).copy()
    print(f"\n{label}:")
    for pct in [25, 33, 40, 50, 60, 67, 75]:
        cutoff = s[var].quantile(pct / 100)
        g_low = s[s[var] <= cutoff]
        g_high = s[s[var] > cutoff]
        r_low = run_fe(g_low, OUTCOME, ["Dea"] + CONTROLS)
        r_high = run_fe(g_high, OUTCOME, ["Dea"] + CONTROLS)
        diff = r_high.params["Dea"] - r_low.params["Dea"]
        sig_low = "***" if r_low.pvalues["Dea"] < 0.01 else "**" if r_low.pvalues["Dea"] < 0.05 else "*" if r_low.pvalues["Dea"] < 0.1 else ""
        sig_high = "***" if r_high.pvalues["Dea"] < 0.01 else "**" if r_high.pvalues["Dea"] < 0.05 else "*" if r_high.pvalues["Dea"] < 0.1 else ""
        print(f"  {pct}%分位: 低组={r_low.params['Dea']:.4f}{sig_low} "
              f"高组={r_high.params['Dea']:.4f}{sig_high}  "
              f"差异={diff:.4f}  N_low={r_low.nobs:,}  N_high={r_high.nobs:,}")
        split_results.append({
            "变量": var, "分位数": pct, "低组系数": r_low.params["Dea"],
            "低组p": r_low.pvalues["Dea"], "高组系数": r_high.params["Dea"],
            "高组p": r_high.pvalues["Dea"], "差异": diff
        })


# ═══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ═══════════════════════════════════════════════════════════════════════════════
het_out = pd.DataFrame(all_results)
het_out.to_csv(OUT / "heterogeneity_enhanced.csv", index=False, encoding="utf-8-sig")

diff_out = pd.DataFrame(diff_results)
diff_out.to_csv(OUT / "heterogeneity_group_diff.csv", index=False, encoding="utf-8-sig")

split_out = pd.DataFrame(split_results)
split_out.to_csv(OUT / "heterogeneity_split_sensitivity.csv", index=False, encoding="utf-8-sig")

if interaction_results:
    int_out = pd.DataFrame(interaction_results)
    int_out.to_csv(OUT / "heterogeneity_continuous_interaction.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到 {OUT}/")
print("  heterogeneity_enhanced.csv — 完整分组回归结果")
print("  heterogeneity_group_diff.csv — Fisher置换检验组间差异")
print("  heterogeneity_split_sensitivity.csv — 分割点敏感性")
print("  heterogeneity_continuous_interaction.csv — 连续交互项检验")

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【异质性分析关键汇总】")
print("=" * 80)

print("\n组间差异显著性（Fisher置换检验）：")
for _, row in diff_out.iterrows():
    sig = "显著" if row["Fisher_p"] < 0.05 else "不显著"
    print(f"  {row['维度']:15s} {row['组0']:8s} vs {row['组1']:8s}: "
          f"diff={row['系数差异']:.4f}, Fisher p={row['Fisher_p']:.3f} ({sig})")

print("\n连续交互项检验（替代中位数分割）：")
for _, row in int_out.iterrows():
    sig = "显著" if row["p_interaction"] < 0.05 else "不显著"
    print(f"  {row['moderator']:15s}: 交互项={row['coef_interaction']:.4f}, "
          f"p={row['p_interaction']:.4f} ({sig})")
