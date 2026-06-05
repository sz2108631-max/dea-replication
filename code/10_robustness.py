"""
补全论文剩余复现内容
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

四项补充：
  1. 表1逐步加控制变量（论文格式）
  2. 倒U型检验 Dea²（论文表5）
  3. PPML回归（稳健性7）
  4. 描述性统计表（论文附录格式）
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

OLD_OUT = Path(__file__).resolve().parent.parent / "output"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(OLD_OUT / "panel_merged.csv")
df["stkcode"] = df["stkcode"].astype(str).str.zfill(6)

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


def make_sample(data):
    cols = (["Dea", "Breadth", "Depth", "Dea_count", "Breadth_count", "Depth_count",
             "Dea_raw", "Breadth_raw", "Depth_raw"]
            + CONTROLS + [OUTCOME, "stkcode", "year", "city_x", "ind_str"])
    available = [c for c in cols if c in data.columns]
    d = data[available].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME, "city"])
    return d


def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def fmt(coef, se, pval):
    if pd.isna(coef) or pd.isna(pval):
        return "N/A"
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars} ({se:.4f})"


s = make_sample(df)
print(f"样本: {len(s):,} obs, {s['stkcode'].nunique():,} firms\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 表1 逐步加控制变量
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("1. 表1 逐步加控制变量（论文格式）")
print("=" * 90)

# 控制变量分组（模拟论文表1结构）
ctl_groups = {
    "(1) 无控制": [],
    "(2) + 公司特征": ["lnage", "lnsize", "lev"],
    "(3) + 治理结构": ["lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own"],
    "(4) + 研发投入": ["lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own", "lnrd"],
    "(5) + 资本劳动比": CONTROLS,  # 全部控制变量
}

results_stepwise = []
for label, ctls in ctl_groups.items():
    indep = ["Dea"] + ctls if ctls else ["Dea"]
    r = run_fe(s, OUTCOME, indep)

    results_stepwise.append({
        "模型": label,
        "Dea系数": r.params["Dea"],
        "Dea标准误": r.std_errors["Dea"],
        "Dea_p值": r.pvalues["Dea"],
        "N": r.nobs,
        "R²_within": r.rsquared_within if hasattr(r, 'rsquared_within') else np.nan,
        "控制变量数": len(ctls),
    })

    # Print full regression output for the final model
    print(f"\n  {label}:")
    print(f"    Dea: {fmt(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea'])}")
    print(f"    N={r.nobs:,}, R²_within={getattr(r, 'rsquared_within', 'N/A')}")
    if ctls:
        # Print significant controls
        sig_ctls = []
        for c in ctls:
            if c in r.params.index and r.pvalues[c] < 0.1:
                sig_ctls.append(f"{c}={fmt(r.params[c], r.std_errors[c], r.pvalues[c])}")
        if sig_ctls:
            print(f"    显著控制变量: {', '.join(sig_ctls)}")

# Save
pd.DataFrame(results_stepwise).to_csv(
    OUT / "table1_stepwise.csv", index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 倒U型检验 Dea²（论文表5）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("2. 倒U型检验 — Dea² 是否显著？")
print("=" * 90)

for var in ["Dea", "Breadth", "Depth"]:
    s_u = s.copy()
    s_u[f"{var}_sq"] = s_u[var] ** 2

    # 包含二次项
    r = run_fe(s_u, OUTCOME, [var, f"{var}_sq"] + CONTROLS)
    lin_coef = r.params.get(var, np.nan)
    lin_p = r.pvalues.get(var, np.nan)
    sq_coef = r.params.get(f"{var}_sq", np.nan)
    sq_p = r.pvalues.get(f"{var}_sq", np.nan)

    print(f"\n  {var}:")
    print(f"    一次项: {fmt(lin_coef, r.std_errors.get(var, np.nan), lin_p)}")
    print(f"    二次项: {fmt(sq_coef, r.std_errors.get(f'{var}_sq', np.nan), sq_p)}")

    # 判断倒U型条件：一次项>0且显著，二次项<0且显著
    if lin_coef > 0 and lin_p < 0.1 and sq_coef < 0 and sq_p < 0.1:
        # 计算转折点
        turning = -lin_coef / (2 * sq_coef)
        pct = (s_u[var] < turning).mean() * 100
        print(f"    ✅ 倒U型关系存在！转折点={turning:.3f}（{pct:.0f}%样本在左侧）")
    elif lin_p < 0.05 and sq_p > 0.1:
        print(f"    → 仅有线性关系，无倒U型证据")
    elif sq_p < 0.1:
        sign = "正U型" if sq_coef > 0 else "倒U型"
        print(f"    → {sign}但一次项不显著")
    else:
        print(f"    → 无非线性关系")

    # 补充：用分位数分组观察非线性模式
    s_u["Dea_quartile"] = pd.qcut(s_u[var], 4, labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])
    print(f"    分位数回归:")
    for q in ["Q1(低)", "Q2", "Q3", "Q4(高)"]:
        sub = s_u[s_u["Dea_quartile"] == q]
        r_q = run_fe(sub, OUTCOME, [var] + CONTROLS)
        print(f"      {q}: {fmt(r_q.params[var], r_q.std_errors[var], r_q.pvalues[var])}  "
              f"N={r_q.nobs:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PPML回归（稳健性7）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("3. PPML回归（Poisson Pseudo-Maximum Likelihood）")
print("=" * 90)

# res is non-negative composite index → PPML applicable
# PPML with Industry×Year FE (high-dim FE Poisson impractical with 4,929 firm dummies)
import statsmodels.api as sm

s_ppml = s.copy()
# Ensure res > 0 for log link (PPML requires non-negative, not strictly positive)
print(f"  res 最小值={s_ppml[OUTCOME].min():.3f}, 最大值={s_ppml[OUTCOME].max():.3f}")
assert s_ppml[OUTCOME].min() >= 0, "res has negative values, cannot use PPML"

s_ppml["ind_str"] = s_ppml["ind_str"].fillna("Unknown")

# PPML with Industry FE + Year FE
ind_dummies = pd.get_dummies(s_ppml["ind_str"], prefix="ind", drop_first=True)
yr_dummies = pd.get_dummies(s_ppml["year"], prefix="yr", drop_first=True)

# Add city clustering via robust covariance
try:
    X_ppml = sm.add_constant(
        pd.concat([s_ppml[["Dea"] + CONTROLS], ind_dummies, yr_dummies], axis=1)
    )
    y_ppml = s_ppml[OUTCOME]

    mod_ppml = sm.GLM(y_ppml, X_ppml.astype(float),
                      family=sm.families.Poisson(link=sm.genmod.families.links.Log()))
    res_ppml = mod_ppml.fit(cov_type="cluster", cov_kwds={"groups": s_ppml["city"]})

    print(f"\n  PPML (Industry FE + Year FE):")
    print(f"    Dea: {fmt(res_ppml.params['Dea'], res_ppml.bse['Dea'], res_ppml.pvalues['Dea'])}")
    print(f"    N={len(s_ppml):,}, Pseudo R²={res_ppml.pseudo_rsquared():.3f}")
    print(f"    显著控制变量:")
    for c in CONTROLS:
        if c in res_ppml.params.index and res_ppml.pvalues[c] < 0.1:
            stars = "***" if res_ppml.pvalues[c] < 0.01 else "**" if res_ppml.pvalues[c] < 0.05 else "*"
            print(f"      {c}: {res_ppml.params[c]:.4f}{stars} ({res_ppml.bse[c]:.4f})")
except Exception as e:
    print(f"  PPML failed: {e}")

# Compare: OLS with same specification
try:
    mod_ols = sm.OLS(y_ppml, X_ppml.astype(float))
    res_ols = mod_ols.fit(cov_type="cluster", cov_kwds={"groups": s_ppml["city"]})
    print(f"\n  OLS (Industry FE + Year FE, 同规格对比):")
    print(f"    Dea: {fmt(res_ols.params['Dea'], res_ols.bse['Dea'], res_ols.pvalues['Dea'])}")
    print(f"    R²={res_ols.rsquared:.3f}")
except Exception as e:
    print(f"  OLS comparison failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 描述性统计表（论文附录格式）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("4. 描述性统计表（论文附录格式）")
print("=" * 90)

desc_vars = {
    "res": "企业供应链韧性",
    "Dea": "数据要素应用能力(DEA)",
    "Breadth": "应用广度",
    "Depth": "应用深度",
    "lnage": "ln(企业年龄)",
    "klr": "资本劳动比",
    "lnsize": "ln(总资产)",
    "bsize": "董事会规模",
    "dual": "两职合一",
    "lnrd": "ln(研发投入)",
    "indrate": "独立董事比例",
    "own": "所有权性质",
    "lev": "资产负债率",
}

available_vars = [v for v in desc_vars if v in s.columns]
desc = s[available_vars].describe().T
desc["N"] = s[available_vars].count()
desc["缺失"] = s[available_vars].isna().sum()
desc = desc.rename(columns={
    "mean": "均值", "std": "标准差", "min": "最小值",
    "25%": "P25", "50%": "中位数", "75%": "P75", "max": "最大值"
})
desc["变量标签"] = [desc_vars.get(v, v) for v in desc.index]

# Reorder
cols_order = ["变量标签", "N", "缺失", "均值", "标准差", "最小值", "P25", "中位数", "P75", "最大值"]
desc = desc[cols_order]

print(desc.to_string())
desc.to_csv(OUT / "descriptive_statistics.csv", encoding="utf-8-sig")

# Panel描述
print(f"\n  面板结构:")
print(f"    观测数: {len(s):,}")
print(f"    企业数: {s['stkcode'].nunique():,}")
print(f"    年份范围: {s['year'].min():.0f}-{s['year'].max():.0f}")
print(f"    平均T: {s.groupby('stkcode').size().mean():.1f}年/企业")
print(f"    城市数: {s['city'].nunique():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("【剩余复现内容完成】")
print(f"{'='*90}")
print(f"""
  已生成文件:
    output/table1_stepwise.csv          — 表1逐步加控制变量
    output/descriptive_statistics.csv   — 描述性统计表

  关键结果:
    1. 表1逐步加控制变量: Dea系数从(1)→(5)的稳定性
    2. 倒U型检验: Dea²是否显著? → 见上方
    3. PPML: Dea在Poisson回归中是否显著? → 见上方
    4. 描述性统计: 全部变量分布特征
""")
