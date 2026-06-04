"""
任务5 — 安慰剂检验（5000次随机置换）— 向量化高速版
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

方法：每年内随机置换Dea → 重新估计FE → 构建系数分布
向量化within-firm/within-year demeaning，~100x加速
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import linalg
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

OLD_OUT = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "output"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
print("Reading data...")
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
OUTCOME = "res"


def fast_demean(X, fidx, yidx):
    """Vectorized within-firm + within-year double-demeaning"""
    n_firms = fidx.max() + 1
    n_years = yidx.max() + 1
    flat = X.ndim == 1
    if flat:
        X = X.reshape(-1, 1)

    # Firm demeaning
    count_f = np.bincount(fidx, minlength=n_firms)
    means_f = np.zeros((n_firms, X.shape[1]))
    for j in range(X.shape[1]):
        means_f[:, j] = np.bincount(fidx, weights=X[:, j], minlength=n_firms)
    means_f = means_f / np.maximum(count_f[:, np.newaxis], 1)
    X_dem = X - means_f[fidx]

    # Year demeaning
    count_y = np.bincount(yidx, minlength=n_years)
    means_y = np.zeros((n_years, X_dem.shape[1]))
    for j in range(X_dem.shape[1]):
        means_y[:, j] = np.bincount(yidx, weights=X_dem[:, j], minlength=n_years)
    means_y = means_y / np.maximum(count_y[:, np.newaxis], 1)
    X_dem = X_dem - means_y[yidx]

    return X_dem.ravel() if flat else X_dem


def run_placebo(data, core_var, n_perm=5000):
    """Permute Dea within each year, re-estimate FE, build null distribution."""
    np.random.seed(42)

    cols = [core_var, OUTCOME] + CONTROLS + ["stkcode", "year", "city_x"]
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=[core_var] + CONTROLS + [OUTCOME, "city"])

    fidx = d.groupby("stkcode").ngroup().values
    yidx = d.groupby("year").ngroup().values
    n_total = len(d)

    print(f"  样本: {n_total:,} obs, {d['stkcode'].nunique():,} firms")

    # ── True coefficient ──────────────────────────────────────────────────────
    d_idx = d.set_index(["stkcode", "year"])
    fml = f"{OUTCOME} ~ {core_var} + {' + '.join(CONTROLS)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d_idx, drop_absorbed=True)
    clus = d.set_index(["stkcode", "year"])["city"]
    r_true = mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)
    true_coef = r_true.params[core_var]
    true_pval = r_true.pvalues[core_var]
    print(f"  实际系数: {true_coef:.5f}, p={true_pval:.4f}")

    # ── Pre-compute fixed components ──────────────────────────────────────────
    X_ctrl = d[CONTROLS].values.astype(float)
    X_ctrl_dem = fast_demean(X_ctrl, fidx, yidx)
    Y_raw = d[OUTCOME].values.astype(float)
    Y_dem = fast_demean(Y_raw, fidx, yidx)

    dea_raw = d[core_var].values.astype(float)
    year_groups = {yr: np.where(yidx == yr)[0] for yr in np.unique(yidx)}

    # ── Permutation loop ─────────────────────────────────────────────────────
    coefs = np.zeros(n_perm)
    t0 = time.time()

    for p in range(n_perm):
        # 每年内随机打乱Dea（保留年份分布，打破企业关联）
        dea_perm = dea_raw.copy()
        for yr, idx in year_groups.items():
            vals = dea_perm[idx].copy()
            np.random.shuffle(vals)
            dea_perm[idx] = vals

        # Demean and estimate
        dea_dem = fast_demean(dea_perm, fidx, yidx)
        X = np.column_stack([dea_dem, X_ctrl_dem])

        XtX = X.T @ X + np.eye(X.shape[1]) * 1e-10
        Xty = X.T @ Y_dem
        try:
            beta = linalg.solve(XtX, Xty, assume_a='pos')
            coefs[p] = beta[0]
        except linalg.LinAlgError:
            coefs[p] = np.nan

        if (p + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (p + 1) / elapsed
            eta = (n_perm - p - 1) / rate
            print(f"  {p+1}/{n_perm} ({100*(p+1)/n_perm:.0f}%), "
                  f"{rate:.0f}/s, ETA {eta:.0f}s")

    # ── Results ──────────────────────────────────────────────────────────────
    valid = coefs[~np.isnan(coefs)]
    p_emp = np.mean(np.abs(valid) >= np.abs(true_coef))

    print(f"\n  === {core_var} 结果 ===")
    print(f"  实际系数:    {true_coef:.5f}")
    print(f"  参数p值:     {true_pval:.4f}")
    print(f"  经验p值:     {p_emp:.4f}")
    print(f"  置换均值:    {np.mean(valid):.6f}")
    print(f"  置换标准差:  {np.std(valid):.6f}")
    print(f"  95%区间:    [{np.percentile(valid, 2.5):.5f}, {np.percentile(valid, 97.5):.5f}]")

    if p_emp < 0.01:
        print(f"  ✅ 通过安慰剂检验（经验p<0.01）")
    elif p_emp < 0.05:
        print(f"  ⚠️ 勉强通过（经验p<0.05）")
    else:
        print(f"  ❌ 未通过安慰剂检验（经验p>=0.05）")

    return {
        "core_var": core_var,
        "true_coef": true_coef,
        "true_pval": true_pval,
        "n_perm": n_perm,
        "n_valid": len(valid),
        "p_empirical": p_emp,
        "perm_coefs": valid,
        "perm_mean": np.mean(valid),
        "perm_std": np.std(valid),
        "perm_p025": np.percentile(valid, 2.5),
        "perm_p975": np.percentile(valid, 97.5)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 执行
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("安慰剂检验 — 5000次随机置换（向量化高速版）")
print("每年内随机打乱Dea，保留年份FE，打破企业关联")
print("=" * 80)

N_PERM = 5000
results = []

for core in ["Dea", "Breadth", "Depth"]:
    print(f"\n{'─'*60}")
    print(f"核心变量: {core}")
    print(f"{'─'*60}")
    results.append(run_placebo(df, core, n_perm=N_PERM))

# ── 保存 ──────────────────────────────────────────────────────────────────────
summary = pd.DataFrame([{
    "核心变量": r["core_var"],
    "实际系数": r["true_coef"],
    "参数p值": r["true_pval"],
    "经验p值": r["p_empirical"],
    "置换均值": r["perm_mean"],
    "置换标准差": r["perm_std"],
    "2.5分位": r["perm_p025"],
    "97.5分位": r["perm_p975"],
    "有效置换数": r["n_valid"]
} for r in results])

summary.to_csv(OUT / "placebo_test_summary.csv", index=False, encoding="utf-8-sig")

for r in results:
    pd.DataFrame({"coefficient": r["perm_coefs"]}).to_csv(
        OUT / f"placebo_distribution_{r['core_var']}.csv",
        index=False, encoding="utf-8-sig")

print(f"\n结果已保存到 {OUT}/")
print(f"  placebo_test_summary.csv — 汇总表")
for r in results:
    print(f"  placebo_distribution_{r['core_var']}.csv — 置换系数分布")
