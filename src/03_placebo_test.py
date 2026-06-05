"""
任务5 — 安慰剂检验（5000次随机置换）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

做法：
  随机打乱 Dea/Breadth/Depth 的取值（在截面维度），保留数据结构不变
  重复 5000 次
  统计实际系数在置换分布中的位置
  计算经验 p 值
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
df_raw = pd.read_csv(OLD_OUT / "panel_merged.csv")
df_raw["stkcode"] = df_raw["stkcode"].astype(str).str.zfill(6)

# 样本筛选
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


def make_sample(data, core_var="Dea"):
    cols = [core_var] + CONTROLS + [OUTCOME, "stkcode", "year", "city_x"]
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=[core_var] + CONTROLS + [OUTCOME])
    return d.dropna(subset=["city"])


def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def run_placebo(data, core_var="Dea", n_perm=5000):
    """
    安慰剂检验：随机置换核心变量
    在截面维度随机打乱（同一年内），保留面板结构
    """
    np.random.seed(42)

    # 准备数据
    s = make_sample(data, core_var)
    print(f"  样本: {len(s):,} obs, {s['stkcode'].nunique():,} 家企业")

    # 实际系数
    r_true = run_fe(s, OUTCOME, [core_var] + CONTROLS)
    true_coef = r_true.params[core_var]
    true_pval = r_true.pvalues[core_var]
    print(f"  实际系数: {true_coef:.5f}, p={true_pval:.4f}")

    # 为每个firm分配一个随机值（保持firm-时间结构）
    firms = s["stkcode"].unique()
    firm_values = s.groupby("stkcode")[core_var].first()

    coefs = np.zeros(n_perm)
    pvals = np.zeros(n_perm)

    for i in range(n_perm):
        # 随机打乱firm-level的DEA值
        shuffled = firm_values.sample(frac=1, random_state=42+i).values
        firm_to_value = dict(zip(firms, shuffled[:len(firms)]))

        s_perm = s.copy()
        s_perm[core_var] = s_perm["stkcode"].map(firm_to_value)

        try:
            r = run_fe(s_perm, OUTCOME, [core_var] + CONTROLS)
            coefs[i] = r.params[core_var]
            pvals[i] = r.pvalues[core_var]
        except Exception:
            coefs[i] = np.nan
            pvals[i] = np.nan

        if (i + 1) % 1000 == 0:
            pct_done = (i + 1) / n_perm * 100
            valid = np.sum(~np.isnan(coefs[:i+1]))
            print(f"  进度: {i+1}/{n_perm} ({pct_done:.0f}%), 有效: {valid}")

    # 经验p值
    valid_coefs = coefs[~np.isnan(coefs)]
    n_valid = len(valid_coefs)
    p_empirical = np.mean(np.abs(valid_coefs) >= np.abs(true_coef))

    return {
        "core_var": core_var,
        "true_coef": true_coef,
        "true_pval": true_pval,
        "n_perm": n_perm,
        "n_valid": n_valid,
        "p_empirical": p_empirical,
        "perm_coefs": valid_coefs,
        "perm_mean": np.mean(valid_coefs),
        "perm_std": np.std(valid_coefs),
        "perm_p5": np.percentile(valid_coefs, 5),
        "perm_p95": np.percentile(valid_coefs, 95)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 执行安慰剂检验
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("安慰剂检验 — 5000次随机置换")
print("=" * 80)

results = []
N_PERM = 5000

for core in ["Dea", "Breadth", "Depth"]:
    print(f"\n{'─'*60}")
    print(f"核心变量: {core}")
    print(f"{'─'*60}")

    placebo_result = run_placebo(df, core_var=core, n_perm=N_PERM)

    # 汇总
    perm_coefs = placebo_result["perm_coefs"]
    results.append({
        "核心变量": core,
        "实际系数": placebo_result["true_coef"],
        "参数p值": placebo_result["true_pval"],
        "经验p值": placebo_result["p_empirical"],
        "置换均值": placebo_result["perm_mean"],
        "置换标准差": placebo_result["perm_std"],
        "5分位": placebo_result["perm_p5"],
        "95分位": placebo_result["perm_p95"],
        "有效置换数": placebo_result["n_valid"]
    })

    print(f"\n  === 结果 ===")
    print(f"  实际系数:    {placebo_result['true_coef']:.5f}")
    print(f"  参数p值:     {placebo_result['true_pval']:.4f}")
    print(f"  经验p值:     {placebo_result['p_empirical']:.4f}")
    print(f"  置换均值:    {placebo_result['perm_mean']:.5f}")
    print(f"  置换标准差:  {placebo_result['perm_std']:.5f}")
    print(f"  95%区间:    [{placebo_result['perm_p5']:.5f}, {placebo_result['perm_p95']:.5f}]")

    # 判断
    if placebo_result["p_empirical"] < 0.01:
        verdict = "✅ 通过安慰剂检验（p<0.01）"
    elif placebo_result["p_empirical"] < 0.05:
        verdict = "⚠️ 勉强通过（p<0.05）"
    elif placebo_result["p_empirical"] < 0.10:
        verdict = "⚠️ 边界显著（p<0.10）"
    else:
        verdict = "❌ 未通过安慰剂检验（p>=0.10）"
    print(f"  {verdict}")


# ═══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ═══════════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame(results)
summary.to_csv(OUT / "placebo_test_summary.csv", index=False, encoding="utf-8-sig")

# 保存置换系数分布（供画图）
for r in results:
    core = r["核心变量"]
    coefs = r.get("perm_coefs", [])
    if len(coefs) > 0:
        pd.DataFrame({"coefficient": coefs}).to_csv(
        OUT / f"placebo_distribution_{core}.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到 {OUT}/")
print("  placebo_test_summary.csv — 安慰剂检验汇总")
print("  placebo_distribution_*.csv — 置换系数分布")
