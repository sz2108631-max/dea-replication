"""
任务5 — 安慰剂检验（100次随机置换）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

做法：
  随机打乱 Dea/Breadth/Depth 的取值（在截面维度），保留数据结构不变
  重复 100 次
  统计实际系数在置换分布中的位置
  计算经验 p 值
"""

import pandas as pd
import numpy as np
import pyreadstat
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from config.settings import CONTROLS
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
df["year"] = df["year"].astype(int)

OUTCOME  = "res"


def make_sample(data, core_var="Dea"):
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    cols = [core_var] + use_ctrl + [OUTCOME, "id_stock", "year", "city"]
    d = data[[c for c in cols if c in data.columns]].copy()
    return d.dropna(subset=[core_var] + use_ctrl + [OUTCOME, "city"])


def run_fe(data, dep, indep):
    d = data.copy().set_index(["id_stock", "year"])
    X = d[[c for c in indep if c in d.columns]]
    Y = d[dep]
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)
    except:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit()


def run_placebo(data, core_var="Dea", n_perm=100):
    """安慰剂检验：随机置换核心变量"""
    np.random.seed(42)
    s = make_sample(data, core_var)
    print(f"  样本: {len(s):,} obs, {s['id_stock'].nunique():,} 家企业")

    # 实际系数
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    r_true = run_fe(s, OUTCOME, [core_var] + use_ctrl)
    true_coef = r_true.params[core_var]
    true_pval = r_true.pvalues[core_var]
    print(f"  实际系数: {true_coef:.5f}, p={true_pval:.4f}")

    # 逐年打乱（保持年内分布，仅打破企业-变量关联）
    coefs = np.zeros(n_perm)
    for i in range(n_perm):
        s_perm = s.copy()
        # 同一年内随机重排 Dea
        s_perm[core_var] = s_perm.groupby("year")[core_var].transform(
            lambda x: x.sample(frac=1, random_state=42+i).values)
        try:
            r = run_fe(s_perm, OUTCOME, [core_var] + use_ctrl)
            coefs[i] = r.params[core_var]
        except Exception:
            coefs[i] = np.nan
        if (i + 1) % 50 == 0:
            valid = np.sum(~np.isnan(coefs[:i+1]))
            print(f"  进度: {i+1}/{n_perm}, 有效: {valid}")

    valid_coefs = coefs[~np.isnan(coefs)]
    n_valid = len(valid_coefs)
    p_empirical = np.mean(np.abs(valid_coefs) >= np.abs(true_coef)) if n_valid > 0 else 1.0

    return {
        "core_var": core_var,
        "true_coef": true_coef,
        "true_pval": true_pval,
        "n_perm": n_perm,
        "n_valid": n_valid,
        "p_empirical": p_empirical,
        "perm_coefs": valid_coefs,
        "perm_mean": np.mean(valid_coefs) if n_valid > 0 else np.nan,
        "perm_std": np.std(valid_coefs) if n_valid > 0 else np.nan,
        "perm_p5": np.percentile(valid_coefs, 5) if n_valid > 0 else np.nan,
        "perm_p95": np.percentile(valid_coefs, 95) if n_valid > 0 else np.nan
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 执行安慰剂检验
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("安慰剂检验 — 100次随机置换")
print("=" * 80)

results = []
N_PERM = 100

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
