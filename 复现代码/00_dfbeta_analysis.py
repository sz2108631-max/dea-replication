"""
DFBETA 样本敏感性分析
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

做法: OLS估计基准回归 → 计算Dea系数的DFBETA → 按DFBETA分位剔除最负观测 → PanelOLS重估
检验基准系数是否受少量极端样本驱动
"""

import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.api as sm
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

OUTCOME  = "res"
CORES = ["Dea", "Breadth", "Depth"]


def star(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""


def make_sample(data, core_var="Dea"):
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    cols = [core_var] + use_ctrl + [OUTCOME, "id_stock", "year"]
    d = data[[c for c in cols if c in data.columns]].copy()
    return d.dropna(subset=[core_var] + use_ctrl + [OUTCOME])


def run_fe(data, dep, indep):
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    d = data.copy().set_index(["id_stock", "year"])
    all_vars = list(dict.fromkeys([c for c in indep + use_ctrl if c in d.columns]))
    X = d[all_vars].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    Y = pd.to_numeric(d[dep], errors="coerce")
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)
    except:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit()


def run_dfbeta_for_var(data, core_var, percentiles=[2, 3, 5, 10]):
    """DFBETA: OLS → DFBETA → trim → PanelOLS"""
    np.random.seed(42)
    s = make_sample(data, core_var)
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    print(f"  {core_var}: 全样本 N={len(s):,}")

    # Full sample PanelOLS
    r_full = run_fe(s, OUTCOME, [core_var] + use_ctrl)
    coef_full = r_full.params[core_var]
    p_full = r_full.pvalues[core_var]
    print(f"    全样本(FE): {coef_full:+.4f}{star(p_full)} (p={p_full:.4f}) N={r_full.nobs:,}")

    # OLS (no FE) for DFBETA computation via statsmodels
    ols_vars = list(dict.fromkeys([core_var] + use_ctrl))
    s_ols = s[ols_vars + [OUTCOME]].dropna()
    X_ols = sm.add_constant(s_ols[ols_vars].values)
    y_ols = s_ols[OUTCOME].values
    try:
        mod_ols = sm.OLS(y_ols, X_ols).fit()
        influence = mod_ols.get_influence()
        # DFBETA for core_var coefficient (index 1 after const)
        dfbetas = influence.dfbetas[:, 1]
    except Exception as e:
        print(f"    DFBETA(OLS)失败: {e}, 使用残差近似")
        residuals = s_ols[OUTCOME] - mod_ols.predict(X_ols)
        dfbetas = residuals.values / residuals.std()

    # Sort by DFBETA: most negative = dragging Dea coefficient down most
    s["dfbeta"] = dfbetas

    results = []
    for pct in percentiles:
        threshold = np.percentile(s["dfbeta"], pct)
        s_trim = s[s["dfbeta"] > threshold].copy()
        n_removed = len(s) - len(s_trim)
        try:
            r_trim = run_fe(s_trim, OUTCOME, [core_var] + use_ctrl)
            coef_trim = r_trim.params[core_var]
            p_trim = r_trim.pvalues[core_var]
            print(f"    剔除DFBETA<P{pct} ({n_removed} obs, {pct:.1f}%): "
                  f"{coef_trim:+.4f}{star(p_trim)} N={r_trim.nobs:,}")
            results.append({
                "变量": core_var, "剔除分位": f"P{pct}",
                "系数": coef_trim, "标准误": r_trim.std_errors[core_var],
                "p值": p_trim, "N": r_trim.nobs, "删除比例": f"{pct:.1f}%"
            })
        except Exception as e:
            print(f"    P{pct}: 失败 ({e})")
            results.append({
                "变量": core_var, "剔除分位": f"P{pct}",
                "系数": np.nan, "标准误": np.nan,
                "p值": np.nan, "N": len(s_trim), "删除比例": f"{pct:.1f}%"
            })

    # Full sample result
    results.insert(0, {
        "变量": core_var, "剔除分位": "全样本",
        "系数": coef_full, "标准误": r_full.std_errors[core_var],
        "p值": p_full, "N": r_full.nobs, "删除比例": "0.0%"
    })

    return results


# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("DFBETA 样本敏感性分析")
print("=" * 80)

df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
df["year"] = df["year"].astype(int)

all_results = []
for core in CORES:
    print(f"\n{'─'*60}")
    print(f"核心变量: {core}")
    print(f"{'─'*60}")
    res = run_dfbeta_for_var(df, core_var=core)
    all_results.extend(res)

# ── 汇总表格 ──
print(f"\n{'='*80}")
print("DFBETA 汇总表")
print(f"{'='*80}")

for core in CORES:
    row = f"{core:<10}"
    for pct in ["全样本", "P2", "P3", "P5", "P10"]:
        matches = [r for r in all_results if r["变量"] == core and r["剔除分位"] == pct]
        if matches:
            r = matches[0]
            c = r["系数"]
            s_str = star(r["p值"]) if not np.isnan(r["p值"]) else ""
            row += f" {c:+.4f}{s_str:<3}  ".rjust(24) if not np.isnan(c) else f"{'失败':>24}"
        else:
            row += f"{'':>24}"
    print(row)

n_row = f"{'N':<10}"
del_row = f"{'删除比例':<10}"
for pct in ["全样本", "P2", "P3", "P5", "P10"]:
    matches = [r for r in all_results if r["变量"] == "Dea" and r["剔除分位"] == pct]
    if matches:
        n_row += f"{matches[0]['N']:>24,}"
        del_row += f"{matches[0]['删除比例']:>24}"
    else:
        n_row += f"{'':>24}"
        del_row += f"{'':>24}"
print(n_row)
print(del_row)

out_df = pd.DataFrame(all_results)
out_path = OUT / "dfbeta_analysis.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n已保存: {out_path}")
