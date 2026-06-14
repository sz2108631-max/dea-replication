"""
更换Y衡量方式 — 稳健性检验
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

四项Y变换：
  1. 缩尾: res 在 1%/99% 分位数处缩尾
  2. 截尾: res 在 1%/99% 分位数处截尾
  3. 滞后: Dea 滞后一期
  4. Z-score: res 标准化为均值0标准差1
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

OUTCOME  = "res"


def star(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""


def run_fe(data, dep, indep):
    use_ctrl = [c for c in CONTROLS if c in data.columns]
    all_vars = [dep] + indep + use_ctrl + ["id_stock", "year"]
    s = data[[c for c in all_vars if c in data.columns]].dropna().copy()
    if len(s) < 200:
        return None
    s_idx = s.set_index(["id_stock", "year"])
    X = s_idx[[c for c in indep + use_ctrl if c in s_idx.columns]]
    Y = s_idx[dep]
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)
    except:
        try:
            mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
            return mod.fit()
        except:
            return None


def fmt(res_obj, var):
    if res_obj is None:
        return "N/A"
    c = res_obj.params.get(var, np.nan)
    se = res_obj.std_errors.get(var, np.nan)
    p = res_obj.pvalues.get(var, np.nan)
    return f"{c:+.4f}{star(p)}"


# ================================================================
# 主程序
# ================================================================
print("=" * 80)
print("更换Y衡量方式 — 稳健性检验")
print("=" * 80)

df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
df["year"] = df["year"].astype(int)
print(f"基准面板: {len(df):,} obs\n")

results_all = []

# ================================================================
# 0. 基准回归
# ================================================================
print("─" * 60)
print("0. 基准回归 (Dea → res)")
print("─" * 60)

r_base = run_fe(df, "res", ["Dea"])
if r_base is not None:
    c0 = r_base.params["Dea"]
    p0 = r_base.pvalues["Dea"]
    print(f"  Dea: {c0:+.4f}{star(p0)} (p={p0:.4f}) N={r_base.nobs:,}")
    results_all.append({"检验": "基准回归", "Dea系数": f"{c0:+.4f}{star(p0)}",
                       "标准误": f"({r_base.std_errors['Dea']:.4f})",
                       "p值": f"{p0:.4f}", "N": r_base.nobs})
else:
    print("  失败")
    c0, p0 = np.nan, np.nan

# ================================================================
# 1. 缩尾 res (1%/99%)
# ================================================================
print("\n" + "─" * 60)
print("1. 缩尾 res (P1/P99)")
print("─" * 60)

d1 = df.copy()
p1, p99 = d1["res"].quantile(0.01), d1["res"].quantile(0.99)
d1["res_w"] = d1["res"].clip(lower=p1, upper=p99)
r1 = run_fe(d1, "res_w", ["Dea"])
if r1 is not None:
    c = r1.params["Dea"]; p = r1.pvalues["Dea"]
    print(f"  Dea: {c:+.4f}{star(p)} (p={p:.4f}) N={r1.nobs:,}")
    results_all.append({"检验": "缩尾res", "Dea系数": f"{c:+.4f}{star(p)}",
                       "标准误": f"({r1.std_errors['Dea']:.4f})",
                       "p值": f"{p:.4f}", "N": r1.nobs})

# ================================================================
# 2. 截尾 res (1%/99%)
# ================================================================
print("\n" + "─" * 60)
print("2. 截尾 res (剔除P1以下 + P99以上)")
print("─" * 60)

d2 = df[(df["res"] >= p1) & (df["res"] <= p99)].copy()
r2 = run_fe(d2, "res", ["Dea"])
if r2 is not None:
    c = r2.params["Dea"]; p = r2.pvalues["Dea"]
    print(f"  Dea: {c:+.4f}{star(p)} (p={p:.4f}) N={r2.nobs:,}")
    results_all.append({"检验": "截尾res", "Dea系数": f"{c:+.4f}{star(p)}",
                       "标准误": f"({r2.std_errors['Dea']:.4f})",
                       "p值": f"{p:.4f}", "N": r2.nobs})

# ================================================================
# 3. 滞后一期 Dea
# ================================================================
print("\n" + "─" * 60)
print("3. 滞后一期 Dea (L.Dea → res)")
print("─" * 60)

d3 = df.copy()
d3 = d3.sort_values(["id_stock", "year"])
d3["Dea_lag"] = d3.groupby("id_stock")["Dea"].shift(1)
r3 = run_fe(d3, "res", ["Dea_lag"])
if r3 is not None:
    c = r3.params["Dea_lag"]; p = r3.pvalues["Dea_lag"]
    print(f"  L.Dea: {c:+.4f}{star(p)} (p={p:.4f}) N={r3.nobs:,}")
    results_all.append({"检验": "滞后一期Dea", "Dea系数": f"{c:+.4f}{star(p)}",
                       "标准误": f"({r3.std_errors['Dea_lag']:.4f})",
                       "p值": f"{p:.4f}", "N": r3.nobs})

# ================================================================
# 4. Z-score 标准化 res
# ================================================================
print("\n" + "─" * 60)
print("4. Z-score res (均值0, 标准差1)")
print("─" * 60)

d4 = df.copy()
d4["res_z"] = (d4["res"] - d4["res"].mean()) / d4["res"].std()
r4 = run_fe(d4, "res_z", ["Dea"])
if r4 is not None:
    c = r4.params["Dea"]; p = r4.pvalues["Dea"]
    print(f"  Dea: {c:+.4f}{star(p)} (p={p:.4f}) N={r4.nobs:,}")
    results_all.append({"检验": "Z-score res", "Dea系数": f"{c:+.4f}{star(p)}",
                       "标准误": f"({r4.std_errors['Dea']:.4f})",
                       "p值": f"{p:.4f}", "N": r4.nobs})

# ================================================================
# 5. Topsis 三级指标回归 (Dea → 各三级指标)
# ================================================================
print("\n" + "─" * 60)
print("5. 三级指标回归 (Dea → 各三级指标)")
print("─" * 60)

try:
    ind_df, _ = pyreadstat.read_dta(OUT / "中间结果" / "indicators_11.dta")
    ind_df["id_stock"] = pd.to_numeric(ind_df["stkcode"], errors="coerce")
    ind_df["year"] = ind_df["year"].astype(int)

    ind_names = {
        "ind1": "库存调整幅度", "ind2": "供需偏离度", "ind3": "供应链集中度",
        "ind4": "供应商稳定性", "ind5": "客户稳定性", "ind6": "盈利能力(ROA)",
        "ind7": "现金周转率", "ind8": "应收占比", "ind9": "申请专利知识宽度",
        "ind10": "授权专利知识宽度", "ind11": "外部知识引用"
    }
    use_ctrl = [c for c in CONTROLS if c in df.columns]

    for ind_col, ind_label in ind_names.items():
        # Merge indicator into panel
        d5 = df.merge(ind_df[["id_stock","year",ind_col]], on=["id_stock","year"], how="left")
        r5 = run_fe(d5, ind_col, ["Dea"])
        if r5 is not None:
            c = r5.params["Dea"]; p = r5.pvalues["Dea"]
            print(f"  Dea → {ind_label}: {c:+.4f}{star(p)} (p={p:.4f}) N={r5.nobs:,}")
            results_all.append({"检验": f"三级指标: {ind_label}", "Dea系数": f"{c:+.4f}{star(p)}",
                               "标准误": f"({r5.std_errors['Dea']:.4f})",
                               "p值": f"{p:.4f}", "N": r5.nobs})
        else:
            print(f"  Dea → {ind_label}: N/A (样本不足)")
except Exception as e:
    print(f"  三级指标回归失败: {e}")
    print(f"  需先运行 01_construct_resilience.py 生成 indicators_11.dta")

# ================================================================
# Save
# ================================================================
print("\n" + "=" * 80)
print("保存结果...")
out_df = pd.DataFrame(results_all)
out_path = OUT / "robustness_y_transform.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(out_df.to_string())
print(f"\n已保存: {out_path}")
