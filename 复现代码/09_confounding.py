"""
排除替代解释：存货机制 × 数字化转型 交互效应
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

检验基准系数为负的成因：
  1. 存货机制: Dea × inv → res
  2. 数字化转型: Dea × DIGI_text → res
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
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)


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


# ================================================================
# 主程序
# ================================================================
print("=" * 80)
print("排除替代解释 — 存货机制 & 数字化转型交互")
print("=" * 80)

# 加载面板
df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
df["year"] = df["year"].astype(int)

# 合并存货比率 (from control dataset)
ctrl, _ = pyreadstat.read_dta(DATA / "常用控制变量2000_2024_Ver3.1.dta")
ctrl["id_stock"] = ctrl["id_stock"].astype(int)
ctrl["year"] = ctrl["year"].astype(int)
if "inv" in ctrl.columns:
    df = df.merge(ctrl[["id_stock","year","inv"]], on=["id_stock","year"], how="left")
    print(f"  存货比率(inv): {df['inv'].notna().sum():,} non-null")

# 合并数字化转型 (from digital_transform_zhao.dta)
digi, _ = pyreadstat.read_dta(DATA / "digital_transform_zhao.dta")
digi["id_stock"] = digi["id_stock"].astype(int)
digi["year"] = digi["year"].astype(int)
if "DIGI_text" in digi.columns:
    df = df.merge(digi[["id_stock","year","DIGI_text"]], on=["id_stock","year"], how="left")
    print(f"  数字化转型(DIGI_text): {df['DIGI_text'].notna().sum():,} non-null")

results_all = []

# ================================================================
# 0. 基准回归
# ================================================================
print("\n" + "-" * 60)
print("0. 基准回归")
print("-" * 60)

r0 = run_fe(df, "res", ["Dea"])
if r0:
    c0 = r0.params["Dea"]; p0 = r0.pvalues["Dea"]
    print(f"  Dea: {c0:+.4f}{star(p0)} (p={p0:.4f}) N={r0.nobs:,}")
    results_all.append({"检验": "基准回归", "Dea系数": f"{c0:+.4f}{star(p0)}",
                       "标准误": f"({r0.std_errors['Dea']:.4f})",
                       "p值": f"{p0:.4f}", "N": r0.nobs})

# ================================================================
# 1. 存货机制: Dea × inv → res
# ================================================================
if "inv" in df.columns:
    print("\n" + "-" * 60)
    print("1. 存货机制: Dea + inv + Dea×inv → res")
    print("-" * 60)

    d1 = df.copy()
    d1["Dea_x_inv"] = d1["Dea"] * d1["inv"]
    r1 = run_fe(d1, "res", ["Dea", "inv", "Dea_x_inv"])
    if r1:
        c_dea = r1.params.get("Dea", np.nan)
        c_inv = r1.params.get("inv", np.nan)
        c_int = r1.params.get("Dea_x_inv", np.nan)
        p_int = r1.pvalues.get("Dea_x_inv", np.nan)
        print(f"  Dea: {c_dea:+.4f}{star(r1.pvalues.get('Dea', np.nan))}")
        print(f"  inv: {c_inv:+.4f}{star(r1.pvalues.get('inv', np.nan))}")
        print(f"  Dea×inv: {c_int:+.4f}{star(p_int)} (p={p_int:.4f})")
        print(f"  N={r1.nobs:,}")
        results_all.append({"检验": "存货×Dea交互", "Dea系数": f"{c_int:+.4f}{star(p_int)}",
                           "标准误": f"({r1.std_errors.get('Dea_x_inv', np.nan):.4f})",
                           "p值": f"{p_int:.4f}", "N": r1.nobs})

# ================================================================
# 2. 数字化转型: Dea × DIGI_text → res
# ================================================================
if "DIGI_text" in df.columns:
    print("\n" + "-" * 60)
    print("2. 数字化转型调节: Dea + DIGI_text + Dea×DIGI_text → res")
    print("-" * 60)

    d2 = df.copy()
    d2["Dea_x_DIGI"] = d2["Dea"] * d2["DIGI_text"]
    r2 = run_fe(d2, "res", ["Dea", "DIGI_text", "Dea_x_DIGI"])
    if r2:
        c_dea = r2.params.get("Dea", np.nan)
        c_digi = r2.params.get("DIGI_text", np.nan)
        c_int = r2.params.get("Dea_x_DIGI", np.nan)
        p_int = r2.pvalues.get("Dea_x_DIGI", np.nan)
        print(f"  Dea: {c_dea:+.4f}{star(r2.pvalues.get('Dea', np.nan))}")
        print(f"  DIGI_text: {c_digi:+.4f}{star(r2.pvalues.get('DIGI_text', np.nan))}")
        print(f"  Dea×DIGI_text: {c_int:+.4f}{star(p_int)} (p={p_int:.4f})")
        print(f"  N={r2.nobs:,}")
        results_all.append({"检验": "数字化转型×Dea交互", "Dea系数": f"{c_int:+.4f}{star(p_int)}",
                           "标准误": f"({r2.std_errors.get('Dea_x_DIGI', np.nan):.4f})",
                           "p值": f"{p_int:.4f}", "N": r2.nobs})

# ================================================================
# Save
# ================================================================
print("\n" + "=" * 80)
print("保存结果...")
out_df = pd.DataFrame(results_all)
out_path = OUT / "confounding_interaction.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(out_df.to_string())
print(f"\n已保存: {out_path}")
