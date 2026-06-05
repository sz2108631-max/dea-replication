"""
额外稳健性 & 机制检验: 融资约束、现金流波动、环境不确定性、产业聚集度、
                  绿色专利替代DV、倒U型检验

数据来源: /Users/weixuan/Desktop/论文/数据资产与生成式AI创新/
"""
import pandas as pd
import numpy as np
import pyreadstat
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_AI = Path("/Users/weixuan/Desktop/论文/数据资产与生成式AI创新")
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual", "lnrd", "indrate", "own", "lev"]


def star(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""

def load_panel():
    """加载基准分析面板"""
    df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
    if "stkcode" in df.columns:
        df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
    df["year"] = df["year"].astype(int)
    return df

def run_fe(data, dep, core, controls=None):
    """双固定效应面板回归"""
    if controls is None:
        controls = CONTROLS
    use_ctrl = [c for c in controls if c in data.columns]
    s = data[[dep, core] + use_ctrl + ["id_stock", "year"]].dropna().copy()
    if len(s) < 200:
        return None, len(s)
    s_idx = s.set_index(["id_stock", "year"])
    X = s_idx[[core] + [c for c in use_ctrl if c in s_idx.columns]]
    Y = s_idx[dep]
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", clusters=s_idx[["id_stock"]])
        return res, len(s)
    except:
        try:
            mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
            return mod.fit(), len(s)
        except:
            return None, len(s)

def fmt(res, var):
    if res is None:
        return ("N/A", "N/A", "N/A")
    c = res.params[var]
    se = res.std_errors[var]
    p = res.pvalues[var]
    return (f"{c:+.4f}{star(p)}", f"({se:.4f})", f"{p:.4f}")


# ================================================================
# 主程序
# ================================================================
print("=" * 80)
print("额外稳健性 & 机制检验")
print("=" * 80)

df = load_panel()
print(f"基准面板: {len(df):,} obs\n")

results_all = []

# ================================================================
# TEST 1: 融资约束缓解机制
# ================================================================
print("─" * 60)
print("TEST 1: 融资约束缓解机制")
print("─" * 60)

# 1a: 融资规模
fin, _ = pyreadstat.read_dta(DATA_AI / "机制" / "最终结果-企业融资规模.dta")
fin["id_stock"] = fin["id_stock"].astype(int)
fin["year"] = fin["year"].astype(int)
d1 = df.merge(fin[["id_stock","year","债券融资规模","股权融资规模","商业信用融资规模"]],
              on=["id_stock","year"], how="left")
print(f"  融资规模: {d1['债券融资规模'].notna().sum():,} non-null")

for v, label in [("债券融资规模","债券融资"), ("股权融资规模","股权融资"), ("商业信用融资规模","商业信用")]:
    r, n = run_fe(d1, v, "Dea")
    if r:
        c, se, p = fmt(r, "Dea")
        print(f"  Dea → {label}: {c} {se} p={p} N={n:,}")
        results_all.append({"检验": "融资约束缓解", "被解释变量": label, "Dea系数": c,
                           "标准误": se, "p值": p, "N": n})

# 1b: 信贷可得性
cred, _ = pyreadstat.read_dta(DATA_AI / "机制" / "最终结果-企业信贷可得性2.dta")
cred["id_stock"] = cred["id_stock"].astype(int)
cred["year"] = cred["year"].astype(int)
d1b = df.merge(cred[["id_stock","year","信贷可得性1","信贷可得性2"]],
               on=["id_stock","year"], how="left")
print(f"  信贷可得性: {d1b['信贷可得性1'].notna().sum():,} non-null")

for v, label in [("信贷可得性1","信贷可得性1"), ("信贷可得性2","信贷可得性2")]:
    r, n = run_fe(d1b, v, "Dea")
    if r:
        c, se, p = fmt(r, "Dea")
        print(f"  Dea → {label}: {c} {se} p={p} N={n:,}")
        results_all.append({"检验": "融资约束缓解", "被解释变量": label, "Dea系数": c,
                           "标准误": se, "p值": p, "N": n})

# ================================================================
# TEST 2: 现金流波动降低机制
# ================================================================
print("\n" + "─" * 60)
print("TEST 2: 现金流波动降低机制")
print("─" * 60)

cfvol, _ = pyreadstat.read_dta(DATA_AI / "机制" / "最终结果-现金流波动.dta")
cfvol["id_stock"] = cfvol["id_stock"].astype(int)
cfvol["year"] = cfvol["year"].astype(int)
d2 = df.merge(cfvol[["id_stock","year","现金流波动"]], on=["id_stock","year"], how="left")
print(f"  现金流波动: {d2['现金流波动'].notna().sum():,} non-null")

r, n = run_fe(d2, "现金流波动", "Dea")
if r:
    c, se, p = fmt(r, "Dea")
    print(f"  Dea → 现金流波动: {c} {se} p={p} N={n:,}")
    results_all.append({"检验": "现金流波动降低", "被解释变量": "现金流波动", "Dea系数": c,
                       "标准误": se, "p值": p, "N": n})

# ================================================================
# TEST 3: 环境不确定性调节效应
# ================================================================
print("\n" + "─" * 60)
print("TEST 3: 环境不确定性调节效应")
print("─" * 60)

env, _ = pyreadstat.read_dta(DATA_AI / "260302主数据_含环境不确定.dta")
env["id_stock"] = env["id_stock"].astype(int)
env["year"] = env["year"].astype(int)
env_cols = [c for c in env.columns if c.startswith("eu_") or c.startswith("ln_eu")]
d3 = df.merge(env[["id_stock","year"] + env_cols], on=["id_stock","year"], how="left")

for eu_var in ["eu_total", "eu_circ"]:
    if eu_var not in d3.columns:
        continue
    print(f"\n  {eu_var}: {d3[eu_var].notna().sum():,} non-null")

    # 中位数分组
    med = d3[eu_var].median()
    d3["eu_high"] = (d3[eu_var] > med).astype(int)

    # 低不确定性组
    lo = d3[d3["eu_high"] == 0]
    r_lo, n_lo = run_fe(lo, "res", "Dea")
    c_lo, se_lo, p_lo = fmt(r_lo, "Dea") if r_lo else ("N/A","N/A","N/A")

    # 高不确定性组
    hi = d3[d3["eu_high"] == 1]
    r_hi, n_hi = run_fe(hi, "res", "Dea")
    c_hi, se_hi, p_hi = fmt(r_hi, "Dea") if r_hi else ("N/A","N/A","N/A")

    print(f"  {eu_var} 低组: {c_lo} {se_lo} p={p_lo} N={n_lo}")
    print(f"  {eu_var} 高组: {c_hi} {se_hi} p={p_hi} N={n_hi}")
    results_all.append({"检验": f"环境不确定性 ({eu_var})", "被解释变量": "res (低不确定性)",
                       "Dea系数": c_lo, "标准误": se_lo, "p值": p_lo, "N": n_lo})
    results_all.append({"检验": f"环境不确定性 ({eu_var})", "被解释变量": "res (高不确定性)",
                       "Dea系数": c_hi, "标准误": se_hi, "p值": p_hi, "N": n_hi})

# ================================================================
# TEST 4: 产业聚集度调节效应
# ================================================================
print("\n" + "─" * 60)
print("TEST 4: 产业聚集度调节效应")
print("─" * 60)

agg, _ = pyreadstat.read_dta(DATA_AI / "产业聚集度_城市行业层面.dta")
agg["year"] = agg["year"].astype(int)

# Bridge: get citycode from env uncertainty data (which has id_stock→citycode mapping)
# The panel's 'city' is numeric 1-433, not admin codes; use env data to bridge
env_for_city = env[["id_stock", "year", "citycode"]].drop_duplicates(subset=["id_stock", "year"])
d4 = df.merge(env_for_city, on=["id_stock", "year"], how="left")
print(f"  citycode 匹配: {d4['citycode'].notna().sum():,} / {len(d4):,}")

if d4["citycode"].notna().sum() > 100:
    # City-year level agglomeration (mean lq_emp per city-year)
    agg_city = agg.groupby(["citycode", "year"])["lq_emp"].mean().reset_index()
    agg_city["citycode"] = agg_city["citycode"].astype(float)
    d4["citycode"] = pd.to_numeric(d4["citycode"], errors="coerce")
    d4 = d4.merge(agg_city, on=["citycode", "year"], how="left")

    if "lq_emp" in d4.columns and d4["lq_emp"].notna().sum() > 100:
        print(f"  产业聚集度(lq_emp): {d4['lq_emp'].notna().sum():,} non-null")
        med_lq = d4["lq_emp"].median()

        lo = d4[d4["lq_emp"] <= med_lq]
        r_lo, n_lo = run_fe(lo, "res", "Dea")
        c_lo, se_lo, p_lo = fmt(r_lo, "Dea") if r_lo else ("N/A", "N/A", "N/A")

        hi = d4[d4["lq_emp"] > med_lq]
        r_hi, n_hi = run_fe(hi, "res", "Dea")
        c_hi, se_hi, p_hi = fmt(r_hi, "Dea") if r_hi else ("N/A", "N/A", "N/A")

        print(f"  低聚集度: {c_lo} {se_lo} p={p_lo} N={n_lo}")
        print(f"  高聚集度: {c_hi} {se_hi} p={p_hi} N={n_hi}")
        results_all.append({"检验": "产业聚集度", "被解释变量": "res (低聚集度)",
                           "Dea系数": c_lo, "标准误": se_lo, "p值": p_lo, "N": n_lo})
        results_all.append({"检验": "产业聚集度", "被解释变量": "res (高聚集度)",
                           "Dea系数": c_hi, "标准误": se_hi, "p值": p_hi, "N": n_hi})
    else:
        print("  [SKIP] lq_emp 合并失败 - 无有效匹配")
else:
    print("  [SKIP] citycode 匹配不足 - 无法进行产业聚集度检验")

# ================================================================
# TEST 5: Bert开源稳健性 — 绿色专利替代DV
# ================================================================
print("\n" + "─" * 60)
print("TEST 5: 绿色专利替代DV")
print("─" * 60)

gpat, _ = pyreadstat.read_dta(DATA_AI / "稳健性/创新水平/绿色专利.dta")
gpat["id_stock"] = gpat["id_stock"].astype(int)
gpat["year"] = gpat["year"].astype(int)
d5 = df.merge(gpat[["id_stock","year","LnGreen","LnGreen_Inv","LnGreen_Grant"]],
              on=["id_stock","year"], how="left")
print(f"  绿色专利: {d5['LnGreen'].notna().sum():,} non-null")

for v, label in [("LnGreen","绿色专利(总量)"), ("LnGreen_Inv","绿色发明专利"),
                 ("LnGreen_Grant","绿色专利授权")]:
    if v not in d5.columns:
        continue
    r, n = run_fe(d5, v, "Dea")
    if r:
        c, se, p = fmt(r, "Dea")
        print(f"  Dea → {label}: {c} {se} p={p} N={n:,}")
        results_all.append({"检验": "绿色专利替代DV", "被解释变量": label, "Dea系数": c,
                           "标准误": se, "p值": p, "N": n})

# ================================================================
# TEST 6: 倒U型检验 (Dea² → res)
# ================================================================
print("\n" + "─" * 60)
print("TEST 6: 倒U型检验")
print("─" * 60)

# Create Dea² centered
d6 = df.copy()
d6["Dea_c"] = d6["Dea"] - d6["Dea"].mean()
d6["Dea_c2"] = d6["Dea_c"] ** 2

# Linear + Quadratic (same sample)
use_ctrl = [c for c in CONTROLS if c in d6.columns]
s_base = d6[["res","Dea_c","Dea_c2"] + use_ctrl + ["id_stock","year"]].dropna()
s_base_idx = s_base.set_index(["id_stock","year"])

def fit_quad(data_idx, X_vars):
    """Fit PanelOLS with clustered SE, fallback to unclustered"""
    Y = data_idx["res"]
    X = data_idx[X_vars]
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)
    except:
        try:
            mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
            return mod.fit()
        except:
            return None

# Linear
res_lin = fit_quad(s_base_idx, ["Dea_c"] + use_ctrl)
if res_lin is not None:
    b1 = res_lin.params["Dea_c"]
    p1 = res_lin.pvalues["Dea_c"]
    print(f"  线性: Dea_c = {b1:+.4f}{star(p1)} (p={p1:.4f})")
else:
    print(f"  线性回归失败")
    b1, p1 = 0, 1

# Quadratic
res_quad = fit_quad(s_base_idx, ["Dea_c","Dea_c2"] + use_ctrl)
if res_quad is not None:
    b1q = res_quad.params["Dea_c"]
    b2q = res_quad.params["Dea_c2"]
    se_b2q = res_quad.std_errors["Dea_c2"]
    p_b2q = res_quad.pvalues["Dea_c2"]

    # Turning point
    if abs(b2q) > 1e-10:
        turn_c = -b1q / (2 * b2q)
        turn_raw = turn_c + d6["Dea"].mean()
    else:
        turn_c, turn_raw = float('inf'), float('inf')

    dea_min, dea_max = d6["Dea"].min(), d6["Dea"].max()
    in_range = dea_min <= turn_raw <= dea_max

    print(f"  二次项: Dea_c = {b1q:+.4f}, Dea_c² = {b2q:+.4f}{star(p_b2q)} (p={p_b2q:.4f})")
    print(f"  转折点 (Dea): {turn_raw:.4f}  样本范围: [{dea_min:.4f}, {dea_max:.4f}]")
    print(f"  转折点在样本内: {'是' if in_range else '否'}")

    if p_b2q < 0.05 and in_range:
        shape = "倒U型(凹)" if b2q < 0 else "正U型(凸)"
    elif p_b2q < 0.05:
        shape = "二次项显著但转折点在样本外"
    else:
        shape = "无显著非线性"

    print(f"  结论: {shape}")

    if res_lin is not None:
        results_all.append({"检验": "倒U型 (线性)", "被解释变量": "res",
                           "Dea系数": f"{b1:+.4f}{star(p1)}",
                           "标准误": f"({res_lin.std_errors['Dea_c']:.4f})",
                           "p值": f"{p1:.4f}", "N": len(s_base_idx)})
    results_all.append({"检验": "倒U型 (二次项)", "被解释变量": "res (Dea²)",
                       "Dea系数": f"{b2q:+.4f}{star(p_b2q)}",
                       "标准误": f"({se_b2q:.4f})", "p值": f"{p_b2q:.4f}",
                       "N": len(s_base_idx),
                       "转折点": f"{turn_raw:.4f}",
                       "形状": shape})
else:
    print(f"  二次项回归失败")

# ================================================================
# Save results
# ================================================================
print("\n" + "=" * 80)
print("保存结果...")

out_df = pd.DataFrame(results_all)
out_path = OUT / "additional_tests_results.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"已保存: {out_path}")
print(out_df.to_string(max_colwidth=50))

print(f"\n>>> 额外检验完成 <<<")
