"""
任务1 — 补全机制检验（论文表2）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

论文做法：Rajan-Zingales (1998) 交互项法
  Res = Dea + Dea × IC + Controls + Firm FE + Year FE
  Res = Breadth + Breadth × IC + Controls + Firm FE + Year FE
  Res = Depth + Depth × IC + Controls + Firm FE + Year FE
  （交易成本同理）

同时补充：
  1. Baron-Kenny 三步法中介检验（当前复现已有，重新确认）
  2. 交互项 vs 中介的方法论对比论证
  3. 报告主效应系数（论文表2严重缺失——未报告主效应）

数据来源：
  内部控制指数: ../dea-replication/data/raw/内部控制指数2000-2024.dta
  交易成本:      ../dea-replication/data/raw/交易成本2000-2024.dta
  面板数据:      ../dea-replication/output/panel_merged.csv
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import pyreadstat
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

# 引用原有项目的输出和数据
OLD_OUT  = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "output"
OLD_DATA = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "data" / "raw"

# ── 读取面板数据 ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(OLD_OUT / "panel_merged.csv")
df_raw["stkcode"] = df_raw["stkcode"].astype(str).str.zfill(6)

# ── 样本筛选 ────────────────────────────────────────────────────────────────────
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

# ── 读取内部控制指数 ──────────────────────────────────────────────────────────────
ic_df, _ = pyreadstat.read_dta(OLD_DATA / "内部控制指数2000-2024.dta",
                                usecols=["stkcd", "year", "IC1"])
ic_df["stkcode"] = ic_df["stkcd"].astype(int).astype(str).str.zfill(6)
ic_df["year"]    = ic_df["year"].astype(int)
ic_df["lnIC"]    = ic_df["IC1"]
ic_df = ic_df[["stkcode", "year", "lnIC"]]

# ── 读取交易成本 ────────────────────────────────────────────────────────────────
cost_df, _ = pyreadstat.read_dta(OLD_DATA / "交易成本2000-2024.dta")
cost_df["stkcode"] = cost_df["stkcd"].astype(int).astype(str).str.zfill(6)
cost_df["year"]    = cost_df["year"].astype(int)
cost_df = cost_df.rename(columns={"交易成本": "cost"})[["stkcode", "year", "cost"]]

# ── 合并中介变量 ────────────────────────────────────────────────────────────────
df = df.merge(ic_df,   on=["stkcode", "year"], how="left")
df = df.merge(cost_df, on=["stkcode", "year"], how="left")

print(f"合并后样本行数: {len(df):,}")
print(f"  lnIC  非空: {df['lnIC'].notna().sum():,}")
print(f"  cost  非空: {df['cost'].notna().sum():,}")

def make_sample(data, extra_drop_na=None):
    cols = (["Dea", "Breadth", "Depth"] + CONTROLS + [OUTCOME, "lnIC", "cost"]
            + ["stkcode", "year", "city_x"])
    d = data[[c for c in cols if c in data.columns]].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME])
    d = d.dropna(subset=["city"])
    if extra_drop_na:
        d = d.dropna(subset=extra_drop_na)
    return d


def run_fe(data, dep, indep, cluster_var="city"):
    """双向固定效应回归"""
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def fmt(coef, se, pval):
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars}\n({se:.4f})"


def print_coef(res, var):
    c, se, pv = res.params[var], res.std_errors[var], res.pvalues[var]
    stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
    print(f"  {var:<20} {c:>10.4f}{stars} ({se:.4f})  p={pv:.4f}")
    return {"var": var, "coef": c, "se": se, "pval": pv,
            "N": res.nobs, "r2_within": res.rsquared_within}


# ═══════════════════════════════════════════════════════════════════════════════
# 第一部分：Baron-Kenny 三步法中介检验（方法论正确做法）
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("第一部分：Baron-Kenny 三步法中介检验")
print("=" * 80)

bk_rows = []

for mechanism, mediator, drop_col in [
    ("IC", "lnIC", ["lnIC"]),
    ("Cost", "cost", ["cost"]),
]:
    s = make_sample(df, extra_drop_na=drop_col)
    if mechanism == "Cost":
        p1, p99 = s["cost"].quantile(0.01), s["cost"].quantile(0.99)
        s["cost"] = s["cost"].clip(lower=p1, upper=p99)

    print(f"\n{'─'*60}")
    print(f"机制: {mechanism} | 样本: {len(s):,} obs, {s['stkcode'].nunique():,} 家企业")
    print(f"{'─'*60}")

    for core in ["Dea", "Breadth", "Depth"]:
        print(f"\n  核心变量: {core}")

        # Step 1: X → Y
        r1 = run_fe(s, OUTCOME, [core] + CONTROLS)
        row = print_coef(r1, core)
        row.update({"mechanism": mechanism, "method": "BK_step1_X→Y", "core": core})
        bk_rows.append(row)

        # Step 2: X → M
        r2 = run_fe(s, mediator, [core] + CONTROLS)
        row = print_coef(r2, core)
        row.update({"mechanism": mechanism, "method": "BK_step2_X→M", "core": core})
        bk_rows.append(row)

        # Step 3: X + M → Y
        r3 = run_fe(s, OUTCOME, [core, mediator] + CONTROLS)
        row_x = print_coef(r3, core)
        row_x.update({"mechanism": mechanism, "method": "BK_step3_X→Y|M", "core": core})
        bk_rows.append(row_x)
        row_m = print_coef(r3, mediator)
        row_m.update({"mechanism": mechanism, "method": "BK_step3_M→Y|X", "core": core})
        bk_rows.append(row_m)

        # 判断中介链
        a_sig = r2.pvalues[core] < 0.05
        b_sig = r3.pvalues[mediator] < 0.05
        print(f"    → 中介链判断: Step2 (X→M) {'✅' if a_sig else '❌'}, "
              f"Step3 (M→Y|X) {'✅' if b_sig else '❌'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 第二部分：Rajan-Zingales 交互项法（论文表2的实际做法）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("第二部分：Rajan-Zingales 交互项法（严格复现论文表2）")
print("=" * 80)

rz_rows = []

for mechanism, mediator, drop_col in [
    ("IC", "lnIC", ["lnIC"]),
    ("Cost", "cost", ["cost"]),
]:
    s = make_sample(df, extra_drop_na=drop_col)
    if mechanism == "Cost":
        p1, p99 = s["cost"].quantile(0.01), s["cost"].quantile(0.99)
        s["cost"] = s["cost"].clip(lower=p1, upper=p99)

    # 构造交互项（去中心化以减少多重共线性）
    for core in ["Dea", "Breadth", "Depth"]:
        s[f"{core}_c"] = s[core] - s[core].mean()
    s[f"{mediator}_c"] = s[mediator] - s[mediator].mean()
    for core in ["Dea", "Breadth", "Depth"]:
        s[f"{core}_x_{mediator}"] = s[f"{core}_c"] * s[f"{mediator}_c"]

    print(f"\n{'─'*60}")
    print(f"机制: {mechanism} | 样本: {len(s):,} obs")
    print(f"{'─'*60}")

    for core in ["Dea", "Breadth", "Depth"]:
        print(f"\n  核心变量: {core}")

        # 完整交互项回归（包含主效应！论文表2严重缺失此信息）
        indep = [f"{core}_c", f"{mediator}_c", f"{core}_x_{mediator}"] + CONTROLS
        r = run_fe(s, OUTCOME, indep)

        for v in [f"{core}_c", f"{mediator}_c", f"{core}_x_{mediator}"]:
            row = print_coef(r, v)
            display_name = v.replace("_c", "").replace(f"{mediator}", f"×{mediator}")
            row.update({"mechanism": mechanism, "method": "RZ_interaction",
                        "core": core, "display_var": v})
            rz_rows.append(row)
        print(f"    N={r.nobs:,}  R²(within)={r.rsquared_within:.4f}")

        # 论文表2的简化做法（只报告交互项，不报告主效应——方法论错误！）
        # 但我们仍然严格复现它以对比
        indep_paper = [core, mediator, f"{core}_x_{mediator}"] + CONTROLS
        r_paper = run_fe(s, OUTCOME, indep_paper)
        row = print_coef(r_paper, f"{core}_x_{mediator}")
        row.update({"mechanism": mechanism, "method": "RZ_paper_style",
                    "core": core, "display_var": "interaction_only"})
        rz_rows.append(row)


# ═══════════════════════════════════════════════════════════════════════════════
# 第三部分：方法论对比 — 交互项显著 ≠ 中介效应存在
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("第三部分：交互项 vs 中介 — 方法论对比")
print("=" * 80)

print("""
关键方法论区别：
  Rajan-Zingales (1998) 交互项法：
    Res = β₁·Dea + β₂·M + β₃·Dea×M + Controls + FE
    检验的是：M 是否「调节」Dea→Res 的效应（调节效应）
    交互项显著 → 效应异质性，不能证明因果传导链 A→B→C

  Baron-Kenny (1986) 中介法：
    Step1: Res = c·Dea + Controls + FE           (总效应)
    Step2: M   = a·Dea + Controls + FE           (X→M 路径)
    Step3: Res = c'·Dea + b·M + Controls + FE    (直接效应 + M→Y 路径)
    检验的是：Dea 是否「通过 M」影响 Res（因果传导）
    三步均显著 + Sobel检验 → 才能声称中介效应

论文的错误：
  论文声称「数据应用深度通过加强内部控制提升韧性」
  但用的是交互项法（调节效应），不是中介法（因果传导）
  这两种方法论不能互相替代！
  交互项显著 ≠ 中介链存在！
""")

# 汇总判断
print("=" * 80)
print("中介链完整性判断（Baron-Kenny）")
print("=" * 80)
bk_df = pd.DataFrame(bk_rows)
for core in ["Dea", "Breadth", "Depth"]:
    for mech in ["IC", "Cost"]:
        subset = bk_df[(bk_df["core"] == core) & (bk_df["mechanism"] == mech)]
        step2 = subset[subset["method"] == "BK_step2_X→M"]
        step3 = subset[subset["method"] == "BK_step3_M→Y|X"]
        if len(step2) and len(step3):
            a_p = step2.iloc[0]["pval"]
            b_p = step3.iloc[0]["pval"]
            chain = "✅ 完整" if (a_p < 0.05 and b_p < 0.05) else \
                    "⚠️ 部分" if (a_p < 0.05 or b_p < 0.05) else "❌ 断裂"
            print(f"  {core:8s} → {mech:4s}: Step2 p={a_p:.4f}  Step3 p={b_p:.4f}  → {chain}")


# ═══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ═══════════════════════════════════════════════════════════════════════════════
bk_out = pd.DataFrame(bk_rows)
bk_out.to_csv(OUT / "mechanism_baron_kenny.csv", index=False, encoding="utf-8-sig")

rz_out = pd.DataFrame(rz_rows)
rz_out.to_csv(OUT / "mechanism_interaction_terms.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到 {OUT}/")
print("  mechanism_baron_kenny.csv — Baron-Kenny三步法结果")
print("  mechanism_interaction_terms.csv — Rajan-Zingales交互项法结果")

# ── 生成论文表2格式的汇总输出 ──────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("【论文表2格式 — 交互项法汇总】")
print("=" * 80)
print(f"{'机制':6s} {'核心X':10s} {'交互项':20s} {'系数':>12s} {'标准误':>10s} {'p值':>10s}")
print("-" * 70)
for _, row in rz_out[rz_out["method"] == "RZ_interaction"].iterrows():
    dv = row["display_var"]
    if "_x_" in dv:
        print(f"{row['mechanism']:6s} {row['core']:10s} {dv:20s} "
              f"{row['coef']:>12.4f} {row['se']:>10.4f} {row['pval']:>10.4f}")
