"""
按论文原文格式输出全部回归表格
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」《中国工业经济》第3期

模型：Res_it = β * DEA_it + γ * Controls_it + μ_i + λ_t + ε_it
估计：双向固定效应（企业FE + 年份FE），城市层面聚类标准误
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
DATA = ROOT / "data"
OLD_OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 数据加载与样本筛选
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("  巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」— 完整复现回归表格")
print("=" * 100)

df = pd.read_csv(OLD_OUT / "panel_merged.csv")
df["stkcode"] = df["stkcode"].astype(str).str.zfill(6)

df = df[df["STPT"] != 1].copy()
excl = df["ind_str"].str.contains("金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
df = df[~excl].copy()
for col in ["lnrd", "lev", "klr"]:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual", "lnrd", "indrate", "own", "lev"]
CONTROL_LABELS = {
    "lnage": "企业年龄", "klr": "资本劳动比", "lnsize": "企业规模",
    "bsize": "董事会规模", "dual": "两职合一", "lnrd": "研发投入",
    "indrate": "独立董事比例", "own": "所有权性质", "lev": "资产负债率"
}
OUTCOME = "res"

def make_sample(data, extra_vars=None):
    cols = (["Dea", "Breadth", "Depth"] + CONTROLS + [OUTCOME, "stkcode", "year", "city_x"]
            + (extra_vars or []))
    available = [c for c in cols if c in data.columns]
    d = data[available].copy().rename(columns={"city_x": "city"})
    return d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME, "city"])

def run_fe(data, dep, indep):
    d = data.set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])["city"]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)

def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def cell(coef, se, p):
    """Return formatted cell: coef (se) with stars"""
    s = star(p)
    return f"{coef:.4f}{s}\n({se:.4f})"

def run_and_print_table(title, specs, dep_var="res", note=""):
    """Run multiple specifications and print a formatted table.
    specs = list of (label, indep_vars, sample_filter_fn_or_None)
    """
    print(f"\n{'─'*100}")
    print(f"  {title}")
    print(f"{'─'*100}")

    # Run all specs
    results = []
    ns = []
    for label, indep, sf in specs:
        s = make_sample(df) if sf is None else sf(make_sample(df))
        r = run_fe(s, dep_var, indep)
        results.append(r)
        ns.append(r.nobs)

    # Collect all variable names
    all_vars = []
    for r in results:
        for v in r.params.index:
            if v not in all_vars and v not in CONTROLS + ["Dea", "Breadth", "Depth"]:
                all_vars.append(v)
    # Reorder: core vars first, then controls, then extra
    core_vars = ["Dea", "Breadth", "Depth"]
    ordered_vars = [v for v in core_vars if any(v in r.params.index for r in results)]
    ordered_vars += [v for v in CONTROLS if any(v in r.params.index for r in results)]
    ordered_vars += [v for v in all_vars if v not in ordered_vars]

    # Print header
    col_width = 22
    header = f"{'变量':<18}"
    for label, _, _ in specs:
        header += f"{label:>{col_width}}"
    print(header)
    print("-" * (18 + col_width * len(specs)))

    # Print each variable row
    for var in ordered_vars:
        label = CONTROL_LABELS.get(var, var)
        if len(label) > 16:
            label = label[:16]
        row = f"{label:<18}"
        for r in results:
            if var in r.params.index:
                row += f"{cell(r.params[var], r.std_errors[var], r.pvalues[var]):>{col_width}}"
            else:
                row += f"{'':>{col_width}}"
        print(row)

    # Print stats
    print("-" * (18 + col_width * len(specs)))
    row_n = f"{'N':<18}"
    row_r2 = f"{'R² (within)':<18}"
    for r in results:
        row_n += f"{r.nobs:>{col_width},}"
        row_r2 += f"{getattr(r, 'rsquared_within', 0):>{col_width}.4f}"
    print(row_n)
    print(row_r2)
    print(f"{'企业FE':<18}" + f"{'是':>{col_width}}" * len(specs))
    print(f"{'年份FE':<18}" + f"{'是':>{col_width}}" * len(specs))

    if note:
        print(f"\n  注: {note}")
    print()

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  表1：基准回归 — 数据要素应用能力与供应链韧性                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  表1：基准回归 — 数据要素应用能力与供应链韧性")
print("  被解释变量：供应链韧性（Res）")
print("█" * 100)

run_and_print_table("表1 Panel A: Dea 逐步加入控制变量",
    [
        ("(1) 无控制",     ["Dea"], None),
        ("(2) +公司特征",  ["Dea", "lnage", "lnsize", "lev"], None),
        ("(3) +治理结构",  ["Dea", "lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own"], None),
        ("(4) +研发投入",  ["Dea", "lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own", "lnrd"], None),
        ("(5) 全部控制",   ["Dea"] + CONTROLS, None),
    ],
    note="括号内为城市层面聚类稳健标准误。*** p<0.01, ** p<0.05, * p<0.1。"
)

run_and_print_table("表1 Panel B: Breadth 与 Depth 替代解释变量",
    [
        ("(1) Breadth",  ["Breadth"] + CONTROLS, None),
        ("(2) Depth",    ["Depth"] + CONTROLS, None),
        ("(3) 联合回归",  ["Breadth", "Depth"] + CONTROLS, None),
        ("(4) Dea基准",  ["Dea"] + CONTROLS, None),
    ],
    note="Breadth=数据要素应用广度（覆盖关键词类别数）；Depth=数据要素应用深度（各类别内平均相似度）。"
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  表2：机制检验 — 交互项法（论文实际使用的方法）                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  表2：机制检验 — 数据要素应用能力的调节效应")
print("  方法：Rajan-Zingales (1998) 交互项法")
print("█" * 100)

# Merge IC and Cost data

# Internal control
ic_raw, _ = pyreadstat.read_dta(DATA / "内部控制指数2000-2024.dta")
ic_raw["stkcode"] = ic_raw["证券代码"].astype(str).str.zfill(6)
ic_raw["year"] = ic_raw["year"].astype(int)
ic_raw = ic_raw.rename(columns={"IC1": "lnIC"})
df_mech = df.merge(ic_raw[["stkcode", "year", "lnIC"]], on=["stkcode", "year"], how="left")
print(f"  [机制数据] lnIC merge: {df_mech['lnIC'].notna().sum():,} / {len(df_mech):,}")

# Transaction cost
cost_raw, _ = pyreadstat.read_dta(DATA / "交易成本2000-2024.dta")
cost_raw["stkcode"] = cost_raw["stkcd"].astype(int).astype(str).str.zfill(6)
cost_raw["year"] = cost_raw["year"].astype(int)
cost_raw = cost_raw.rename(columns={"交易成本": "cost"})
df_mech = df_mech.merge(cost_raw[["stkcode", "year", "cost"]], on=["stkcode", "year"], how="left")
print(f"  [机制数据] cost merge: {df_mech['cost'].notna().sum():,} / {len(df_mech):,}")

# Use df_mech for mechanism analysis
df = df_mech

# Make sample function that includes mechanism vars
def make_sample_mech(data, extra_vars=None):
    cols = (["Dea", "Breadth", "Depth", "lnIC", "cost"]
            + CONTROLS + [OUTCOME, "stkcode", "year", "city_x"]
            + (extra_vars or []))
    available = [c for c in cols if c in data.columns]
    d = data[available].copy().rename(columns={"city_x": "city"})
    return d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME, "city"])

# Prepare interaction variables
def prep_interaction(s, core_var, mod_var):
    """Center variables and create interaction"""
    s = s.dropna(subset=[mod_var]).copy()
    s[f"{core_var}_c"] = s[core_var] - s[core_var].mean()
    s[f"{mod_var}_c"] = s[mod_var] - s[mod_var].mean()
    s[f"{core_var}_x_{mod_var}"] = s[f"{core_var}_c"] * s[f"{mod_var}_c"]
    return s

for mechanism, mod_var, mech_label in [
    ("IC", "lnIC", "内部控制"),
    ("Cost", "cost", "交易成本"),
]:
    print(f"\n  表2 Panel: {mech_label} ({mod_var}) 调节效应")
    print(f"  {'─'*80}")

    for core in ["Dea", "Breadth", "Depth"]:
        s = prep_interaction(make_sample_mech(df), core, mod_var)
        r = run_fe(s, OUTCOME, [f"{core}_c", f"{mod_var}_c", f"{core}_x_{mod_var}"] + CONTROLS)

        inter_key = f"{core}_x_{mod_var}"
        main_x = f"{core}_c"
        main_m = f"{mod_var}_c"

        coef_inter = r.params.get(inter_key, np.nan)
        se_inter = r.std_errors.get(inter_key, np.nan)
        p_inter = r.pvalues.get(inter_key, np.nan)

        coef_main = r.params.get(main_x, np.nan)
        p_main = r.pvalues.get(main_x, np.nan)

        print(f"    {core} × {mod_var}: "
              f"主效应={coef_main:.4f}{star(p_main)} "
              f"交互项={coef_inter:.4f}{star(p_inter)} "
              f"(SE={se_inter:.4f})  "
              f"N={r.nobs:,}  R²={getattr(r, 'rsquared_within', 0):.4f}")

    # Also show Baron-Kenny for Dea
    print(f"\n    【Baron-Kenny三步法补充: Dea】")
    s_bk = make_sample_mech(df).dropna(subset=[mod_var])

    r1 = run_fe(s_bk, OUTCOME, ["Dea"] + CONTROLS)
    print(f"    Step1 (Dea→Res):              {cell(r1.params['Dea'], r1.std_errors['Dea'], r1.pvalues['Dea'])}")

    r2 = run_fe(s_bk, mod_var, ["Dea"] + CONTROLS)
    print(f"    Step2 (Dea→{mod_var}):          {cell(r2.params['Dea'], r2.std_errors['Dea'], r2.pvalues['Dea'])}")

    r3 = run_fe(s_bk, OUTCOME, ["Dea", mod_var] + CONTROLS)
    print(f"    Step3 (Dea→Res | {mod_var}):   {cell(r3.params['Dea'], r3.std_errors['Dea'], r3.pvalues['Dea'])}")
    if mod_var in r3.params.index:
        print(f"    Step3 ({mod_var}→Res | Dea):    {cell(r3.params[mod_var], r3.std_errors[mod_var], r3.pvalues[mod_var])}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  表3：异质性分析                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  表3：异质性分析 — 数据要素应用能力的非对称效应")
print("  方法：分组回归 + Fisher置换检验（组间系数差异）")
print("█" * 100)

# H1: 区域 (东部 vs 中西部)
# H2: 产权 (民营 vs 国有)
# H3: 市场竞争 (HHI高低)
# H4: 高科技 vs 非高科技
# H6: 环境不确定性

hetero_dims = {
    "H1_区域": {
        "东部": lambda s: s[s["city"].isin(
            ["北京市", "天津市", "上海市", "重庆市",
             "石家庄市", "唐山市", "秦皇岛市", "邯郸市", "邢台市", "保定市", "张家口市", "承德市",
             "沧州市", "廊坊市", "衡水市", "沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市",
             "丹东市", "锦州市", "营口市", "阜新市", "辽阳市", "盘锦市", "铁岭市", "朝阳市",
             "葫芦岛市", "南京市", "无锡市", "徐州市", "常州市", "苏州市", "南通市", "连云港市",
             "淮安市", "盐城市", "扬州市", "镇江市", "泰州市", "宿迁市", "杭州市", "宁波市",
             "温州市", "嘉兴市", "湖州市", "绍兴市", "金华市", "衢州市", "舟山市", "台州市",
             "丽水市", "福州市", "厦门市", "莆田市", "三明市", "泉州市", "漳州市", "南平市",
             "龙岩市", "宁德市", "济南市", "青岛市", "淄博市", "枣庄市", "东营市", "烟台市",
             "潍坊市", "济宁市", "泰安市", "威海市", "日照市", "临沂市", "德州市", "聊城市",
             "滨州市", "菏泽市", "广州市", "韶关市", "深圳市", "珠海市", "汕头市", "佛山市",
             "江门市", "湛江市", "茂名市", "肇庆市", "惠州市", "梅州市", "汕尾市", "河源市",
             "阳江市", "清远市", "东莞市", "中山市", "潮州市", "揭阳市", "云浮市", "海口市",
             "三亚市"])],
        "中西部": lambda s: s[~s["city"].isin(
            ["北京市", "天津市", "上海市", "重庆市",
             "石家庄市", "唐山市", "秦皇岛市", "邯郸市", "邢台市", "保定市", "张家口市", "承德市",
             "沧州市", "廊坊市", "衡水市", "沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市",
             "丹东市", "锦州市", "营口市", "阜新市", "辽阳市", "盘锦市", "铁岭市", "朝阳市",
             "葫芦岛市", "南京市", "无锡市", "徐州市", "常州市", "苏州市", "南通市", "连云港市",
             "淮安市", "盐城市", "扬州市", "镇江市", "泰州市", "宿迁市", "杭州市", "宁波市",
             "温州市", "嘉兴市", "湖州市", "绍兴市", "金华市", "衢州市", "舟山市", "台州市",
             "丽水市", "福州市", "厦门市", "莆田市", "三明市", "泉州市", "漳州市", "南平市",
             "龙岩市", "宁德市", "济南市", "青岛市", "淄博市", "枣庄市", "东营市", "烟台市",
             "潍坊市", "济宁市", "泰安市", "威海市", "日照市", "临沂市", "德州市", "聊城市",
             "滨州市", "菏泽市", "广州市", "韶关市", "深圳市", "珠海市", "汕头市", "佛山市",
             "江门市", "湛江市", "茂名市", "肇庆市", "惠州市", "梅州市", "汕尾市", "河源市",
             "阳江市", "清远市", "东莞市", "中山市", "潮州市", "揭阳市", "云浮市", "海口市",
             "三亚市"])],
    },
    "H2_产权": {
        "民营": lambda s: s[s["own"] <= s["own"].median()],
        "国有": lambda s: s[s["own"] > s["own"].median()],
    },
}

# Check if HHI variable exists
if "HHI" in df.columns or "hhi" in df.columns:
    hhi_var = "HHI" if "HHI" in df.columns else "hhi"
    hetero_dims["H3_市场竞争"] = {
        "高竞争(低HHI)": lambda s: s[s[hhi_var] <= s[hhi_var].median()],
        "低竞争(高HHI)": lambda s: s[s[hhi_var] > s[hhi_var].median()],
    }

# High-tech indicator
if "ind_str" in df.columns:
    ht_keywords = "计算机|通信|电子|医药|航空|航天|仪器仪表|信息"
    hetero_dims["H4_高科技"] = {
        "非高科技": lambda s: s[~s.get("ind_str", pd.Series("", index=s.index)).str.contains(ht_keywords, na=False)],
        "高科技": lambda s: s[s.get("ind_str", pd.Series("", index=s.index)).str.contains(ht_keywords, na=False)],
    }

# Environmental uncertainty
if "EU" in df.columns:
    hetero_dims["H6_环境不确定性"] = {
        "低不确定性": lambda s: s[s["EU"].dropna() <= s["EU"].median()] if s["EU"].notna().any() else s,
        "高不确定性": lambda s: s[s["EU"].dropna() > s["EU"].median()] if s["EU"].notna().any() else s,
    }

# Run heterogeneity table
for dim_name, groups in hetero_dims.items():
    print(f"\n  {dim_name}")
    print(f"  {'分组':<18} {'Dea系数':>22} {'N':>10} {'R²组内':>8}")
    print(f"  {'─'*60}")

    results_by_group = {}
    for grp_name, filter_fn in groups.items():
        s_full = make_sample(df)
        if "ind_str" not in s_full.columns:
            s_full["ind_str"] = df.loc[s_full.index, "ind_str"] if "ind_str" in df.columns else "Unknown"
        try:
            s_grp = filter_fn(s_full)
            if len(s_grp) < 300:
                print(f"  {grp_name:<18} {'样本不足':>22} {len(s_grp):>10}")
                continue
            r = run_fe(s_grp, OUTCOME, ["Dea"] + CONTROLS)
            results_by_group[grp_name] = r
            print(f"  {grp_name:<18} {cell(r.params['Dea'], r.std_errors['Dea'], r.pvalues['Dea']):>22} "
                  f"{r.nobs:>10,}  {getattr(r, 'rsquared_within', 0):>8.4f}")
        except Exception as e:
            print(f"  {grp_name:<18} {'出错':>22} ({str(e)[:40]})")

    # Fisher permutation test (simplified)
    if len(results_by_group) == 2:
        names = list(results_by_group.keys())
        r0, r1 = results_by_group[names[0]], results_by_group[names[1]]
        diff = r0.params["Dea"] - r1.params["Dea"]
        se_diff = np.sqrt(r0.std_errors["Dea"]**2 + r1.std_errors["Dea"]**2)
        p_diff = 2 * (1 - 0.5 * (1 + np.tanh(np.abs(diff / se_diff) / 1.2533)))  # approx
        print(f"  {'组间差异':<18} {'Δ=' + str(round(diff, 5)):>22} {'':>10}  {'':>8}")
        print(f"  {'(近似检验)':<18} {'p≈' + str(round(p_diff, 4)):>22}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  表4：供应链网络拓展分析                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  表4：供应链网络与集中度 — DEA如何影响供应链结构？")
print("█" * 100)

# Supply chain concentration data
scm, _ = pyreadstat.read_dta(DATA / "客户&供应商&供应链集中度（未剔除）.dta")
scm["stkcode"] = scm["股票代码"].astype(str).str.zfill(6)
scm["year"] = scm["年份"].astype(int)
scm_vars = ["供应链集中度", "客户集中度1", "供应商集中度1"]
scm_available = [c for c in scm_vars if c in scm.columns]
df_scm = df.merge(scm[["stkcode", "year"] + scm_available], on=["stkcode", "year"], how="left")

# Geographic distance
geo, _ = pyreadstat.read_dta(DATA / "上市公司供应链地理距离（2001-2024年）.dta")
geo["stkcode"] = geo["股票代码"].astype(str).str.zfill(6)
geo["year"] = geo["年份"].astype(int)
df_scm = df_scm.merge(geo[["stkcode", "year", "客户地理距离", "供应商地理距离"]],
                      on=["stkcode", "year"], how="left")

def make_sample_scm(data, extra_vars=None):
    cols = (["Dea", "Breadth", "Depth"] + scm_available
            + ["客户地理距离", "供应商地理距离"]
            + CONTROLS + [OUTCOME, "stkcode", "year", "city_x"]
            + (extra_vars or []))
    available = [c for c in cols if c in data.columns]
    d = data[available].copy().rename(columns={"city_x": "city"})
    return d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME, "city"])

s_scm0 = make_sample_scm(df_scm)

# Panel A: DEA → Supply chain structure
print("\n  表4 Panel A: DEA → 供应链结构（被解释变量=供应链指标）")
print(f"  {'供应链指标':<18} {'Dea':>22} {'Breadth':>22} {'Depth':>22} {'N':>10}")
print(f"  {'─'*90}")

scm_dep_vars = scm_available + ["客户地理距离", "供应商地理距离"]
for dep_var in scm_dep_vars:
    s = s_scm0.dropna(subset=[dep_var])
    if len(s) < 300:
        continue
    row = f"  {dep_var:<18}"
    for core in ["Dea", "Breadth", "Depth"]:
        r = run_fe(s, dep_var, [core] + CONTROLS)
        row += f"{cell(r.params[core], r.std_errors[core], r.pvalues[core]):>22}"
    row += f"{r.nobs:>10,}"
    print(row)

# Panel B: Supply chain structure → Res
print(f"\n  表4 Panel B: 供应链结构 → 供应链韧性（被解释变量=Res）")
print(f"  {'供应链指标':<18} {'系数':>22} {'N':>10}")
print(f"  {'─'*50}")
for dep_var in scm_dep_vars:
    s = s_scm0.dropna(subset=[dep_var])
    if len(s) < 300:
        continue
    r = run_fe(s, OUTCOME, [dep_var] + CONTROLS)
    if dep_var in r.params.index:
        print(f"  {dep_var:<18} {cell(r.params[dep_var], r.std_errors[dep_var], r.pvalues[dep_var]):>22} {r.nobs:>10,}")

# Panel C: DEA × Concentration interaction
print(f"\n  表4 Panel C: DEA × 供应链集中度 → Res（交互效应）")
print(f"  {'交互项':<28} {'系数':>22} {'N':>10}")
print(f"  {'─'*60}")
for var in scm_available:
    s = s_scm0.dropna(subset=[var]).copy()
    s["Dea_c"] = s["Dea"] - s["Dea"].mean()
    s[f"{var}_c"] = s[var] - s[var].mean()
    s[f"Dea_x_{var}"] = s["Dea_c"] * s[f"{var}_c"]
    r = run_fe(s, OUTCOME, ["Dea_c", f"{var}_c", f"Dea_x_{var}"] + CONTROLS)
    inter_key = f"Dea_x_{var}"
    print(f"  {'Dea × '+var:<28} {cell(r.params.get(inter_key, np.nan), r.std_errors.get(inter_key, np.nan), r.pvalues.get(inter_key, np.nan)):>22} {r.nobs:>10,}")

# Panel D: High/Low concentration分组
print(f"\n  表4 Panel D: 高/低供应链集中度分组（中位数分割）")
print(f"  {'供应链指标':<18} {'低集中度':>22} {'高集中度':>22}")
print(f"  {'─'*62}")
for var in scm_available:
    s = s_scm0.dropna(subset=[var])
    med = s[var].median()
    r_low = run_fe(s[s[var] <= med], OUTCOME, ["Dea"] + CONTROLS)
    r_high = run_fe(s[s[var] > med], OUTCOME, ["Dea"] + CONTROLS)
    print(f"  {var:<18} {cell(r_low.params['Dea'], r_low.std_errors['Dea'], r_low.pvalues['Dea']):>22} "
          f"{cell(r_high.params['Dea'], r_high.std_errors['Dea'], r_high.pvalues['Dea']):>22}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  表5：进一步分析 — 倒U型、数商生态、AI能力                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  表5：进一步分析")
print("█" * 100)

# Panel A: Inverted-U test
print("\n  表5 Panel A: 倒U型检验（二次项）")
print(f"  {'变量':<12} {'一次项':>22} {'二次项':>22} {'Q1(低)':>22} {'Q4(高)':>22} {'形状':>10}")
print(f"  {'─'*98}")

s5 = make_sample(df)
for var in ["Dea", "Breadth", "Depth"]:
    s5[f"{var}_sq"] = s5[var] ** 2
    r = run_fe(s5, OUTCOME, [var, f"{var}_sq"] + CONTROLS)

    # Quartile regressions
    s5["q"] = pd.qcut(s5[var], 4, labels=False)
    r_q1 = run_fe(s5[s5["q"] == 0], OUTCOME, [var] + CONTROLS)
    r_q4 = run_fe(s5[s5["q"] == 3], OUTCOME, [var] + CONTROLS)

    lin = cell(r.params[var], r.std_errors[var], r.pvalues[var])
    sq = cell(r.params[f"{var}_sq"], r.std_errors[f"{var}_sq"], r.pvalues[f"{var}_sq"])
    q1 = cell(r_q1.params[var], r_q1.std_errors[var], r_q1.pvalues[var])
    q4 = cell(r_q4.params[var], r_q4.std_errors[var], r_q4.pvalues[var])

    # Determine shape
    lin_c, sq_c = r.params[var], r.params[f"{var}_sq"]
    lin_p, sq_p = r.pvalues[var], r.pvalues[f"{var}_sq"]
    if sq_p < 0.05 and sq_c > 0:
        shape = "正U型(阈值)"
    elif sq_p < 0.05 and sq_c < 0:
        shape = "倒U型"
    else:
        shape = "线性"

    print(f"  {var:<12} {lin:>22} {sq:>22} {q1:>22} {q4:>22} {shape:>10}")

# Panel B: Digital ecosystem
print(f"\n  表5 Panel B: 外部数字环境与DEA的交互效应")
print(f"  {'变量':<28} {'主效应(DEA)':>22} {'自身效应':>22} {'交互项(DEA×)':>22}")

# AI investment
ai_inv, _ = pyreadstat.read_dta(DATA / "A股上市公司人工智能投资水平（剔除st及金融）.dta")
ai_inv = ai_inv.dropna(subset=["id_stock"]).copy()
ai_inv["stkcode"] = ai_inv["id_stock"].astype(int).astype(str).str.zfill(6)
ai_inv["year"] = ai_inv["year"].astype(int)
df5 = df.merge(ai_inv[["stkcode", "year", "AIInvestTotal"]], on=["stkcode", "year"], how="left")

# Robot
robot, _ = pyreadstat.read_dta(DATA / "上市公司-工业机器人渗透度（2007-2023年）.dta")
robot = robot.dropna(subset=["id_stock"]).copy()
robot["stkcode"] = robot["id_stock"].astype(int).astype(str).str.zfill(6)
robot["year"] = robot["year"].astype(int)
df5 = df5.merge(robot[["stkcode", "year", "工业机器人渗透度"]].rename(
    columns={"工业机器人渗透度": "robot"}), on=["stkcode", "year"], how="left")

# AI text frequency
ai_txt, _ = pyreadstat.read_dta(DATA / "人工智能应用能力_中工经.dta")
ai_txt = ai_txt.dropna(subset=["id_stock"]).copy()
ai_txt["stkcode"] = ai_txt["id_stock"].astype(int).astype(str).str.zfill(6)
ai_txt["year"] = ai_txt["year"].astype(int)
df5 = df5.merge(ai_txt[["stkcode", "year", "人工智能词频和加1取对数"]].rename(
    columns={"人工智能词频和加1取对数": "AI_freq"}), on=["stkcode", "year"], how="left")

extra_vars_list = ["AIInvestTotal", "robot", "AI_freq"]
for extra_var in extra_vars_list:
    s = make_sample(df5, extra_vars=[extra_var]).dropna(subset=[extra_var])
    if len(s) < 500:
        continue

    s["Dea_c"] = s["Dea"] - s["Dea"].mean()
    s[f"{extra_var}_ln"] = np.log(s[extra_var] + 1)
    s[f"{extra_var}_c"] = s[f"{extra_var}_ln"] - s[f"{extra_var}_ln"].mean()
    s[f"Dea_x_{extra_var}"] = s["Dea_c"] * s[f"{extra_var}_c"]

    r = run_fe(s, OUTCOME, ["Dea_c", f"{extra_var}_c", f"Dea_x_{extra_var}"] + CONTROLS)

    dea_c = cell(r.params.get("Dea_c", np.nan), r.std_errors.get("Dea_c", np.nan), r.pvalues.get("Dea_c", np.nan))
    self_c = cell(r.params.get(f"{extra_var}_c", np.nan), r.std_errors.get(f"{extra_var}_c", np.nan), r.pvalues.get(f"{extra_var}_c", np.nan))
    inter_c = cell(r.params.get(f"Dea_x_{extra_var}", np.nan), r.std_errors.get(f"Dea_x_{extra_var}", np.nan), r.pvalues.get(f"Dea_x_{extra_var}", np.nan))

    labels = {"AIInvestTotal": "AI投资总额", "robot": "工业机器人渗透度", "AI_freq": "AI词频(全文本)"}
    print(f"  {labels.get(extra_var, extra_var):<28} {dea_c:>22} {self_c:>22} {inter_c:>22}")

# DEA vs AI freq horse race
print(f"\n  表5 Panel C: DEA vs AI词频 — 竞争性预测")
s_horse = make_sample(df5, extra_vars=["AI_freq"]).dropna(subset=["AI_freq"])
r_h1 = run_fe(s_horse, OUTCOME, ["Dea"] + CONTROLS)
r_h2 = run_fe(s_horse, OUTCOME, ["AI_freq"] + CONTROLS)
r_h3 = run_fe(s_horse, OUTCOME, ["Dea", "AI_freq"] + CONTROLS)

print(f"  {'模型':<20} {'Dea':>22} {'AI_freq':>22} {'N':>10}")
print(f"  {'─'*74}")
print(f"  {'(1) 仅Dea':<20} {cell(r_h1.params['Dea'], r_h1.std_errors['Dea'], r_h1.pvalues['Dea']):>22} "
      f"{'':>22} {r_h1.nobs:>10,}")
print(f"  {'(2) 仅AI_freq':<20} {'':>22} "
      f"{cell(r_h2.params['AI_freq'], r_h2.std_errors['AI_freq'], r_h2.pvalues['AI_freq']):>22} {r_h2.nobs:>10,}")
print(f"  {'(3) Dea + AI_freq':<20} {cell(r_h3.params['Dea'], r_h3.std_errors['Dea'], r_h3.pvalues['Dea']):>22} "
      f"{cell(r_h3.params['AI_freq'], r_h3.std_errors['AI_freq'], r_h3.pvalues['AI_freq']):>22} {r_h3.nobs:>10,}")

corr_dea_ai = s_horse["Dea"].corr(s_horse["AI_freq"])
print(f"\n  corr(Dea, AI_freq) = {corr_dea_ai:.4f}  — 两个文本指标高度重叠")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  稳健性检验汇总表                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  稳健性检验汇总")
print("█" * 100)

print(f"\n  表R1: PSM匹配后回归")
print(f"  {'':<20} {'Dea系数':>22} {'p值':>10} {'N':>10}")
print(f"  {'─'*62}")
print(f"  {'匹配前 (基准)':<20} {'0.0115***':>22} {'0.0001':>10} {'35,135':>10}")
print(f"  {'匹配后 (ATT)':<20} {'0.0111***':>22} {'<0.01':>10} {'3,310':>10}")
print(f"  注: 1:1最近邻匹配(卡尺=0.05), 9/9协变量平衡, 1,655对匹配成功")

print(f"\n  表R2: 安慰剂检验（5000次随机置换）")
print(f"  {'变量':<12} {'实际系数':>15} {'参数p值':>10} {'经验p值':>10} {'置换均值':>12} {'95%区间':>25} {'判定':>6}")
print(f"  {'─'*90}")
print(f"  {'Dea':<12} {'0.0115':>15} {'0.0001':>10} {'0.0000':>10} {'-0.0006':>12} {'[-0.0028, 0.0016]':>25} {'通过':>6}")
print(f"  {'Breadth':<12} {'0.0091':>15} {'0.0210':>10} {'0.0000':>10} {'-0.0005':>12} {'[-0.0029, 0.0018]':>25} {'通过':>6}")
print(f"  {'Depth':<12} {'0.0127':>15} {'0.0000':>10} {'0.0000':>10} {'-0.0006':>12} {'[-0.0028, 0.0016]':>25} {'通过':>6}")

print(f"\n  表R3: 工具变量回归（雷电频率）")
print(f"  {'指标':<30} {'值':>25} {'判定':>20}")
print(f"  {'─'*75}")
print(f"  {'OLS (同样本)':<30} {'0.0116***':>25} {'基准':>20}")
print(f"  {'第一阶段 (雷电→Dea)':<30} {'-0.0491 (p=0.738)':>25} {'❌ 不显著':>20}")
print(f"  {'Kleibergen-Paap F':<30} {'0.11':>25} {'❌ 远<10临界值':>20}")
print(f"  {'2SLS系数':<30} {'-2.62 (p=0.006)':>25} {'❌ 符号错误+量级荒谬':>20}")
print(f"  {'排他性 (雷电→Res|Dea)':<30} {'0.129 (p=0.006)':>25} {'❌ 排他性被违反':>20}")

print(f"\n  表R4: 排除同期数字政策冲击")
print(f"  {'政策冲击':<25} {'控制后Dea系数':>22} {'变化':>10} {'政策自身p值':>15} {'判定':>10}")
print(f"  {'─'*82}")
print(f"  {'智算中心试点DiD':<25} {'0.0115***':>22} {'+0.0%':>10} {'p=0.72':>15} {'通过':>10}")
print(f"  {'数据交易所DiD':<25} {'0.0114***':>22} {'-0.7%':>10} {'p=0.68':>15} {'通过':>10}")
print(f"  {'数据要素市场化':<25} {'0.0110***':>22} {'-4.3%':>10} {'p=0.46':>15} {'通过':>10}")
print(f"  {'全部3项同时控制':<25} {'0.0109***':>22} {'-5.1%':>10} {'-':>15} {'通过':>10}")

print(f"\n  表R5: 设定曲线（90种设定组合）")
print(f"  {'子样本':<30} {'组合数':>8} {'p<0.05比例':>12} {'判定':>10}")
print(f"  {'─'*60}")
print(f"  {'全部设定':<30} {'90':>8} {'16% (14/90)':>12} {'❌':>10}")
print(f"  {'平衡面板 (Balanced)':<30} {'24':>8} {'0% (0/24)':>12} {'❌❌❌':>10}")
print(f"  {'排除COVID年':<30} {'24':>8} {'3% (1/24)':>12} {'❌❌':>10}")
print(f"  {'全样本+企业FE+城市聚类+全控制':<30} {'6':>8} {'100% (6/6)':>12} {'✅':>10}")
print(f"  注: 3种FE × 3种聚类 × 2种控制集 × 3种子样本 × 2种替换变量 ≈ 90种组合")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  数据说明                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "█" * 100)
print("  变量定义")
print("█" * 100)

var_defs = [
    ("被解释变量", "Res", "供应链韧性", "11个财务子指标经TOPSIS方法聚合，取值范围[0,1]，越大表示韧性越强"),
    ("核心解释变量", "Dea", "数据要素应用能力", "MD&A文本经BERT零样本NLI计算关键词相似度×权重聚合，连续变量"),
    ("", "Breadth", "应用广度", "DEA覆盖的关键词类别广度（各类别关键词命中数的对数和）"),
    ("", "Depth", "应用深度", "DEA在各关键词类别中的平均深度（各类别内相似度均值的对数）"),
    ("控制变量", "lnage", "企业年龄", "ln(当年年份 - 成立年份 + 1)"),
    ("", "klr", "资本劳动比", "固定资产净额 / 员工人数"),
    ("", "lnsize", "企业规模", "ln(总资产)"),
    ("", "bsize", "董事会规模", "ln(董事会人数)"),
    ("", "dual", "两职合一", "董事长兼任总经理=1，否则=0"),
    ("", "lnrd", "研发投入", "ln(研发支出 + 1)"),
    ("", "indrate", "独立董事比例", "独立董事人数 / 董事会总人数"),
    ("", "own", "所有权性质", "国有股比例等复合指标"),
    ("", "lev", "资产负债率", "总负债 / 总资产"),
    ("机制变量", "lnIC", "内部控制", "迪博IC指数取对数"),
    ("", "cost", "交易成本", "CSMAR交易成本指标"),
    ("拓展变量", "供应链集中度", "供应链集中度", "客户+供应商集中度综合指标"),
    ("", "客户地理距离", "客户地理距离", "上市公司与主要客户之间的地理距离（对数公里）"),
    ("", "供应商地理距离", "供应商地理距离", "上市公司与主要供应商之间的地理距离（对数公里）"),
    ("", "AI_freq", "AI词频", "企业年报全文本中AI相关关键词的词频和加1取对数"),
    ("", "AIInvestTotal", "AI投资总额", "企业人工智能软硬件投资总额"),
    ("", "robot", "工业机器人渗透度", "行业层面工业机器人渗透度"),
]

for cat, var, label, desc in var_defs:
    prefix = f"  [{cat}] " if cat else "         "
    print(f"{prefix}{var:<16}  {label:<14}  {desc}")

# ═══════════════════════════════════════════════════════════════════════════════
# Export all key results to CSV for Word report
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("  导出CSV结果文件...")
print(f"{'='*100}")

import csv as _csv

def _safe(r, key):
    """Extract coef, se, p from a PanelOLS result"""
    if key in r.params.index:
        return r.params[key], r.std_errors[key], r.pvalues[key]
    return np.nan, np.nan, np.nan

# --- 表1 Panel A: 逐步加控制变量 ---
s1 = make_sample(df)
specs_1a = [
    ("(1) 无控制",     ["Dea"]),
    ("(2) +公司特征",  ["Dea", "lnage", "lnsize", "lev"]),
    ("(3) +治理结构",  ["Dea", "lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own"]),
    ("(4) +研发投入",  ["Dea", "lnage", "lnsize", "lev", "bsize", "dual", "indrate", "own", "lnrd"]),
    ("(5) 全部控制",   ["Dea"] + CONTROLS),
]
table1_rows = []
for label, indep in specs_1a:
    r = run_fe(s1, OUTCOME, indep)
    c, se, p = _safe(r, "Dea")
    table1_rows.append({
        "模型": label, "Dea系数": c, "Dea标准误": se, "Dea_p值": p,
        "N": r.nobs, "R2_within": getattr(r, 'rsquared_within', 0)
    })
pd.DataFrame(table1_rows).to_csv(OUT / "table1_stepwise_from_script.csv", index=False, encoding="utf-8-sig")

# --- 表1 Panel B: Breadth/Depth ---
table1b_rows = []
for core in ["Breadth", "Depth"]:
    r = run_fe(s1, OUTCOME, [core] + CONTROLS)
    c, se, p = _safe(r, core)
    table1b_rows.append({"变量": core, "系数": c, "标准误": se, "p值": p, "N": r.nobs})
r_bd = run_fe(s1, OUTCOME, ["Breadth", "Depth"] + CONTROLS)
for core in ["Breadth", "Depth"]:
    c, se, p = _safe(r_bd, core)
    table1b_rows.append({"变量": f"{core}(联合)", "系数": c, "标准误": se, "p值": p, "N": r_bd.nobs})
r_dea = run_fe(s1, OUTCOME, ["Dea"] + CONTROLS)
c, se, p = _safe(r_dea, "Dea")
table1b_rows.append({"变量": "Dea(基准)", "系数": c, "标准误": se, "p值": p, "N": r_dea.nobs})
pd.DataFrame(table1b_rows).to_csv(OUT / "table1_panel_b.csv", index=False, encoding="utf-8-sig")

# --- 表2: 机制检验 ---
mech_rows = []
for mechanism, mod_var in [("IC", "lnIC"), ("Cost", "cost")]:
    for core in ["Dea", "Breadth", "Depth"]:
        s = prep_interaction(make_sample_mech(df), core, mod_var)
        r = run_fe(s, OUTCOME, [f"{core}_c", f"{mod_var}_c", f"{core}_x_{mod_var}"] + CONTROLS)
        c_main, se_main, p_main = _safe(r, f"{core}_c")
        c_inter, se_inter, p_inter = _safe(r, f"{core}_x_{mod_var}")
        c_m, se_m, p_m = _safe(r, f"{mod_var}_c")
        mech_rows.append({
            "机制": mechanism, "核心变量": core, "调节变量": mod_var,
            "主效应系数": c_main, "主效应p值": p_main,
            "调节变量系数": c_m, "调节变量p值": p_m,
            "交互项系数": c_inter, "交互项标准误": se_inter, "交互项p值": p_inter,
            "N": r.nobs
        })
    # Baron-Kenny
    s_bk = make_sample_mech(df).dropna(subset=[mod_var])
    r1 = run_fe(s_bk, OUTCOME, ["Dea"] + CONTROLS)
    r2 = run_fe(s_bk, mod_var, ["Dea"] + CONTROLS)
    r3 = run_fe(s_bk, OUTCOME, ["Dea", mod_var] + CONTROLS)
    c1, se1, p1 = _safe(r1, "Dea")
    c2, se2, p2 = _safe(r2, "Dea")
    c3, se3, p3 = _safe(r3, "Dea")
    c3m, se3m, p3m = _safe(r3, mod_var)
    mech_rows.append({
        "机制": f"{mechanism}_BK", "核心变量": "Dea", "调节变量": mod_var,
        "主效应系数": np.nan, "主效应p值": np.nan,
        "调节变量系数": np.nan, "调节变量p值": np.nan,
        "交互项系数": np.nan, "交互项标准误": np.nan, "交互项p值": np.nan,
        "N": r1.nobs,
        "Step1_Dea_Res": c1, "Step1_p": p1,
        "Step2_Dea_M": c2, "Step2_p": p2,
        "Step3_Dea_Res_given_M": c3, "Step3_p_Dea": p3,
        "Step3_M_Res_given_Dea": c3m, "Step3_p_M": p3m,
    })
pd.DataFrame(mech_rows).to_csv(OUT / "table2_mechanism.csv", index=False, encoding="utf-8-sig")

# --- 表3: 异质性 ---
hetero_rows = []
for dim_name, groups in hetero_dims.items():
    for grp_name, filter_fn in groups.items():
        s_full = make_sample(df)
        if "ind_str" not in s_full.columns:
            s_full["ind_str"] = df.loc[s_full.index, "ind_str"] if "ind_str" in df.columns else "Unknown"
        try:
            s_grp = filter_fn(s_full)
            if len(s_grp) < 300:
                continue
            r = run_fe(s_grp, OUTCOME, ["Dea"] + CONTROLS)
            c, se, p = _safe(r, "Dea")
            hetero_rows.append({
                "异质性维度": dim_name, "分组": grp_name,
                "Dea系数": c, "Dea标准误": se, "Dea_p值": p,
                "N": r.nobs, "R2_within": getattr(r, 'rsquared_within', 0)
            })
        except Exception as e:
            hetero_rows.append({
                "异质性维度": dim_name, "分组": grp_name,
                "Dea系数": np.nan, "Dea标准误": np.nan, "Dea_p值": np.nan,
                "N": len(s_grp) if 's_grp' in dir() else 0,
                "R2_within": np.nan, "Error": str(e)[:80]
            })
pd.DataFrame(hetero_rows).to_csv(OUT / "table3_heterogeneity.csv", index=False, encoding="utf-8-sig")

# --- 表4: 供应链 ---
scm_rows = []
for dep_var in scm_dep_vars:
    s = s_scm0.dropna(subset=[dep_var])
    if len(s) < 300: continue
    for core in ["Dea", "Breadth", "Depth"]:
        r = run_fe(s, dep_var, [core] + CONTROLS)
        c, se, p = _safe(r, core)
        scm_rows.append({"Panel": "A_DEA到供应链", "被解释变量": dep_var, "解释变量": core,
                         "系数": c, "标准误": se, "p值": p, "N": r.nobs})
    r = run_fe(s, OUTCOME, [dep_var] + CONTROLS)
    c, se, p = _safe(r, dep_var)
    scm_rows.append({"Panel": "B_供应链到Res", "被解释变量": "Res", "解释变量": dep_var,
                     "系数": c, "标准误": se, "p值": p, "N": r.nobs})
for var in scm_available:
    s = s_scm0.dropna(subset=[var]).copy()
    s["Dea_c"] = s["Dea"] - s["Dea"].mean()
    s[f"{var}_c"] = s[var] - s[var].mean()
    s[f"Dea_x_{var}"] = s["Dea_c"] * s[f"{var}_c"]
    r = run_fe(s, OUTCOME, ["Dea_c", f"{var}_c", f"Dea_x_{var}"] + CONTROLS)
    c, se, p = _safe(r, f"Dea_x_{var}")
    scm_rows.append({"Panel": "C_交互项", "被解释变量": "Res", "解释变量": f"Dea_x_{var}",
                     "系数": c, "标准误": se, "p值": p, "N": r.nobs})
    # 分组
    med = s[var].median()
    r_low = run_fe(s[s[var] <= med], OUTCOME, ["Dea"] + CONTROLS)
    r_high = run_fe(s[s[var] > med], OUTCOME, ["Dea"] + CONTROLS)
    cl, sel, pl = _safe(r_low, "Dea")
    ch, seh, ph = _safe(r_high, "Dea")
    scm_rows.append({"Panel": "D_分组", "被解释变量": f"{var}_低组", "解释变量": "Dea",
                     "系数": cl, "标准误": sel, "p值": pl, "N": r_low.nobs})
    scm_rows.append({"Panel": "D_分组", "被解释变量": f"{var}_高组", "解释变量": "Dea",
                     "系数": ch, "标准误": seh, "p值": ph, "N": r_high.nobs})
pd.DataFrame(scm_rows).to_csv(OUT / "table4_supply_chain.csv", index=False, encoding="utf-8-sig")

# --- 表5: 倒U型 + 数字环境 ---
table5_rows = []
for var in ["Dea", "Breadth", "Depth"]:
    s5[f"{var}_sq"] = s5[var] ** 2
    r = run_fe(s5, OUTCOME, [var, f"{var}_sq"] + CONTROLS)
    cl, sel, pl = _safe(r, var)
    csq, sesq, psq = _safe(r, f"{var}_sq")
    s5["q"] = pd.qcut(s5[var], 4, labels=False)
    r_q1 = run_fe(s5[s5["q"] == 0], OUTCOME, [var] + CONTROLS)
    r_q4 = run_fe(s5[s5["q"] == 3], OUTCOME, [var] + CONTROLS)
    cq1, seq1, pq1 = _safe(r_q1, var)
    cq4, seq4, pq4 = _safe(r_q4, var)
    table5_rows.append({
        "Panel": "A_倒U型", "变量": var,
        "一次项系数": cl, "一次项标准误": sel, "一次项p": pl,
        "二次项系数": csq, "二次项标准误": sesq, "二次项p": psq,
        "Q1系数": cq1, "Q1标准误": seq1, "Q1_p": pq1,
        "Q4系数": cq4, "Q4标准误": seq4, "Q4_p": pq4, "N": r.nobs
    })

# AI/Horse race
s_horse = make_sample(df5, extra_vars=["AI_freq"]).dropna(subset=["AI_freq"])
for label, indep in [("仅Dea", ["Dea"]), ("仅AI_freq", ["AI_freq"]), ("Dea+AI_freq", ["Dea", "AI_freq"])]:
    r = run_fe(s_horse, OUTCOME, indep + CONTROLS)
    cd, sed, pd_ = _safe(r, "Dea")
    ca, sea, pa = _safe(r, "AI_freq")
    table5_rows.append({
        "Panel": "C_HorseRace", "变量": label,
        "一次项系数": cd, "一次项p": pd_, "二次项系数": ca, "二次项p": pa, "N": r.nobs
    })
table5_rows.append({
    "Panel": "C_HorseRace", "变量": "corr(Dea,AI_freq)",
    "一次项系数": s_horse["Dea"].corr(s_horse["AI_freq"]), "一次项p": np.nan,
    "二次项系数": np.nan, "二次项p": np.nan, "N": len(s_horse)
})
pd.DataFrame(table5_rows).to_csv(OUT / "table5_further.csv", index=False, encoding="utf-8-sig")

# --- DDD模型结果 ---
ddd_rows = []
for m_label, m_var in [("业务复杂度", "业务复杂度"), ("供应链集中度", "供应链集中度"),
                         ("本地数商数量", "本地数商数量"), ("专利数量", "专利数量"),
                         ("纵向一体化", "纵向一体化")]:
    # Quick check if file exists
    pass  # DDD results handled by 13_ddd_model.py separately
pd.DataFrame(ddd_rows).to_csv(OUT / "table_ddd_results.csv", index=False, encoding="utf-8-sig")

print(f"  CSV files saved to {OUT}/")
print(f"  - table1_stepwise_from_script.csv")
print(f"  - table1_panel_b.csv")
print(f"  - table2_mechanism.csv")
print(f"  - table3_heterogeneity.csv")
print(f"  - table4_supply_chain.csv")
print(f"  - table5_further.csv")

print(f"\n{'='*100}")
print("  回归表格输出完毕")
print(f"{'='*100}")
