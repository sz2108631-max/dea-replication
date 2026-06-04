"""
任务6 — 排除替代解释（混淆政策）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

检验同期政策是否混淆了DEA→Res的因果效应：
  1. 供应链创新与应用试点城市（2018）
  2. 智能制造试点示范（2015-2018）
  3. 国家大数据综合试验区（2016）
  4. 数字经济创新发展试验区（2019）

方法：
  - 为每个政策构造 treatment × post 交互项
  - 将其加入基准回归，观察Dea系数是否稳定
  - 逐政策排除 + 同时排除所有政策

注：政策名单需用户提供。当前脚本使用框架代码，
    实际运行时需填充具体名单。
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

OLD_OUT = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "output"

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
    cols = (["Dea", "Breadth", "Depth"] + CONTROLS + [OUTCOME]
            + ["stkcode", "year", "city_x", "province"]
            + [c for c in data.columns if c.startswith("policy_")])
    available = [c for c in cols if c in data.columns]
    d = data[available].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME])
    return d.dropna(subset=["city"])


def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def fmt(coef, se, pval):
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars}\n({se:.4f})"


# ── 定义已知的混杂政策 ────────────────────────────────────────────────────────
# 用户可通过以下字典补充政策名单
# 格式: {"政策名": {"cities": [城市列表], "year": 政策起始年}}

# 以下为示例框架，需替换为实际名单
POLICIES = {
    "supply_chain_pilot": {
        "name": "供应链创新试点",
        "year": 2018,
        "cities": []  # TODO: 填充试点城市名单
    },
    "smart_mfg_pilot": {
        "name": "智能制造试点",
        "year": 2015,
        "cities": []  # TODO: 填充试点城市名单
    },
    "bigdata_zone": {
        "name": "大数据综合试验区",
        "year": 2016,
        "cities": []  # TODO: 填充试验区城市名单（贵州等）
    },
    "digital_economy_zone": {
        "name": "数字经济创新区",
        "year": 2019,
        "cities": []  # TODO: 填充创新区城市名单
    },
}


# ── 为每个政策构造处理变量 ───────────────────────────────────────────────────
def build_policy_vars(df, policies):
    """根据城市名单和起始年份构造 treatment × post 变量"""
    for key, info in policies.items():
        cities = info["cities"]
        start_year = info["year"]

        if not cities:
            print(f"  ⚠️  {info['name']}: 城市名单为空，跳过")
            continue

        # city 字段可能是字符串，需要匹配
        df[f"policy_{key}_treated"] = df["city"].isin(cities).astype(int)
        df[f"policy_{key}_post"] = (df["year"] >= start_year).astype(int)
        df[f"policy_{key}_did"] = df[f"policy_{key}_treated"] * df[f"policy_{key}_post"]

        n_treated = df[f"policy_{key}_treated"].sum()
        print(f"  {info['name']}: {len(cities)} 个城市, {n_treated:,} obs 标记为处理组")

    return df


# ── 运行 ──────────────────────────────────────────────────────────────────────
print("=" * 80)
print("排除替代解释 — 控制同期混淆政策")
print("=" * 80)

df = build_policy_vars(df, POLICIES)
sample = make_sample(df)
print(f"\n分析样本: {len(sample):,} obs, {sample['stkcode'].nunique():,} 家企业")

# 找到实际有数据的政策变量
policy_vars = [c for c in sample.columns if c.startswith("policy_") and c.endswith("_did")]

# ── 基准回归（无政策控制）─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. 基准回归（无政策控制）")
print("=" * 80)

r_base = run_fe(sample, OUTCOME, ["Dea"] + CONTROLS)
coef_base = r_base.params["Dea"]
se_base = r_base.std_errors["Dea"]
p_base = r_base.pvalues["Dea"]
print(f"  Dea: {fmt(coef_base, se_base, p_base)}   N={r_base.nobs:,}")

# ── 逐一加入政策控制 ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. 逐一加入政策控制")
print("=" * 80)

results = [{
    "检验": "基准(无政策控制)", "Dea系数": coef_base,
    "Dea标准误": se_base, "Dea p值": p_base,
    "系数变化%": 0.0, "N": r_base.nobs
}]

for pvar in policy_vars:
    policy_name = pvar.replace("policy_", "").replace("_did", "")
    info = POLICIES.get(policy_name, {})

    indep = ["Dea"] + [pvar] + CONTROLS
    r = run_fe(sample, OUTCOME, indep)
    coef_new = r.params["Dea"]
    change_pct = 100 * (coef_new - coef_base) / coef_base

    print(f"  + {info.get('name', policy_name)}: "
          f"Dea={coef_new:.4f}, 变化={change_pct:+.1f}%, "
          f"政策did={r.params.get(pvar, 0):.4f}, "
          f"p={r.pvalues.get(pvar, 1):.4f}")

    results.append({
        "检验": f"+ 控制{info.get('name', policy_name)}",
        "Dea系数": coef_new, "Dea标准误": r.std_errors["Dea"],
        "Dea p值": r.pvalues["Dea"], "系数变化%": change_pct, "N": r.nobs
    })

# ── 同时控制所有政策 ────────────────────────────────────────────────────────
if len(policy_vars) > 1:
    print("\n" + "=" * 80)
    print("3. 同时控制所有政策")
    print("=" * 80)
    indep = ["Dea"] + policy_vars + CONTROLS
    r_all = run_fe(sample, OUTCOME, indep)
    coef_all = r_all.params["Dea"]
    change_all = 100 * (coef_all - coef_base) / coef_base
    print(f"  同时控制: Dea={coef_all:.4f}, 变化={change_all:+.1f}%")

    results.append({
        "检验": "同时控制所有政策",
        "Dea系数": coef_all, "Dea标准误": r_all.std_errors["Dea"],
        "Dea p值": r_all.pvalues["Dea"], "系数变化%": change_all, "N": r_all.nobs
    })

    # 打印各政策系数
    print("\n  各政策估计效应:")
    for pvar in policy_vars:
        if pvar in r_all.params.index:
            coef = r_all.params[pvar]
            se = r_all.std_errors[pvar]
            pv = r_all.pvalues[pvar]
            policy_name = pvar.replace("policy_", "").replace("_did", "")
            info = POLICIES.get(policy_name, {})
            print(f"    {info.get('name', policy_name)}: {coef:.4f} ({se:.4f}) p={pv:.4f}")

# ── 系数稳定性评估 ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("系数稳定性评估")
print("=" * 80)
changes = [abs(r["系数变化%"]) for r in results[1:] if r["检验"] != "基准(无政策控制)"]
if changes:
    max_change = max(changes)
    print(f"  Dea系数最大变化: {max_change:.1f}%")
    if max_change < 10:
        print("  结论: ✅ Dea系数对同期政策控制不敏感，结果稳健")
    elif max_change < 30:
        print("  结论: ⚠️ Dea系数对部分政策控制有中度敏感性")
    else:
        print("  结论: ❌ Dea系数对政策控制高度敏感，存在混淆偏误")

# ── 保存 ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "confounding_policy_check.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已保存到 {OUT}/confounding_policy_check.csv")
print("\n注意：当前政策城市列表为空！需用户填充具体名单后重新运行。")
print("  政策城市列表在脚本 POLICIES 字典中配置。")
