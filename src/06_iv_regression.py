"""
任务3 — IV工具变量回归（雷电频率）
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

IV: 雷击频率（地级市层面）
逻辑：雷电→通信中断→数字基础设施受损→企业DEA下降（第一阶段）
      雷电频率不直接影响企业供应链韧性（排他性约束）

检验内容：
  1. 第一阶段：Dea ~ lightning + Controls + FE（预期负向）
  2. 第二阶段：Res ~ Dea_hat + Controls + FE
  3. 弱IV检验：Kleibergen-Paap F统计量
  4. 排他性约束讨论
  5. Reduced-form: Res ~ lightning + Controls + FE
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
import pyreadstat
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)

OLD_OUT = Path(__file__).resolve().parent.parent.parent / "dea-replication" / "output"

# ── 读取面板数据 ──────────────────────────────────────────────────────────────
df = pd.read_csv(OLD_OUT / "panel_merged.csv")
df["stkcode"] = df["stkcode"].astype(str).str.zfill(6)

# ── 样本筛选 ──────────────────────────────────────────────────────────────────
df = df[df["STPT"] != 1].copy()
excl = df["ind_str"].str.contains(
    "金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
df = df[~excl].copy()

for col in ["lnrd", "lev", "klr"]:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

# ── 读取雷电数据 ──────────────────────────────────────────────────────────────
lightning, _ = pyreadstat.read_dta(DATA / "lightning.dta")
print(f"雷电数据: {len(lightning)} 个城市")
print(f"雷电频率分布: mean={lightning['雷击频率'].mean():.2f}, "
      f"std={lightning['雷击频率'].std():.2f}, "
      f"min={lightning['雷击频率'].min():.2f}, max={lightning['雷击频率'].max():.2f}")

# 合并 — citycode 匹配
lightning["citycode"] = lightning["citycode"].astype(float)
df["citycode"] = df["citycode"].astype(float)
df = df.merge(lightning[["citycode", "雷击频率"]], on="citycode", how="left")
df["ln_lightning"] = np.log(df["雷击频率"] + 1)

n_merged = df["雷击频率"].notna().sum()
print(f"雷电数据合并: {n_merged:,} / {len(df):,} obs ({100*n_merged/len(df):.1f}%)")
print(f"  覆盖 {df[df['雷击频率'].notna()]['stkcode'].nunique():,} 家企业")

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual",
            "lnrd", "indrate", "own", "lev"]
OUTCOME  = "res"


def make_sample(data, extra_drop=None):
    cols = (["Dea", "Breadth", "Depth"] + CONTROLS + [OUTCOME]
            + ["stkcode", "year", "city_x", "雷击频率", "ln_lightning"]
            + (extra_drop or []))
    available = [c for c in cols if c in data.columns]
    d = data[available].copy()
    d = d.rename(columns={"city_x": "city"})
    d = d.dropna(subset=["Dea"] + CONTROLS + [OUTCOME, "雷击频率"])
    return d.dropna(subset=["city"])


def run_fe(data, dep, indep, cluster_var="city"):
    d = data.copy().set_index(["stkcode", "year"])
    fml = f"{dep} ~ {' + '.join(indep)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(fml, data=d, drop_absorbed=True)
    clus = data.set_index(["stkcode", "year"])[cluster_var]
    return mod.fit(cov_type="clustered", cluster_entity=False, clusters=clus)


def fmt(coef, se, pval):
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:.4f}{stars} ({se:.4f})"


s = make_sample(df)
print(f"\nIV分析样本: {len(s):,} obs, {s['stkcode'].nunique():,} 家企业")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. OLS基准（可比较样本）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. OLS基准回归（IV样本）")
print("=" * 80)
r_ols = run_fe(s, OUTCOME, ["Dea"] + CONTROLS)
print(f"  Dea: {fmt(r_ols.params['Dea'], r_ols.std_errors['Dea'], r_ols.pvalues['Dea'])}  "
      f"N={r_ols.nobs:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reduced-form: Res ~ lightning + Controls
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. Reduced-form: Res ~ ln_lightning + Controls")
print("=" * 80)
r_rf = run_fe(s, OUTCOME, ["ln_lightning"] + CONTROLS)
print(f"  ln_lightning: {fmt(r_rf.params['ln_lightning'], r_rf.std_errors['ln_lightning'], r_rf.pvalues['ln_lightning'])}")
print(f"  注意：如果排他性约束成立，此系数应仅通过DEA渠道")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 第一阶段：Dea ~ lightning + Controls（检验相关性）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. 第一阶段：Dea ~ ln_lightning + Controls")
print("=" * 80)
r_fs = run_fe(s, "Dea", ["ln_lightning"] + CONTROLS)
print(f"  ln_lightning: {fmt(r_fs.params['ln_lightning'], r_fs.std_errors['ln_lightning'], r_fs.pvalues['ln_lightning'])}")

# 弱IV检验 — F统计量
# 由于固定效应模型没有直接的F统计量，我们计算排除工具变量的t统计量
t_stat = r_fs.params["ln_lightning"] / r_fs.std_errors["ln_lightning"]
f_stat = t_stat ** 2  # 单个IV：F ≈ t²
print(f"  第一阶段 t-stat = {t_stat:.2f}")
print(f"  Kleibergen-Paap F ≈ {f_stat:.2f}")
if f_stat > 10:
    print(f"  ✅ F > 10，排除弱IV问题（Stock & Yogo, 2005）")
elif f_stat > 5:
    print(f"  ⚠️ 5 < F < 10，边界弱IV")
else:
    print(f"  ❌ F < 5，严重弱IV问题！IV估计不可靠")

# 第一阶段符号检验
coef_fs = r_fs.params["ln_lightning"]
print(f"  第一阶段系数={coef_fs:.4f} → "
      + ("负向（预期方向：雷电↑→通讯中断→DEA↓）" if coef_fs < 0
         else "正向（与预期相反！雷电↑→DEA↑？" if coef_fs > 0
         else "零效应"))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 手动2SLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. 第二阶段：Res ~ Dea_hat + Controls（手动2SLS）")
print("=" * 80)

# 获取第一阶段拟合值（修复 MultiIndex 问题）
s_idx = s.set_index(["stkcode", "year"])
s_idx["Dea_hat"] = r_fs.fitted_values["fitted_values"]
s = s_idx.reset_index()

# 第二阶段回归
r_ss = run_fe(s, OUTCOME, ["Dea_hat"] + CONTROLS)
print(f"  Dea_hat (IV): {fmt(r_ss.params['Dea_hat'], r_ss.std_errors['Dea_hat'], r_ss.pvalues['Dea_hat'])}  "
      f"N={r_ss.nobs:,}")

# 与OLS对比
print(f"\n  OLS Dea系数:    {r_ols.params['Dea']:.5f}")
print(f"  2SLS Dea系数:   {r_ss.params['Dea_hat']:.5f}")
print(f"  2SLS/OLS比值:   {r_ss.params['Dea_hat']/r_ols.params['Dea']:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 排他性约束讨论
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. 排他性约束讨论")
print("=" * 80)

# 测试：雷电→供应链韧性的直接路径
# 控制DEA后，雷电应不显著
r_excl = run_fe(s, OUTCOME, ["Dea", "ln_lightning"] + CONTROLS)
coef_light_excl = r_excl.params["ln_lightning"]
p_light_excl = r_excl.pvalues["ln_lightning"]
print(f"  控制DEA后 ln_lightning → Res: {fmt(coef_light_excl, r_excl.std_errors['ln_lightning'], p_light_excl)}")
print(f"  → {'✅ 控制DEA后闪电不显著，排他性约束成立' if p_light_excl > 0.05 else '❌ 闪电仍有直接效应，排他性约束可能不满足'}")

print(f"""
排他性约束评估：
  论文逻辑：雷电 → 通信基础设施受损 → 企业DEA ↓（通过数字能力间接影响供应链）

  质疑要点：
  1. 雷电/极端天气 → 直接破坏物流和仓储 → 供应链韧性本身就受损
     （此路径不经过DEA，违反排他性约束）
  2. 高雷电地区企业可能更倾向于投资韧性基础设施（正向适应效应）
  3. 雷电频率与地理/气候因素相关，这些因素可能与经济发展水平相关

  检验结果：控制DEA后闪电频率{'不' if p_light_excl > 0.05 else ''}显著，
  排他性约束{'可能成立' if p_light_excl > 0.05 else '存在先验违反风险'}
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 安慰剂检验：用同一IV预测不应相关的变量
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("6. 安慰剂检验：ln_lightning → firm_age（不应相关）")
print("=" * 80)
r_plc = run_fe(s, "lnage", ["ln_lightning"] + [c for c in CONTROLS if c != "lnage"])
print(f"  ln_lightning → lnage: {fmt(r_plc.params['ln_lightning'], r_plc.std_errors['ln_lightning'], r_plc.pvalues['ln_lightning'])}")
print(f"  → {'⚠️ IV与安慰剂变量相关，可能有遗漏因素' if r_plc.pvalues['ln_lightning'] < 0.05 else '✅ IV与安慰剂变量不相关'}")

# ── 保存 ──────────────────────────────────────────────────────────────────────
results = {
    "OLS_coef": r_ols.params["Dea"], "OLS_se": r_ols.std_errors["Dea"],
    "OLS_p": r_ols.pvalues["Dea"], "OLS_N": r_ols.nobs,
    "FS_coef": coef_fs, "FS_se": r_fs.std_errors["ln_lightning"],
    "FS_p": r_fs.pvalues["ln_lightning"], "FS_F": f_stat,
    "RF_coef": r_rf.params["ln_lightning"], "RF_se": r_rf.std_errors["ln_lightning"],
    "RF_p": r_rf.pvalues["ln_lightning"],
    "2SLS_coef": r_ss.params["Dea_hat"], "2SLS_se": r_ss.std_errors["Dea_hat"],
    "2SLS_p": r_ss.pvalues["Dea_hat"], "2SLS_N": r_ss.nobs,
    "Excl_light_coef": coef_light_excl, "Excl_light_p": p_light_excl,
}
pd.DataFrame([results]).to_csv(OUT / "iv_regression_results.csv", index=False, encoding="utf-8-sig")
print(f"\n结果保存: {OUT}/iv_regression_results.csv")
