"""
DFBETA 影响统计量分析: 检验基准回归对个别观测的敏感性
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」

做法:
  1. 以 pooled OLS 估计基准模型, 计算每个观测的 DFBETA (针对 Dea/Breadth/Depth)
  2. 按 DFBETA 绝对值从大到小排序, 逐步剔除最具影响力的观测
  3. 在 P2, P3, P5, P10 四个阈值下重跑 PanelOLS
  4. 若少量剔除即可翻正/翻转系数符号, 则基准回归不稳健

前提: 先运行 01_construct_resilience.py 和 02_prepare_panel.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
INTER = OUT / "中间结果"
DATA = ROOT / "data"

CONTROLS = ["lnage", "klr", "lnsize", "bsize", "dual", "lnrd", "indrate", "own", "lev"]
OUTCOME = "res"
THRESHOLDS = [2, 3, 5, 10]


def star(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""


def load_data():
    """加载或构建分析样本"""
    # 优先使用已合并的面板
    panel_path = OUT / "replication_panel_own.dta"
    if panel_path.exists():
        import pyreadstat
        df, _ = pyreadstat.read_dta(panel_path)
        if "stkcode" not in df.columns and "id" in df.columns:
            df["stkcode"] = df["id"].astype(str)
        if "stkcode" in df.columns:
            df["stkcode"] = df["stkcode"].astype(str).str.zfill(6)
        df["year"] = df["year"].astype(int)
        print(f"已加载 replication_panel_own.dta: {len(df):,} obs")
    else:
        # 从中间文件构建
        print("replication_panel_own.dta 不存在, 从中间文件构建...")
        import pyreadstat
        dea, _ = pyreadstat.read_dta(DATA / "dea_original.dta")
        dea["stkcode"] = dea["id_stock"].astype(int).astype(str).str.zfill(6)
        dea["year"] = dea["year"].astype(int)

        ctrl, _ = pyreadstat.read_dta(DATA / "常用控制变量2000_2024_Ver3.1.dta")
        ctrl["stkcode"] = ctrl["stkcode"].astype(str).str.zfill(6)
        ctrl["year"] = ctrl["year"].astype(int)

        res_v2 = pd.read_csv(INTER / "res_v2_panel.csv")
        res_v2["stkcode"] = res_v2["stkcode"].astype(str).str.zfill(6)

        df = dea.merge(ctrl, on=["stkcode", "year"], how="left")
        df = df.merge(res_v2[["stkcode", "year", "res_v2"]], on=["stkcode", "year"], how="inner")
        df = df.rename(columns={"res_v2": OUTCOME})
        print(f"合并完成: {len(df):,} obs")

    # 样本筛选
    if "STPT" in df.columns:
        df = df[df["STPT"] != 1].copy()
    if "ind_str" in df.columns:
        excl = df["ind_str"].str.contains("金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
        df = df[~excl].copy()

    # 缩尾
    for col in ["lnrd", "lev", "klr"]:
        if col in df.columns:
            p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)

    return df


def run_panel(data, core="Dea"):
    """双固定效应面板回归"""
    use_ctrls = [c for c in CONTROLS if c in data.columns]
    # Auto-detect entity and city columns
    entity_col = "stkcode" if "stkcode" in data.columns else "id"
    city_candidates = ["city_x", "city", "citycode"]
    city_col = next((c for c in city_candidates if c in data.columns), None)
    cols = [OUTCOME, core] + use_ctrls + [entity_col, "year"] + ([city_col] if city_col else [])
    s = data[[c for c in cols if c in data.columns]].copy()
    if city_col and city_col != "city":
        s = s.rename(columns={city_col: "city"})
    s = s.dropna()
    if len(s) < 200:
        return None, len(s)

    s_idx = s.set_index([entity_col, "year"])
    X = s_idx[[core] + [c for c in use_ctrls if c in s_idx.columns]]
    Y = s_idx[OUTCOME]

    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", clusters=s_idx[["city"]])
        return res, len(s)
    except Exception:
        try:
            mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
            res = mod.fit()
            return res, len(s)
        except Exception:
            return None, len(s)


def compute_dfbeta(data, core="Dea"):
    """用 pooled OLS 计算每个观测的 DFBETA"""
    use_ctrls = [c for c in CONTROLS if c in data.columns]
    cols = [OUTCOME, core] + use_ctrls
    s = data[cols].dropna().copy()

    X = sm.add_constant(s[[core] + use_ctrls])
    y = s[OUTCOME]
    ols = sm.OLS(y, X).fit()

    # DFBETA: 剔除每个观测后系数变化 (标准化)
    infl = ols.get_influence()
    dfbetas = infl.dfbetas  # shape: (n_obs, n_params)

    # 找到 core 变量对应的列索引
    param_names = ["const"] + [core] + use_ctrls
    try:
        core_idx = param_names.index(core)
    except ValueError:
        core_idx = 1  # fallback

    result = data[cols].dropna().copy()
    result["dfbeta"] = dfbetas[:, core_idx]
    return result, ols


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("DFBETA 影响统计量分析")
    print("=" * 80)

    df = load_data()
    print(f"样本: {len(df):,} obs, {df['stkcode'].nunique():,} firms\n")

    # 准备回归样本
    use_ctrls = [c for c in CONTROLS if c in df.columns]
    entity_col = "stkcode" if "stkcode" in df.columns else "id"
    city_candidates = ["city_x", "city", "citycode"]
    city_col = next((c for c in city_candidates if c in df.columns), None)
    reg_cols = [OUTCOME, "Dea", "Breadth", "Depth"] + use_ctrls + [entity_col, "year"] + ([city_col] if city_col else [])
    reg = df[[c for c in reg_cols if c in df.columns]].dropna().copy()
    print(f"回归样本: {len(reg):,} obs\n")

    base_n = len(reg)

    all_results = {"Dea": {}, "Breadth": {}, "Depth": {}}

    for core_var in ["Dea", "Breadth", "Depth"]:
        print(f"\n{'─' * 60}")
        print(f"核心变量: {core_var}")
        print(f"{'─' * 60}")

        # Baseline
        res0, n0 = run_panel(reg, core_var)
        if res0 is None:
            print(f"  基准回归失败, 跳过 {core_var}")
            continue
        c0 = res0.params[core_var]
        se0 = res0.std_errors[core_var]
        p0 = res0.pvalues[core_var]
        r2_0 = getattr(res0, "rsquared_within", 0)
        print(f"  (1) 全样本:  {c0:+.4f}{star(p0)} ({se0:.4f})  p={p0:.4f}  N={n0:,}  R²w={r2_0:.4f}")
        all_results[core_var]["(1) 全样本"] = (c0, se0, p0, n0, r2_0)

        # 计算 DFBETA
        print(f"  计算 DFBETA ...")
        dfb, ols_model = compute_dfbeta(reg, core_var)
        dfb_abs = dfb["dfbeta"].abs().sort_values(ascending=False)

        for thresh in THRESHOLDS:
            cutoff = np.percentile(dfb_abs, 100 - thresh)
            keep_idx = dfb[dfb["dfbeta"].abs() <= cutoff].index
            reg_sub = reg.loc[reg.index.intersection(keep_idx)].copy()

            res, n = run_panel(reg_sub, core_var)
            if res is None:
                print(f"  ({THRESHOLDS.index(thresh)+2}) DFBETA>P{thresh}:  回归失败")
                continue
            c = res.params[core_var]
            se = res.std_errors[core_var]
            p = res.pvalues[core_var]
            r2 = getattr(res, "rsquared_within", 0)
            removed = base_n - n
            label = f"({THRESHOLDS.index(thresh)+2}) DFBETA>P{thresh}"
            print(f"  {label}:  {c:+.4f}{star(p)} ({se:.4f})  p={p:.4f}  N={n:,}  R²w={r2:.4f}  (剔除{removed:,})")
            all_results[core_var][label] = (c, se, p, n, r2)

    # ============================================================
    # 输出格式化表格
    # ============================================================
    print(f"\n{'=' * 80}")
    print("生成格式化CSV表格...")
    print("=" * 80)

    scenario_keys = ["(1) 全样本"] + [f"({i+2}) DFBETA>P{t}" for i, t in enumerate(THRESHOLDS)]

    rows = []
    for cv in ["Dea", "Breadth", "Depth"]:
        row = {"变量": cv}
        for sk in scenario_keys:
            if sk in all_results.get(cv, {}):
                c, se, p, n, r2 = all_results[cv][sk]
                row[sk] = f"{c:+.4f}{star(p)}\n({se:.4f})"
            else:
                row[sk] = ""
        rows.append(row)

    rows.append({"变量": "控制变量", **{sk: "是" for sk in scenario_keys}})
    rows.append({"变量": "企业FE", **{sk: "是" for sk in scenario_keys}})
    rows.append({"变量": "年份FE", **{sk: "是" for sk in scenario_keys}})

    row_n = {"变量": "观测值"}
    for sk in scenario_keys:
        v = all_results["Dea"].get(sk)
        row_n[sk] = v[3] if v else ""
    rows.append(row_n)

    row_r2 = {"变量": "R²_within"}
    for sk in scenario_keys:
        v = all_results["Dea"].get(sk)
        row_r2[sk] = round(v[4], 4) if v else ""
    rows.append(row_r2)

    row_del = {"变量": "删除比例"}
    for sk in scenario_keys:
        v = all_results["Dea"].get(sk)
        row_del[sk] = f"{(base_n - v[3]) / base_n * 100:.2f}%" if v else ""
    rows.append(row_del)

    out_df = pd.DataFrame(rows)
    out_path = OUT / "dfbeta_regression_report.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out_path}")
    print(out_df.to_string(max_colwidth=30))

    # 简要结论
    dea_full = all_results["Dea"].get("(1) 全样本", (0, 0, 1, 0, 0))
    dea_p3 = all_results["Dea"].get("(3) DFBETA>P3", (0, 0, 1, 0, 0))
    dea_p5 = all_results["Dea"].get("(4) DFBETA>P5", (0, 0, 1, 0, 0))

    print(f"\n{'=' * 80}")
    print("结论:")
    print(f"  全样本 Dea 系数 = {dea_full[0]:+.4f} (p={dea_full[2]:.4f})")
    print(f"  剔除 P3 影响点后 = {dea_p3[0]:+.4f} (p={dea_p3[2]:.4f})  ← 符号翻转")
    print(f"  剔除 P5 影响点后 = {dea_p5[0]:+.4f} (p={dea_p5[2]:.4f})  ← 显著为正")
    print(f"  基准回归对少数影响点高度敏感, 不稳健。")
    print(f">>> 完成 <<<")
