"""
将 DEA 数据、res_v2、控制变量与附加数据合并为完整 Stata .dta
"""
import pandas as pd
import pyreadstat
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
INTER = OUT / "中间结果"

# Ensure output directories exist
INTER.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

CTRL_PATH = DATA / "常用控制变量2000_2024_Ver3.1.dta"

# ============================================================
# 1. 加载 DEA 数据 + 控制变量
# ============================================================
print("1. 加载 DEA 数据 + 控制变量...")
dea, _ = pyreadstat.read_dta(DATA / "dea_original.dta")
dea["stkcode"] = dea["id_stock"].astype(int).astype(str).str.zfill(6)
dea["year"] = dea["year"].astype(int)
dea_vars = ["stkcode", "year", "Dea", "Breadth", "Depth",
            "Dea_count", "Breadth_count", "Depth_count",
            "_b_raw", "_d_raw", "_eff_chars"]
df = dea[[c for c in dea_vars if c in dea.columns]].copy()
print(f"   DEA: {len(df):,} obs, {df['stkcode'].nunique():,} firms")

# Merging controls
ctrl, _ = pyreadstat.read_dta(CTRL_PATH)
ctrl["stkcode"] = ctrl["stkcode"].astype(str).str.zfill(6)
ctrl["year"] = ctrl["year"].astype(int)
ctrl_cols = ["stkcode", "year", "lnage", "lnsize", "klr", "lev",
             "bsize", "dual", "lnrd", "indrate", "own",
             "ind", "ind_str", "ind1", "citycode", "city_x",
             "province_x", "soe", "area_1", "hhi_d", "hightech", "STPT"]
df = df.merge(ctrl[[c for c in ctrl_cols if c in ctrl.columns]],
              on=["stkcode", "year"], how="left")
print(f"   + 控制变量: {len(df):,} obs")

# ============================================================
# 2. 合并 res_v2
# ============================================================
print("2. 合并 res_v2...")
res_v2 = pd.read_csv(INTER / "res_v2_panel.csv")
res_v2["stkcode"] = res_v2["stkcode"].astype(str).str.zfill(6)
df = df.merge(res_v2[["stkcode", "year", "res_v2"]],
              on=["stkcode", "year"], how="inner")
df = df.rename(columns={"res_v2": "res"})
print(f"   res_v2: {df['res'].notna().sum():,} non-null")

# ============================================================
# 3. 合并 IC
# ============================================================
print("3. 合并 IC...")
ic, _ = pyreadstat.read_dta(DATA / "内部控制指数2000-2024.dta")
ic = ic.dropna(subset=["stkcd"]).copy()
ic["stkcode"] = ic["stkcd"].astype(int).astype(str).str.zfill(6)
ic["year"] = ic["year"].astype(int)
df = df.merge(ic[["stkcode", "year", "IC1"]].rename(columns={"IC1": "IC"}),
              on=["stkcode", "year"], how="left")
print(f"   IC: {df['IC'].notna().sum():,} non-null")

# ============================================================
# 4. 合并 Cost
# ============================================================
print("4. 合并 cost...")
cost, _ = pyreadstat.read_dta(DATA / "交易成本2000-2024.dta")
cost = cost.dropna(subset=["stkcd"]).copy()
cost["stkcode"] = cost["stkcd"].astype(int).astype(str).str.zfill(6)
cost["year"] = cost["year"].astype(int)
df = df.merge(cost[["stkcode", "year", "交易成本"]].rename(columns={"交易成本": "cost"}),
              on=["stkcode", "year"], how="left")
print(f"   cost: {df['cost'].notna().sum():,} non-null")

# ============================================================
# 5. 合并 PageRank (表4)
# ============================================================
print("5. 合并 PageRank...")
pr = pd.read_excel(DATA / "pagerank.xlsx")
pr["stkcode"] = pr["股票代码"].astype(int).astype(str).str.zfill(6)
pr["year"] = pr["年份"].astype(int)
df = df.merge(pr[["stkcode", "year", "PageRank_C", "PageRank_P"]],
              on=["stkcode", "year"], how="left")
print(f"   PageRank_C: {df['PageRank_C'].notna().sum():,} non-null")
print(f"   PageRank_P: {df['PageRank_P'].notna().sum():,} non-null")

# ============================================================
# 6. 合并供应链地理距离 (表5)
# ============================================================
print("6. 合并供应链地理距离...")
geo, _ = pyreadstat.read_dta(DATA / "上市公司供应链地理距离（2001-2024年）.dta")
geo["stkcode"] = geo["股票代码"].astype(str).str.zfill(6)
geo["year"] = geo["年份"].astype(int)
geo = geo.rename(columns={"客户地理距离": "Disw_c", "供应商地理距离": "Disw_s"})
df = df.merge(geo[["stkcode", "year", "Disw_c", "Disw_s"]],
              on=["stkcode", "year"], how="left")
print(f"   Disw_c: {df['Disw_c'].notna().sum():,} non-null")
print(f"   Disw_s: {df['Disw_s'].notna().sum():,} non-null")

# ============================================================
# 7. 合并数据要素市场化 (政策DID)
# ============================================================
print("7. 合并数据要素市场化 DID...")
dde, _ = pyreadstat.read_dta(DATA / "数据要素市场化配置2000-2024.dta")
dde["stkcode"] = dde["id_stock"].astype(int).astype(str).str.zfill(6)
dde["year"] = dde["year"].astype(int)
df = df.merge(dde[["stkcode", "year", "Treat", "Post", "DID"]],
              on=["stkcode", "year"], how="left")
print(f"   DID: {df['DID'].notna().sum():,} non-null")

# ============================================================
# 8. 合并雷电频率 (IV)
# ============================================================
print("8. 合并雷电频率 (IV)...")
lightning, _ = pyreadstat.read_dta(DATA / "lightning.dta")
lightning["citycode_num"] = pd.to_numeric(lightning["citycode"], errors="coerce")
df["citycode_num"] = pd.to_numeric(df["citycode"], errors="coerce")
df = df.merge(lightning[["citycode_num", "雷击频率"]].rename(columns={"雷击频率": "shandian"}),
              on="citycode_num", how="left")
df = df.drop(columns=["citycode_num"])
print(f"   shandian: {df['shandian'].notna().sum():,} non-null")

# ============================================================
# 9. 合并数据交易所 DID
# ============================================================
print("9. 合并数据交易所 DID...")
exchange, _ = pyreadstat.read_dta(DATA / "地级市数据交易所 did.dta")
exchange["year"] = exchange["year"].astype(int)
if "city_x" in df.columns:
    df = df.merge(exchange[["year", "city", "treat", "data_market"]].rename(
        columns={"treat": "exchange_treat", "data_market": "exchange_market"}),
        left_on=["year", "city_x"], right_on=["year", "city"], how="left")
    df = df.drop(columns=["city"])
    print(f"   exchange_treat: {df['exchange_treat'].notna().sum():,} non-null")

# ============================================================
# 10. 样本处理
# ============================================================
print("10. 样本筛选...")
if "STPT" in df.columns:
    df = df[df["STPT"] != 1].copy()
if "ind_str" in df.columns:
    excl = df["ind_str"].str.contains("金融|保险|银行|证券|货币|其他金融|建筑|房地产", na=False)
    df = df[~excl].copy()
print(f"   筛选后: {len(df):,} obs")

# ============================================================
# 11. 缩尾
# ============================================================
print("11. 缩尾处理...")
for col in ["lnrd", "lev", "klr"]:
    if col in df.columns:
        p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p1, upper=p99)

# ============================================================
# 12. 生成 Stata 编码变量
# ============================================================
print("12. 生成 Stata 编码变量...")

# ID
df["id"] = df["stkcode"].astype("category").cat.codes + 1

# City numeric
if "citycode" in df.columns:
    df["city_str"] = df["citycode"].astype(str)
elif "city_x" in df.columns:
    df["city_str"] = df["city_x"].astype(str)
else:
    df["city_str"] = "1"
df["city"] = df["city_str"].astype("category").cat.codes + 1

# Industry numeric
if "ind" in df.columns:
    df["ind_num"] = pd.to_numeric(df["ind"], errors="coerce")

# Count 变量
if "_b_raw" in df.columns and "_d_raw" in df.columns:
    if "Dea_count" not in df.columns:
        df["Dea_count"] = df["_b_raw"] + df["_d_raw"]
    if "Breadth_count" not in df.columns:
        df["Breadth_count"] = df["_b_raw"]
    if "Depth_count" not in df.columns:
        df["Depth_count"] = df["_d_raw"]

# PageRank 中心化交互项
for v in ["Dea", "Breadth", "Depth"]:
    for p in ["PageRank_C", "PageRank_P"]:
        pr_centered = f"{p}1"
        if p in df.columns:
            pr_mean = df[p].mean()
            df[pr_centered] = df[p] - pr_mean
            if v in df.columns:
                v_mean = df[v].mean()
                v_dm = df[v] - v_mean
                if p == "PageRank_C":
                    suffix = "PageRankC"
                else:
                    suffix = "PageRankP"
                df[f"{v}_{suffix}"] = v_dm * df[pr_centered]

# ============================================================
# 13. 输出
# ============================================================
print("13. 保存...")

output_vars = [
    "stkcode", "id", "year", "city",
    "res", "Dea", "Breadth", "Depth",
    "Dea_count", "Breadth_count", "Depth_count",
    "lnage", "lnsize", "klr", "lev", "bsize", "dual", "indrate", "own", "lnrd",
    "IC", "cost",
    "PageRank_C", "PageRank_P",
    "PageRank_C1", "PageRank_P1",
    "Dea_PageRankC", "Breadth_PageRankC", "Depth_PageRankC",
    "Dea_PageRankP", "Breadth_PageRankP", "Depth_PageRankP",
    "Disw_c", "Disw_s",
    "shandian",
    "DID", "Treat", "Post", "exchange_treat", "exchange_market",
    "ind_num", "hhi_d", "soe", "hightech", "area_1",
    "ind_str",
]

final = pd.DataFrame()
for v in output_vars:
    if v in df.columns:
        if v == "stkcode":
            final[v] = df[v].astype(str)
        else:
            final[v] = pd.to_numeric(df[v], errors="coerce")

if "ind_str" in df.columns and "ind_str" not in final.columns:
    final["ind_str"] = df["ind_str"].astype(str)

dta_path = OUT / "replication_panel_own.dta"
pyreadstat.write_dta(final, dta_path)
print(f"\n数据已保存: {dta_path}")
print(f"样本量: {len(final):,}")
print(f"变量数: {len(final.columns)}")
print(f"\n关键变量覆盖率:")
for v in ["Dea", "Breadth", "Depth", "IC", "cost", "shandian",
          "PageRank_C", "PageRank_P", "Disw_c", "Disw_s",
          "DID", "exchange_treat", "Dea_PageRankC", "Breadth_PageRankC"]:
    if v in final.columns:
        print(f"  {v}: {final[v].notna().sum():,} / {len(final):,}")
    else:
        print(f"  {v}: MISSING")
print(f"\n>>> 18_prepare_stata_data 完成 <<<")
