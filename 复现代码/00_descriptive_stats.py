"""
描述性统计 — 原文 vs 复现对比
巫强 et al. (2026)「企业数据要素应用能力与供应链韧性」
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── 读取数据 ──
df, _ = pyreadstat.read_dta(OUT / "replication_panel_own.dta")
df["id_stock"] = pd.to_numeric(df["stkcode"], errors="coerce")
df["year"] = df["year"].astype(int)

# 筛选2011-2023（与原文同期）
d = df[df["year"].between(2011, 2023)].copy()

VARS = ["res", "Dea", "Breadth", "Depth",
        "lnage", "lnsize", "klr", "lev", "bsize", "dual", "lnrd", "indrate", "own"]

print("=" * 80)
print("描述性统计 — 原文 vs 复现")
print("=" * 80)
print(f"\n样本: {len(d):,} obs, {d['id_stock'].nunique():,} firms (2011-2023)")
print(f"\n{'变量':<14} {'N':>8} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
print("-" * 65)

stats_rows = []
for v in VARS:
    if v in d.columns:
        s = d[v].dropna()
        stats_rows.append({
            "变量": v, "N": len(s), "均值": s.mean(), "标准差": s.std(),
            "最小值": s.min(), "最大值": s.max()
        })
        print(f"{v:<14} {len(s):>8} {s.mean():>10.4f} {s.std():>10.4f} {s.min():>10.4f} {s.max():>10.4f}")

# 保存
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT / "descriptive_stats.csv", index=False, encoding="utf-8-sig")
print(f"\n已保存: {OUT}/descriptive_stats.csv")
