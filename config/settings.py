"""
全局设定 — 复现代码共用参数
"""

# 核心模型设定
OUTCOME = "res"
CONTROLS = ["lnage", "dual", "lev"]

# 固定效应设定
ENTITY_FE = "stkcode"
TIME_FE = "year"
CLUSTER_VAR = "city"

# 数据路径
PANEL_PATH = "output/replication_panel_own.dta"
