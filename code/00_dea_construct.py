"""
主处理流程：逐文件处理 MD&A 文本，生成 Dea/Breadth/Depth 面板数据

论文完整流程（1:1复现）：
  ① 分句
  ② 关键词预筛（快速过滤无关句）
  ③ [Module B] 正则预筛 → BERT二分类（阈值0.60）
  ④ [Module C] 场景约束 + 辅助词特殊处理
  ⑤ 关键词词频汇总
  ⑥ ln(count + 1) 量化

使用方式：
  python code/00_dea_construct.py                   # 使用零样本BERT（默认）
  python code/00_dea_construct.py --no-bert         # 仅规则（验证基线，对应论文Dea_count）
  python code/00_dea_construct.py --model fine-tuned # 使用fine-tuned模型（需先训练）
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional

import pandas as pd
from tqdm import tqdm

# 将当前目录加入路径
sys.path.insert(0, str(Path(__file__).parent))
from dea_module_bert import regex_prefilter, BertDeploymentClassifier
from dea_module_rules import (
    load_keyword_sets, split_sentences,
    apply_scene_constraint, quick_has_keyword,
    count_effective_chars,
)

# ─────────────────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
# ★ MD&A 文本目录 — 请修改为实际路径
#    目录结构应为: <BASE_DIR>/<年份>/文本/<stkcode>_<年份>-12-31.txt
BASE_DIR    = Path(os.environ.get("CMDA_DIR", str(ROOT / "cmda_texts")))
CONFIG_PATH = ROOT / "config" / "keywords_expanded.json"
OUTPUT_DIR  = ROOT / "output"
YEARS       = range(2011, 2024)   # 论文样本：2011-2023，仅年报(12-31)
BATCH_SIZE  = 64                   # BERT批量推理大小

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 文件处理（纯规则版，用于多进程）
# ─────────────────────────────────────────────────────────────────────────────

_KW_SETS = None  # 进程内全局，避免重复加载

def _init_worker(config_path):
    global _KW_SETS
    _KW_SETS = load_keyword_sets(config_path)

def process_file_rules_only(fpath: str) -> Optional[dict]:
    """
    纯规则版（无BERT）：用于Stage 1基线，对应论文 Dea_count
    多进程安全
    """
    try:
        fname = Path(fpath).name
        stkcode = fname.split("_")[0]
        year    = int(fname.split("_")[1].split("-")[0])

        with open(fpath, encoding="utf-8", errors="ignore") as f:
            text = f.read()

        sentences = split_sentences(text)

        b_total = d_total = 0
        for sent in sentences:
            if not quick_has_keyword(sent, _KW_SETS):
                continue
            # 仅Module C（无BERT）
            b, d = apply_scene_constraint(sent, _KW_SETS)
            b_total += b
            d_total += d

        return _make_record(stkcode, year, b_total, d_total, text)
    except Exception as e:
        log.warning(f"处理失败 {fpath}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 文件处理（含BERT版，单进程，GPU批量推理）
# ─────────────────────────────────────────────────────────────────────────────

def process_file_with_bert(fpath: str, kw_sets: dict,
                            bert_clf) -> Optional[dict]:
    """
    完整版（含BERT Module B + Module C）
    单进程调用，BERT在外部统一初始化
    """
    try:
        fname = Path(fpath).name
        stkcode = fname.split("_")[0]
        year    = int(fname.split("_")[1].split("-")[0])

        with open(fpath, encoding="utf-8", errors="ignore") as f:
            text = f.read()

        sentences = split_sentences(text)

        # ── Step 1: 关键词预筛（快速过滤无关句）──────────────────────────
        candidate_sents = [s for s in sentences
                           if quick_has_keyword(s, kw_sets)]

        if not candidate_sents:
            return _make_record(stkcode, year, 0, 0, text)

        # ── Step 2: 正则预筛（Module B 第一层，无需BERT）─────────────────
        need_bert, pre_accept, pre_reject = [], [], []
        for sent in candidate_sents:
            decision = regex_prefilter(sent)
            if decision == 'reject':
                pre_reject.append(sent)
            elif decision == 'accept':
                pre_accept.append(sent)
            else:  # 'bert'
                need_bert.append(sent)

        # ── Step 3: BERT推理（Module B 第二层，批量）────────────────────
        bert_decisions = bert_clf.predict_batch(need_bert)
        bert_accepted = [s for s, ok in zip(need_bert, bert_decisions) if ok]

        # 通过Module B的句子
        valid_sents = pre_accept + bert_accepted

        # ── Step 4: 场景约束（Module C）──────────────────────────────────
        b_total = d_total = 0
        for sent in valid_sents:
            b, d = apply_scene_constraint(sent, kw_sets)
            b_total += b
            d_total += d

        return _make_record(stkcode, year, b_total, d_total, text)

    except Exception as e:
        log.warning(f"处理失败 {fpath}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(stkcode, year, b_raw, d_raw, text_full):
    """
    生成单条记录
    主指标分母：无（直接取raw count取对数）
    稳健性指标分母：MD&A有效字数（剔除停用词，对应论文描述）
    """
    total     = b_raw + d_raw
    eff_chars = count_effective_chars(text_full)   # 剔除停用词后字数

    return {
        "stkcode":        stkcode,
        "year":           year,
        # 主指标：ln(count+1)
        "Breadth":        math.log(b_raw + 1),
        "Depth":          math.log(d_raw + 1),
        "Dea":            math.log(total + 1),
        # 稳健性指标：词频 / MD&A有效字数（剔除停用词）×1000
        "Breadth_count":  b_raw / eff_chars * 1000 if eff_chars else 0,
        "Depth_count":    d_raw / eff_chars * 1000 if eff_chars else 0,
        "Dea_count":      total / eff_chars * 1000 if eff_chars else 0,
        # 原始计数（调试/验证用）
        "_b_raw":         b_raw,
        "_d_raw":         d_raw,
        "_eff_chars":     eff_chars,
    }


def collect_annual_files() -> list[str]:
    """收集2011-2023年所有年报(12-31)文件路径"""
    files = []
    for year in YEARS:
        txt_dir = BASE_DIR / str(year) / "文本"
        if not txt_dir.exists():
            log.warning(f"目录不存在: {txt_dir}")
            continue
        annual = [str(txt_dir / f)
                  for f in os.listdir(txt_dir)
                  if f.endswith(".txt") and "12-31" in f]
        log.info(f"{year}: {len(annual)} 个年报文件")
        files.extend(annual)
    log.info(f"总计: {len(files)} 个文件")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dea指标构建主流程")
    parser.add_argument("--no-bert", action="store_true",
                        help="仅使用规则（基线，对应论文Dea_count稳健性检验）")
    parser.add_argument("--model", choices=["zero-shot","fine-tuned"],
                        default="zero-shot",
                        help="BERT模型类型（默认zero-shot）")
    parser.add_argument("--model-path", default="models/bert_negation",
                        help="fine-tuned模型路径（--model fine-tuned时使用）")
    parser.add_argument("--workers", type=int, default=4,
                        help="多进程数（仅--no-bert模式使用）")
    parser.add_argument("--output", default=None,
                        help="输出文件名（默认自动命名）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 输出文件名
    if args.output:
        out_path = OUTPUT_DIR / args.output
    elif args.no_bert:
        out_path = OUTPUT_DIR / "dea_rules_only.csv"
    else:
        out_path = OUTPUT_DIR / f"dea_bert_{args.model}.csv"

    files = collect_annual_files()

    # ── 模式A：纯规则（多进程） ──────────────────────────────────────────
    if args.no_bert:
        log.info(f"模式：纯规则（{args.workers}进程并行）")
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(str(CONFIG_PATH),)
        ) as pool:
            results = list(tqdm(
                pool.imap(process_file_rules_only, files, chunksize=50),
                total=len(files), desc="处理文件"
            ))

    # ── 模式B：BERT完整版（单进程，GPU批量推理） ──────────────────────────
    else:
        log.info(f"模式：BERT {args.model}")
        kw_sets = load_keyword_sets(str(CONFIG_PATH))

        # 加载BERT分类器
        if args.model == "fine-tuned":
            from module_b_bert import BertFineTunedClassifier
            bert_clf = BertFineTunedClassifier(model_path=args.model_path)
            log.info(f"已加载fine-tuned模型: {args.model_path}")
        else:
            bert_clf = BertDeploymentClassifier(device="auto")

        results = []
        for fpath in tqdm(files, desc="处理文件（BERT）"):
            rec = process_file_with_bert(fpath, kw_sets, bert_clf)
            results.append(rec)

    # ── 汇总输出 ─────────────────────────────────────────────────────────
    records = [r for r in results if r is not None]
    df = pd.DataFrame(records)
    df = df.sort_values(["stkcode", "year"]).reset_index(drop=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    log.info(f"\n✅ 完成！共 {len(df)} 条记录 → {out_path}")
    log.info("\n=== 描述统计 ===")
    log.info(df[["Dea","Breadth","Depth","Dea_count"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
