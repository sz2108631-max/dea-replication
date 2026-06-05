"""
关键词扩展脚本：用 Word2Vec 从 MD&A 语料库扩展种子词

流程：
  1. 读取所有年份中 12-31 结尾的 txt 文件（年报正文）
  2. jieba 分词 + 自定义种子词词典确保种子词不被切断
  3. 训练 Word2Vec Skip-Gram（dim=200，window=10）
  4. 对 keywords.json 中每个种子词，计算与词表中全部名词短语的余弦相似度
     保留 cos >= 0.75 的候选词（论文阈值）
  5. 扩展后写入 keywords_expanded.json（原文件不覆盖）
"""

import json, glob, os, re, logging
from pathlib import Path
from collections import defaultdict
import jieba
import jieba.posseg as pseg
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ─── 路径 ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
# ★ MD&A 文本目录 — 请修改为实际路径
CORPUS_DIR = Path(os.environ.get("CMDA_DIR", str(BASE / "cmda_texts")))
KWORDS_SRC = BASE / "config" / "keywords.json"
KWORDS_OUT = BASE / "config" / "keywords_expanded.json"
MODEL_PATH = BASE / "config" / "word2vec_mda.model"

# ─── 超参数 ────────────────────────────────────────────────────────────────
TOP_N    = 500      # 每个种子词取前 N 近邻（足够大以捕捉所有 >=0.75 的候选）
SIM_THR  = 0.75     # 余弦相似度阈值（论文原文：0.75）
W2V_DIM  = 200
W2V_WIN  = 10
W2V_MIN  = 5        # 最低词频
W2V_ITER = 10
W2V_WORKERS = 8
W2V_SG   = 1        # 1=Skip-Gram（论文指定），0=CBOW

# 名词相关 POS 标签（jieba posseg）
NOUN_FLAGS = {
    "n",    # 普通名词
    "nz",   # 其他专名
    "nr",   # 人名
    "ns",   # 地名
    "nt",   # 机构名
    "nw",   # 作品名
    "vn",   # 名动词
    "an",   # 名形词
    "eng",  # 英文（保留，如 GPU/AI 等）
}


def load_seed_words(kw_path: Path) -> dict:
    with open(kw_path) as f:
        return json.load(f)


def collect_all_seed_words(kw: dict) -> list[str]:
    words = []
    for key, dim in kw.items():
        if key.startswith("_") or not isinstance(dim, dict):
            continue
        for cat in dim.values():
            if not isinstance(cat, dict):
                continue
            words.extend(cat.get("core", []))
            words.extend(cat.get("auxiliary", []))
    return list(set(words))


def add_custom_words(seed_words: list[str]):
    """确保种子词在 jieba 中不被切断"""
    for w in seed_words:
        jieba.add_word(w, freq=100000)


def iter_corpus_sentences(corpus_dir: Path):
    """
    遍历所有年份/文本/XXXXXX_YYYY-12-31.txt
    按行 yield 分词列表（过滤长度<2的 token 和纯英文/数字 token）
    """
    pattern = str(corpus_dir / "*" / "文本" / "*_*-12-31.txt")
    files = sorted(glob.glob(pattern))
    log.info(f"共找到 {len(files)} 个年报文本文件")
    if not files:
        raise FileNotFoundError(f"未找到任何文件，请检查路径: {pattern}")

    # 过滤规则
    re_en_num = re.compile(r"^[a-zA-Z0-9\s]+$")
    re_punct  = re.compile(r"^[\W_]+$")

    for i, fp in enumerate(files):
        if i % 5000 == 0:
            log.info(f"  分词进度: {i}/{len(files)}")
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = [
                w for w in jieba.cut(line)
                if len(w) >= 2
                and not re_en_num.match(w)
                and not re_punct.match(w)
            ]
            if tokens:
                yield tokens


class CorpusSentences:
    """可多次迭代（gensim Word2Vec 需要多遍扫描）"""
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.pattern = str(corpus_dir / "*" / "文本" / "*_*-12-31.txt")
        self.files = sorted(glob.glob(self.pattern))
        self.re_en_num = re.compile(r"^[a-zA-Z0-9\s]+$")
        self.re_punct  = re.compile(r"^[\W_]+$")
        log.info(f"语料库文件数: {len(self.files)}")

    def __iter__(self):
        for fp in self.files:
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                tokens = [
                    w for w in jieba.cut(line)
                    if len(w) >= 2
                    and not self.re_en_num.match(w)
                    and not self.re_punct.match(w)
                ]
                if tokens:
                    yield tokens


def train_or_load_model(corpus_dir: Path, model_path: Path) -> Word2Vec:
    if model_path.exists():
        log.info(f"加载已有模型: {model_path}")
        return Word2Vec.load(str(model_path))

    log.info("开始训练 Word2Vec …")
    sentences = CorpusSentences(corpus_dir)
    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_DIM,
        window=W2V_WIN,
        min_count=W2V_MIN,
        epochs=W2V_ITER,
        workers=W2V_WORKERS,
        sg=W2V_SG,      # 1=Skip-Gram（论文指定）
    )
    model.save(str(model_path))
    log.info(f"模型已保存: {model_path}  词表大小: {len(model.wv)}")
    return model


def build_noun_vocab(model: Word2Vec) -> set:
    """
    对 Word2Vec 词表中所有词做 jieba POS 标注，
    保留名词类词（NOUN_FLAGS），返回名词词集合。
    """
    log.info(f"对词表 {len(model.wv)} 个词做 POS 标注，提取名词短语…")
    noun_vocab = set()
    for i, word in enumerate(model.wv.index_to_key):
        if i % 100000 == 0:
            log.info(f"  POS 标注进度: {i}/{len(model.wv)}")
        # jieba posseg 对单词标注
        for w, flag in pseg.cut(word):
            if flag in NOUN_FLAGS:
                noun_vocab.add(word)
                break
    log.info(f"名词词表大小: {len(noun_vocab)}")
    return noun_vocab


def expand_seed(model: Word2Vec, seed: str, top_n: int,
                threshold: float, noun_vocab: set) -> list[str]:
    """
    返回与 seed 余弦相似度 >= threshold 且属于名词词表的词
    """
    if seed not in model.wv:
        return []
    results = model.wv.most_similar(seed, topn=top_n)
    return [w for w, score in results
            if score >= threshold and w in noun_vocab]


def expand_keywords(kw: dict, model: Word2Vec, noun_vocab: set) -> dict:
    """
    对 kw 中每个 core/auxiliary 列表做扩展：
    - 原有种子词保留
    - 新扩展词追加（去重，不跨类别去重，保持结构不变）
    """
    import copy
    kw_exp = copy.deepcopy(kw)

    for dim_name, dim_cats in kw.items():
        if dim_name.startswith("_") or not isinstance(dim_cats, dict):
            continue
        for cat_name, cat_dict in dim_cats.items():
            if not isinstance(cat_dict, dict):
                continue
            for role in ("core", "auxiliary"):
                seeds = cat_dict.get(role, [])
                if not seeds:
                    continue
                all_new = set()
                for seed in seeds:
                    new_words = expand_seed(model, seed, TOP_N, SIM_THR, noun_vocab)
                    all_new.update(new_words)
                # 去掉已有种子词（在 core+auxiliary 中）
                existing = set(cat_dict.get("core", []) + cat_dict.get("auxiliary", []))
                added = [w for w in all_new if w not in existing]
                kw_exp[dim_name][cat_name][role] = seeds + added

                if added:
                    log.info(f"  [{dim_name}/{cat_name}/{role}] +{len(added)} 词: {added[:8]}…")

    return kw_exp


def report_expansion(kw_orig: dict, kw_exp: dict):
    print("\n===== 扩展统计 =====")
    total_orig = total_exp = 0
    for dim in kw_orig:
        if dim.startswith("_") or not isinstance(kw_orig[dim], dict):
            continue
        for cat in kw_orig[dim]:
            for role in ("core", "auxiliary"):
                o = len(kw_orig[dim][cat].get(role, []))
                e = len(kw_exp[dim][cat].get(role, []))
                total_orig += o
                total_exp  += e
                if e > o:
                    print(f"  {dim}/{cat}/{role}: {o} → {e} (+{e-o})")
    print(f"\n合计: {total_orig} → {total_exp} 词 (+{total_exp-total_orig})")


def main():
    kw_orig = load_seed_words(KWORDS_SRC)
    seed_words = collect_all_seed_words(kw_orig)

    log.info(f"种子词数: {len(seed_words)}")
    log.info("为 jieba 添加自定义词 …")
    add_custom_words(seed_words)

    model = train_or_load_model(CORPUS_DIR, MODEL_PATH)

    # 检查种子词覆盖率
    missing = [w for w in seed_words if w not in model.wv]
    if missing:
        log.warning(f"以下种子词不在词表中（词频过低或未出现）: {missing}")

    # 构建名词词表（论文方法：仅保留名词短语候选）
    noun_vocab = build_noun_vocab(model)

    log.info(f"扩展关键词（阈值={SIM_THR}，名词过滤）…")
    kw_expanded = expand_keywords(kw_orig, model, noun_vocab)

    report_expansion(kw_orig, kw_expanded)

    with open(KWORDS_OUT, "w", encoding="utf-8") as f:
        json.dump(kw_expanded, f, ensure_ascii=False, indent=2)
    log.info(f"扩展词典已写入: {KWORDS_OUT}")


if __name__ == "__main__":
    main()
