"""
Module C：场景约束规则 + 辅助性关键词特殊处理
论文原文：
  "仅当某一语句中出现至少一个指向具体数据基础设施、数据处理流程、
   数据存储形态或业务应用场景的场景类或技术类术语，且整体语境被判定
   为实际部署/应用场景时，才统计该语句中关键词的频次；
   若'数字化''智能''AI''数据驱动''技术赋能''互联网技术''信息技术'
   '数据管理''信息管理'等语义范畴较宽泛的辅助性关键词仅以笼统口号
   或原则性表述出现，而缺乏对具体设施、技术对象或业务场景的明确指向，
   则该关键词不计入企业层面的广度或深度应用能力指标。"

论文稳健性指标：关键词词频 / MD&A总字数（剔除停用词）×1000
"""

import re
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 停用词表（哈工大停用词表 + 补充）
# 用途：① 稳健性指标分母（有效字数）  ② 分句时表格行识别（实词密度）
# ─────────────────────────────────────────────────────────────────────────────

def _load_stopwords() -> frozenset:
    """优先加载哈工大停用词表文件，不存在则使用内置备用词表"""
    hit_path = Path(__file__).parent.parent / "config" / "hit_stopwords.txt"
    words = set()
    if hit_path.exists():
        with open(hit_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)
    # 补充内置关键停用字（单字，用于逐字过滤）
    extra = (
        "的地得着了过把被也就都而且然而虽然即使如果因为由于"
        "因此于是从而或者不但不仅以及与和及对于关于至于"
        "我你他她它们我们你们他们该此这那本各自每"
        "在是以为从到向于由将被啊呀哦嗯吧呢吗么"
        "一二三四五六七八九十百千万亿"
        "，。！？；：""''（）【】《》〈〉、…—"
    )
    words.update(extra)
    return frozenset(words)

_STOPWORDS: frozenset = _load_stopwords()


def count_effective_chars(text: str) -> int:
    """
    计算文本有效字符数（剔除停用词、数字、标点、空白）
    用于稳健性指标分母：Dea_count = 词频 / 有效字数 × 1000
    """
    text = re.sub(r'\s+', '', text)
    count = sum(
        1 for ch in text
        if ch not in _STOPWORDS       # 非停用词
        and not ch.isascii()          # 非ASCII（去英文/数字/标点）
        and not ch.isdigit()          # 非数字
        and '\u4e00' <= ch <= '\u9fff' # 必须是汉字
    )
    return count if count > 0 else max(len(text), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 关键词词典加载
# ─────────────────────────────────────────────────────────────────────────────

def load_keyword_sets(config_path: str = "config/keywords_expanded.json"):
    """加载关键词配置，返回各类词集合"""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    breadth_core, breadth_aux = set(), set()
    depth_core, depth_aux     = set(), set()

    for cat_data in cfg["breadth"].values():
        breadth_core.update(cat_data["core"])
        breadth_aux.update(cat_data["auxiliary"])

    for cat_data in cfg["depth"].values():
        depth_core.update(cat_data["core"])
        depth_aux.update(cat_data["auxiliary"])

    # 按长度降序排列（长词优先匹配，避免短词截断长词）
    all_breadth = sorted(breadth_core | breadth_aux, key=len, reverse=True)
    all_depth   = sorted(depth_core   | depth_aux,   key=len, reverse=True)

    return {
        "breadth_core": breadth_core,
        "breadth_aux":  breadth_aux,
        "depth_core":   depth_core,
        "depth_aux":    depth_aux,
        "all_breadth":  all_breadth,
        "all_depth":    all_depth,
        "all_aux":      breadth_aux | depth_aux,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 口号/笼统表述识别（辅助词专属规则）
# 论文原文：以"持续推进数字化转型""不断提升智能化水平"等笼统口号出现
# ─────────────────────────────────────────────────────────────────────────────

_SLOGAN_PATTERNS = re.compile(
    r'(?:'
    # 模式1: 持续/积极/大力 + 推进/提升/加强 + 数字化/智能化
    r'(?:持续|不断|积极|大力|深入|全面|加快|着力|加强|努力|深化)'
    r'.{0,8}'
    r'(?:推进|提升|加强|深化|推动|完善|建设|发展|实现|打造)'
    r'.{0,12}'
    r'(?:数字化|智能化|信息化|数字化转型|数字转型|智能转型)'
    r'|'
    # 模式2: 数字化/AI/智能 + 战略/方向/目标/愿景/规划
    r'(?:数字化|智能化|AI|人工智能|数据驱动).{0,15}'
    r'(?:战略|方向|目标|愿景|规划|路径|蓝图|布局)'
    r'|'
    # 模式3: 提升/提高/加强 + (数字化/智能化/信息化/数据管理) + 能力/水平/效率
    r'(?:提升|提高|加强|增强).{0,8}'
    r'(?:数字化|智能化|信息化|数据管理|信息管理).{0,8}'
    r'(?:能力|水平|效率|质量|程度|水准)'
    r'|'
    # 模式4: 技术赋能/数字赋能/互联网技术+赋能/助力/支撑（笼统赋能表达）
    r'(?:技术赋能|数字赋能|互联网技术|信息技术).{0,10}'
    r'(?:赋能|助力|支撑|支持|引领|驱动|推动)'
    r')',
    re.S
)


def is_slogan(sentence: str, matched_aux: set) -> bool:
    """
    判断句子是否为口号式/笼统原则性表述
    仅在matched_aux非空、core词为空时调用
    """
    return bool(_SLOGAN_PATTERNS.search(sentence))


# ─────────────────────────────────────────────────────────────────────────────
# 核心：场景约束 → 确定句子中有效的关键词计数
# ─────────────────────────────────────────────────────────────────────────────

def apply_scene_constraint(sentence: str, kw_sets: dict) -> tuple[int, int]:
    """
    对单个句子应用场景约束规则，返回 (breadth_count, depth_count)

    规则（严格按论文逻辑）：
    ① 找出句子中匹配的所有广度词/深度词（区分core/aux）
    ② 如果有core词：所有匹配词（含aux）正常计入
    ③ 如果只有辅助词：
         → 句子是口号式笼统表述 → 全部不计入（count=0）
         → 句子有具体操作描述 → 辅助词正常计入
    """
    # 广度词匹配
    matched_b_core = [kw for kw in kw_sets["breadth_core"] if kw in sentence]
    matched_b_aux  = [kw for kw in kw_sets["breadth_aux"]  if kw in sentence]
    # 深度词匹配
    matched_d_core = [kw for kw in kw_sets["depth_core"] if kw in sentence]
    matched_d_aux  = [kw for kw in kw_sets["depth_aux"]  if kw in sentence]

    # ── 广度词计数 ──
    if matched_b_core:
        # 有具体技术词：core + aux 全部计入
        b_count = len(matched_b_core) + len(matched_b_aux)
    elif matched_b_aux:
        # 仅有辅助词：检查是否口号
        b_count = 0 if is_slogan(sentence, set(matched_b_aux)) else len(matched_b_aux)
    else:
        b_count = 0

    # ── 深度词计数 ──
    if matched_d_core:
        d_count = len(matched_d_core) + len(matched_d_aux)
    elif matched_d_aux:
        d_count = 0 if is_slogan(sentence, set(matched_d_aux)) else len(matched_d_aux)
    else:
        d_count = 0

    return b_count, d_count


# ─────────────────────────────────────────────────────────────────────────────
# 文本预处理
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    中文分句，用于 MD&A 文本（含财务表格的董事会报告全文）。

    处理逻辑：
      Step 1：按句末强标点（。！？；）和换行切为初步句子
      Step 2：利用哈工大停用词表计算实词密度，过滤财务表格行
              （实词密度低 or 含明确表格标志词 → 丢弃）
      Step 3：对保留句子，若超过 MAX_SENT 字，在逗号处二次切分
      Step 4：最终超限的片段硬截断（BERT 512 token 上限保护）

    BERT 上限：512 token，中文≈1字/token，hypothesis≈20token
    安全上限：MAX_SENT = 480 字
    """
    MAX_SENT       = 480
    MIN_LEN        = 8
    MIN_WORD_RATIO = 0.28   # 实词（去停用词后汉字）占比低于此值 → 表格/噪声行

    # ── 实词密度（基于哈工大停用词表）────────────────────────────────────
    def _word_ratio(s: str) -> float:
        """有效实词汉字 / 总字符数"""
        if not s:
            return 0.0
        effective = sum(
            1 for ch in s
            if '\u4e00' <= ch <= '\u9fff' and ch not in _STOPWORDS
        )
        return effective / len(s)

    # 明确表格标志词 → 无论密度直接丢弃
    _TABLE_RE = re.compile(
        r'单位[：:](?:元|万元|千元|人民币|美元|港元)|'
        r'(?:√适用|□适用|□不适用|√不适用)|'
        r'前\s*[三五]\s*(?:名|大)\s*(?:客户|供应商|股东)|'
        r'(?:营业收入|营业成本|毛利率).{0,5}(?:营业收入|营业成本|毛利率)|'
        r'合计\s*[\d,]{5,}'
    )

    def _is_table_sent(s: str) -> bool:
        if _TABLE_RE.search(s):
            return True
        if _word_ratio(s) < MIN_WORD_RATIO:
            return True
        return False

    # ── 逗号二次切分 ──────────────────────────────────────────────────────
    def _split_by_comma(seg: str) -> list[str]:
        parts = re.split(r'[，,]', seg)
        result, buf = [], ''
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(buf) + len(p) + 1 <= MAX_SENT:
                buf = (buf + '，' + p) if buf else p
            else:
                if buf:
                    result.append(buf)
                buf = p
        if buf:
            result.append(buf)
        return result

    # ── 主流程 ────────────────────────────────────────────────────────────
    raw_sents = re.split(r'[。！？；\n]+', text)
    sentences = []

    for seg in raw_sents:
        seg = seg.strip()
        if len(seg) < MIN_LEN:
            continue
        if _is_table_sent(seg):
            continue

        if len(seg) <= MAX_SENT:
            sentences.append(seg)
        else:
            for sub in _split_by_comma(seg):
                sub = sub.strip()
                if len(sub) < MIN_LEN or _is_table_sent(sub):
                    continue
                sentences.append(sub[:MAX_SENT])

    return sentences


def quick_has_keyword(sentence: str, kw_sets: dict) -> bool:
    """
    快速预判：句子是否包含任何关键词（跳过无关句子，降低BERT调用量）
    """
    for kw in kw_sets["all_breadth"]:
        if kw in sentence:
            return True
    for kw in kw_sets["all_depth"]:
        if kw in sentence:
            return True
    return False
