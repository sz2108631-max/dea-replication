"""
Module B：BERT二分类否定过滤器
论文原文：利用正则化模板识别包含否定词的短语结构，将其标记为潜在否定语境；
         使用基于BERT得到的二分类模型对相关句段是否反映'实际部署/应用'
         关系来判别，并以0.60作为分类概率阈值，仅将预测概率不低于该阈值
         的陈述保留为后续计量的有效语料。

实现策略（两阶段）：
  Step 1：正则预筛
            - 明确否定句（尚未/并未/不涉及/…+部署/应用）→ 直接 reject
            - 明确未来计划（计划/拟/将+建立/引入/…）→ 直接 reject
            - 处于探索/起步阶段 → 直接 reject
            - 含模糊否定词（未/不/没/无）→ 交 BERT 精判
            - 其余 → 直接 accept
  Step 2：BERT零样本NLI分类（阈值0.60，本地模型）
"""

import re
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：正则预筛
# 论文原文：利用正则化模板识别包含否定词的短语结构，将其标记为潜在否定语境
# 设计原则：宁可漏拦（让BERT兜底），不可误拦（正确句不应被reject）
# ─────────────────────────────────────────────────────────────────────────────

# ── 公共子模式 ────────────────────────────────────────────────────────────────
_DEPLOY_VERB = (
    r'(?:部署|应用|使用|实施|建立|创建|构建|搭建|采用|引入|实现|'
    r'配备|接入|建设|开发|开展|启用|推广|推进|落地|落实|上线|投产|'
    r'实装|集成|对接|嵌入|接入|纳入|打造|形成)'
)

_NEGATION_ADV = (
    r'(?:尚未|暂未|目前未|目前尚未|还未|从未|并未|未曾|始终未|'
    r'一直未|迄今未|至今未|仍未|至今仍未|尚不|目前不|并不|从不|'
    r'始终不|还没有|目前没有|尚没有|并没有|一直没有|尚未能|未能|'
    r'无法|不能|不可|不具备|暂不)'
)


# ── 模式一：明确否定（否定副词 + 部署/应用类动词） ────────────────────────────
_DEFINITE_NEGATION = re.compile(
    r'(?:'
    # 1a: 否定副词 → 部署/应用类动词（"尚未部署""并未引入""无法实现"等）
    + _NEGATION_ADV + r'.{0,15}' + _DEPLOY_VERB +
    r'|'
    # 1b: 动词 → 否定完成（"部署尚未完成""应用仍未落地"，注意"完善"不在此列）
    + _DEPLOY_VERB + r'.{0,8}(?:尚未完成|尚未落地|仍未完成|仍未落地|没有完成|未能落地)' +
    r'|'
    # 1c: 不涉及/不包括/缺乏 + 技术/系统/平台/能力
    r'(?:不涉及|不包括|不具备|缺乏|暂不涉及|暂不具备)'
    r'.{0,12}'
    r'(?:数据|技术|系统|平台|能力|功能|基础|条件|模块|应用)'
    r')',
    re.S
)

# ── 模式二：明确未来计划（尚未发生，不计入指标） ─────────────────────────────
_FUTURE_PLAN = re.compile(
    r'(?:'
    # 2a: 明确计划类词 + 部署/引入类动词
    r'(?:计划|拟|拟于|准备|预计|预期|打算|有意|有望|力争|争取|考虑)'
    r'.{0,18}'
    + _DEPLOY_VERB +
    r'|'
    # 2b: 将+副词 + 动词（"将积极引入""将逐步建立""将全面推进"）
    r'将(?:积极|大力|着力|逐步|全面|持续|不断|继续|进一步|努力|尽快|'
    r'加快|稳步|有序|有力|进一步加快|全力|尽快|抓紧|努力)'
    r'.{0,15}'
    + _DEPLOY_VERB +
    r'|'
    # 2c: 将+年份/时间词 + 动词（"将于2025年部署""将在未来建设"）
    r'将(?:于|在)\d{0,4}年?.{0,8}'
    + _DEPLOY_VERB +
    r'|'
    # 2d: 将+直接接动词，无副词（"将建立""将构建""将部署"）
    r'将' + _DEPLOY_VERB +
    r'|'
    # 2e: 下一步/未来/后续/今后 + (将/计划/拟) + 动词
    r'(?:下一步|未来|后续|今后|将来|长远|明年|下年|今年内)'
    r'.{0,12}(?:将|计划|拟|准备|着力|推进)?'
    r'.{0,8}'
    + _DEPLOY_VERB +
    r'|'
    # 2f: 正在研究/论证/评估（尚未落地）
    r'(?:正在研究|正在探索|正在论证|正在规划|正在评估|正在调研|'
    r'正在考察|积极研究|积极探索|深入研究|深入探索)'
    r'.{0,18}'
    + _DEPLOY_VERB +
    r')',
    re.S
)

# ── 模式三：处于探索/起步阶段（尚未落地） ────────────────────────────────────
_EARLY_STAGE = re.compile(
    r'(?:尚处于|仍处于|处于|仍在|尚在|还处于|目前处于|仍属于|尚属于)'
    r'.{0,12}'
    r'(?:研究|探索|起步|论证|规划|试验|试点|调研|考察|初级|初步|萌芽|摸索|筹备|准备)'
    r'.{0,8}'
    r'(?:阶段|时期|过程|期间|中|阶)',
    re.S
)


def regex_prefilter(sentence: str) -> str:
    """
    正则预筛 —— 对应论文"利用正则化模板识别包含否定词的短语结构"

    返回:
      'reject'  → 明确否定/计划/探索阶段，直接排除，无需BERT
      'accept'  → 不含任何否定/将来词，直接保留
      'bert'    → 含模糊否定词（未/不/没/无），交BERT精判
    """
    if (_DEFINITE_NEGATION.search(sentence)
            or _FUTURE_PLAN.search(sentence)
            or _EARLY_STAGE.search(sentence)):
        return 'reject'
    # 含否定类字符但未被上述规则明确捕捉 → 交BERT进一步判断（潜在否定语境）
    if re.search(r'[未不没无]', sentence):
        return 'bert'
    return 'accept'


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：BERT零样本分类（对应论文0.60阈值）
# ─────────────────────────────────────────────────────────────────────────────

class BertDeploymentClassifier:
    """
    基于中文NLI模型的零样本二分类器
    判断句子是否描述"实际部署/应用"（论文阈值=0.60）

    模型：IDEA-CCNL/Erlangshen-Roberta-110M-NLI（本地 models/ 目录）
      - 专为中文NLI训练，推理速度快
      - 支持MPS（Mac GPU）加速
    """

    # 优先使用本地模型（开源项目离线可用），找不到时回落到HuggingFace
    _LOCAL_MODEL = str(Path(__file__).parent.parent / "models" / "Erlangshen-Roberta-110M-NLI")
    MODEL_NAME   = _LOCAL_MODEL if Path(_LOCAL_MODEL).exists() else "IDEA-CCNL/Erlangshen-Roberta-110M-NLI"
    THRESHOLD    = 0.60   # 论文明确给出的分类阈值

    # 零样本分类的假设模板（对应"实际部署/应用"vs"方向性/否定性"）
    HYPOTHESIS_DEPLOY   = "该公司已实际部署或应用了相关数据技术"
    HYPOTHESIS_NEGATION = "该描述仅为方向性表达或否定性陈述"

    def __init__(self, device: str = "auto"):
        print(f"[Module B] 加载BERT模型: {self.MODEL_NAME}")
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                print("[Module B] 使用 Mac MPS GPU 加速")
            else:
                device = "cpu"
                print("[Module B] 使用 CPU")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.MODEL_NAME,
            device=device,
            # 明确设置截断，避免超长句子引发 tensor size 不匹配错误
            tokenizer_kwargs={"truncation": True, "max_length": 512},
        )
        self.device = device

    def predict_batch(self, sentences: list[str]) -> list[bool]:
        """
        批量判断句子是否为"实际部署/应用"语境
        返回 bool 列表，True = 有效（保留），False = 无效（过滤）
        """
        if not sentences:
            return []

        # 预截断：超过 400 字的句子先在字符层面截断，保留前400字
        # （对应 BERT 约 512 token 上限，中文约 1 字 ≈ 1 token）
        sentences_trunc = [s[:400] if len(s) > 400 else s for s in sentences]

        results = self.classifier(
            sentences_trunc,
            candidate_labels=[self.HYPOTHESIS_DEPLOY, self.HYPOTHESIS_NEGATION],
            batch_size=32,
            truncation=True,
        )

        decisions = []
        for res in results:
            idx  = res['labels'].index(self.HYPOTHESIS_DEPLOY)
            prob = res['scores'][idx]
            decisions.append(prob >= self.THRESHOLD)

        return decisions

    def predict_one(self, sentence: str) -> bool:
        return self.predict_batch([sentence])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune版本框架（需要标注数据后使用）
# ─────────────────────────────────────────────────────────────────────────────

class BertFineTunedClassifier:
    """
    精确复现：fine-tune版BERT二分类器
    需先用 label_studio 完成标注，再运行 train() 训练

    标注格式（data/annotation/labeled.csv）：
      sentence,label
      "企业已在生产线部署了机器视觉系统",1
      "公司计划引入人工智能技术",0
      "持续深化数字化转型战略",0
      ...
    建议标注量：600-1000条（各类型均衡）
    """

    BASE_MODEL = "bert-base-chinese"
    THRESHOLD  = 0.60

    def __init__(self, model_path: str = None, device: str = "auto"):
        from transformers import BertForSequenceClassification, BertTokenizer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device    = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path or self.BASE_MODEL)
        self.model     = BertForSequenceClassification.from_pretrained(
            model_path or self.BASE_MODEL, num_labels=2
        ).to(self.device)
        self.model.eval()

    def train(self, labeled_csv: str, output_dir: str = "models/bert_negation",
              epochs: int = 3, batch_size: int = 16):
        """Fine-tune训练"""
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW

        df = pd.read_csv(labeled_csv)

        class SentenceDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True,
                    max_length=256, return_tensors="pt"
                )
                self.labels = torch.tensor(labels)
            def __len__(self): return len(self.labels)
            def __getitem__(self, i):
                return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]

        dataset = SentenceDataset(df['sentence'].tolist(), df['label'].tolist(), self.tokenizer)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=50,
            num_training_steps=len(loader) * epochs
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch, labels in loader:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                labels = labels.to(self.device)
                loss   = self.model(**batch, labels=labels).loss
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"模型已保存至 {output_dir}")

    def predict_batch(self, sentences: list[str]) -> list[bool]:
        results = []
        for i in range(0, len(sentences), 32):
            batch  = sentences[i:i+32]
            inputs = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=256, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            results.extend([p >= self.THRESHOLD for p in probs])
        return results
