"""
BERT 金融新闻情感分类模型训练
使用 bert-base-chinese 微调，二分类：利好 / 利空
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import Config


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    content = obj.get('content', '')
                    if content:
                        texts.append(content)
                except json.JSONDecodeError:
                    continue
    return texts


def generate_sentiment_label(text):
    if not text:
        return 0
    positive_keywords = [
        '增长', '上涨', '突破', '新高', '利好', '盈利', '中标', '签约',
        '合作', '创新', '改革', '开放', '扩张', '获批', '获奖', '提速',
        '优化', '领先', '提升', '改善', '稳健', '复苏', '景气', '热销',
        '大卖', '翻倍', '超额', '完成', '达成', '战略', '投资', '融资',
        '回购', '增持', '买入', '推荐', '看好', '机遇', '发展', '普惠',
        '净利润增长', '营收增长', '同比上升', '同比增', '同比上涨',
    ]
    negative_keywords = [
        '下跌', '下滑', '亏损', '减持', '利空', '爆雷', '违约', '逾期',
        '诉讼', '处罚', '警示', '退市', '暴跌', '预亏', '负债', '债务',
        '风险', '危机', '裁员', '停产', '投诉', '维权', '欺诈', '违规',
        '被查', '调查', '做空', '减值', '商誉', '下调', '卖出', '减持',
        '质押', '冻结', '追逃', '判刑', '诈骗', '泡沫', '困境', '萎缩',
        '净利润下滑', '同比下降', '同比降', '同比下跌', '负增长',
    ]
    pos_score = sum(1 for kw in positive_keywords if kw in text)
    neg_score = sum(1 for kw in negative_keywords if kw in text)
    if pos_score > neg_score:
        return 1
    elif neg_score > pos_score:
        return 0
    else:
        return 1 if pos_score > 0 else 0


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("正在加载数据...")
    texts = load_data(Config.DATA_PATH)
    labels = [generate_sentiment_label(t) for t in texts]
    print(f"数据总量: {len(texts)} 条")
    print(f"利好: {labels.count(1)} 条, 利空: {labels.count(0)} 条")

    # 2. 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")

    # 3. 加载 Tokenizer 和模型
    print(f"正在加载 BERT 模型: {Config.BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        Config.BERT_MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        local_files_only=True
    )
    model.to(device)

    # 4. 创建数据集
    train_dataset = NewsDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = NewsDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 5. 优化器和学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # 6. 训练循环
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # 7. 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"训练准确率: {train_acc:.4f} | "
              f"验证准确率: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(Config.BERT_MODEL_PATH, exist_ok=True)
            model.save_pretrained(Config.BERT_MODEL_PATH)
            tokenizer.save_pretrained(Config.BERT_MODEL_PATH)
            print(f"  >>> 保存最佳模型 (准确率: {best_acc:.4f})")

    print(f"\n{'='*50}")
    print(f"训练完成! 最佳验证准确率: {best_acc:.4f}")
    print(f"模型保存路径: {Config.BERT_MODEL_PATH}")
    print(f"{'='*50}")

    # 8. 最终评估
    model = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels_batch.cpu().numpy())

    print(f"\n最终分类报告:")
    print(classification_report(
        final_labels, final_preds,
        target_names=['利空', '利好'],
        zero_division=0
    ))


if __name__ == '__main__':
    train()
