"""
BERT 模型推理模块，支持单条预测和批量预测
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from config import Config


class NewsPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_PATH, local_files_only=True)
        self.model = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_PATH, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """单条文本预测，返回 (标签, 置信度)"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

        label = Config.ID2LABEL[pred_id]
        return label, confidence

    def predict_batch(self, texts):
        """批量文本预测，返回 [(标签, 置信度), ...]"""
        if not texts:
            return []
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1)

        results = []
        for i, pred_id in enumerate(pred_ids):
            pred_id = pred_id.item()
            confidence = probs[i][pred_id].item()
            label = Config.ID2LABEL[pred_id]
            results.append((label, confidence))
        return results
