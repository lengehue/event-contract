"""
新闻文本分类模型
使用 TF-IDF + 随机森林(RandomForest) 对金融新闻进行利好/利空二分类

数据: financial_news_2019.csv (JSON Lines格式, 含content字段)
标签: 1=利好, 0=利空
"""

import json
import os
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import jieba


MODEL_DIR = r'D:\code\python\news_analysis_model\model'


def load_data(file_path):
    """加载 JSON Lines 格式的CSV文件"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    texts.append(obj.get('content', ''))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame({'content': texts})


def generate_sentiment_label(text):
    """
    基于关键词规则判断新闻情感倾向 (利好/利空)
    1 = 利好 (正面消息)
    0 = 利空 (负面消息)
    实际使用时请替换为真实标注数据
    """
    if pd.isna(text):
        return 0

    positive_keywords = [
        '增长', '上涨', '突破', '新高', '利好', '盈利', '中标', '签约',
        '合作', '创新', '改革', '开放', '扩张', '获批', '获奖', '提速',
        '优化', '领先', '提升', '改善', '稳健', '复苏', '景气', '热销',
        '大卖', '翻倍', '超额', '完成', '达成', '战略', '投资', '融资',
        '回购', '增持', '买入', '推荐', '看好', '机遇', '发展', '普惠',
    ]
    negative_keywords = [
        '下跌', '下滑', '亏损', '减持', '利空', '爆雷', '违约', '逾期',
        '诉讼', '处罚', '警示', '退市', '暴跌', '预亏', '负债', '债务',
        '风险', '危机', '裁员', '停产', '投诉', '维权', '欺诈', '违规',
        '被查', '调查', '做空', '减值', '商誉', '下调', '卖出', '减持',
        '质押', '冻结', '追逃', '判刑', '诈骗', '泡沫', '困境', '萎缩',
    ]

    pos_score = sum(1 for kw in positive_keywords if kw in text)
    neg_score = sum(1 for kw in negative_keywords if kw in text)

    if pos_score > neg_score:
        return 1
    elif neg_score > pos_score:
        return 0
    else:
        return 1 if pos_score > 0 else 0


def chinese_tokenizer(text):
    """使用jieba进行中文分词"""
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    words = jieba.lcut(text)
    return ' '.join(w for w in words if len(w) > 1)


def save_model(model, vectorizer, save_dir):
    """保存模型和向量化器"""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'rf_model.pkl')
    vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"模型已保存至: {model_path}")
    print(f"向量化器已保存至: {vectorizer_path}")


def main():
    # ========================
    # 1. 加载数据
    # ========================
    data_path = r'D:\code\python\news_analysis_model\data\financial_news_2019.csv'
    df = load_data(data_path)
    print(f"数据总量: {len(df)} 条")

    # ========================
    # 2. 生成情感标签: 1=利好, 0=利空
    # ========================
    df['label'] = df['content'].apply(generate_sentiment_label)
    print(f"\n标签分布:")
    print(f"  利好(1): {(df['label'] == 1).sum()} 条")
    print(f"  利空(0): {(df['label'] == 0).sum()} 条")

    # ========================
    # 3. 中文分词
    # ========================
    print("\n正在进行中文分词...")
    df['text_cut'] = df['content'].apply(chinese_tokenizer)

    # ========================
    # 4. 划分训练集和测试集
    # ========================
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_cut'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # ========================
    # 5. TF-IDF 特征提取
    # ========================
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF 特征维度: {X_train_tfidf.shape[1]}")

    # ========================
    # 6. 随机森林模型训练
    # ========================
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_tfidf, y_train)

    # ========================
    # 7. 模型评估
    # ========================
    y_pred = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"随机森林模型准确率: {acc:.4f}")
    print(f"{'='*40}")
    print(f"\n分类报告:")
    print(classification_report(
        y_test, y_pred,
        target_names=['利空(0)', '利好(1)'],
        zero_division=0
    ))

    # ========================
    # 8. 特征重要性 Top 20
    # ========================
    feature_names = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    top_indices = importances.argsort()[::-1][:20]
    print("Top 20 重要特征:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    # ========================
    # 9. 保存模型
    # ========================
    save_model(rf, tfidf, MODEL_DIR)


if __name__ == '__main__':
    main()
