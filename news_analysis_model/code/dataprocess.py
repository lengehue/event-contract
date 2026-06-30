"""
数据预处理模块
功能：
1. 清洗原始新闻数据（去重、去除空值、去除特殊字符）
2. 文本标准化（统一编码、去除多余空格）
3. 生成训练/验证/测试集划分文件
4. 数据统计分析
"""

import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config


def clean_text(text):
    """
    清洗文本：去除特殊字符、多余空格、HTML标签等

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""

    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除URL链接
    text = re.sub(r'http[s]?://\S+', '', text)

    # 去除特殊字符，保留中文、英文、数字和常见标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）【】《》·]', '', text)

    # 去除多余空格和换行符
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_and_clean_data(file_path):
    """
    加载并清洗数据

    Args:
        file_path: 数据文件路径

    Returns:
        DataFrame: 清洗后的数据
    """
    print(f"正在加载数据: {file_path}")

    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                content = obj.get('content', '')

                # 清洗文本
                cleaned_content = clean_text(content)

                # 跳过空文本或过短文本
                if len(cleaned_content) < 10:
                    continue

                # 根据关键词生成标签（与model.py保持一致）
                label = generate_sentiment_label(cleaned_content)

                texts.append(cleaned_content)
                labels.append(label)

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue

    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })

    print(f"数据加载完成，共 {len(df)} 条有效记录")
    return df


def generate_sentiment_label(text):
    """
    基于关键词规则生成情感标签

    Args:
        text: 新闻文本

    Returns:
        int: 0表示利空，1表示利好
    """
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


def analyze_data(df):
    """
    数据分析统计

    Args:
        df: 数据DataFrame
    """
    print("\n" + "=" * 60)
    print("数据统计分析")
    print("=" * 60)

    total = len(df)
    bullish = (df['label'] == 1).sum()
    bearish = (df['label'] == 0).sum()

    print(f"\n总样本数: {total}")
    print(f"利好样本: {bullish} ({bullish / total * 100:.2f}%)")
    print(f"利空样本: {bearish} ({bearish / total * 100:.2f}%)")

    # 文本长度统计
    df['text_length'] = df['text'].str.len()
    print(f"\n文本长度统计:")
    print(f"  最小长度: {df['text_length'].min()}")
    print(f"  最大长度: {df['text_length'].max()}")
    print(f"  平均长度: {df['text_length'].mean():.2f}")
    print(f"  中位数: {df['text_length'].median():.2f}")

    # 检查重复
    duplicates = df.duplicated(subset=['text']).sum()
    print(f"\n重复文本数: {duplicates}")

    print("=" * 60 + "\n")


def save_dataset(df, output_dir):
    """
    保存数据集到文件

    Args:
        df: 数据DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整清洗后的数据
    full_path = os.path.join(output_dir, 'cleaned_data.csv')
    df[['text', 'label']].to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"✓ 完整数据已保存: {full_path}")

    # 按8:1:1划分训练集、验证集、测试集
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    # 保存各数据集
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df[['text', 'label']].to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df[['text', 'label']].to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df[['text', 'label']].to_csv(test_path, index=False, encoding='utf-8-sig')

    print(f"✓ 训练集已保存: {train_path} ({len(train_df)} 条)")
    print(f"✓ 验证集已保存: {val_path} ({len(val_df)} 条)")
    print(f"✓ 测试集已保存: {test_path} ({len(test_df)} 条)")

    # 保存为JSONL格式（兼容原有格式）
    for name, data in [('train', train_df), ('val', val_df), ('test', test_df)]:
        jsonl_path = os.path.join(output_dir, f'{name}.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in data.iterrows():
                obj = {
                    'content': row['text'],
                    'label': int(row['label'])
                }
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        print(f"✓ JSONL格式已保存: {jsonl_path}")


def preprocess_data(input_file=None, output_dir=None):
    """
    主预处理函数

    Args:
        input_file: 输入数据文件路径（默认使用Config.DATA_PATH）
        output_dir: 输出目录（默认使用data目录下的processed子目录）
    """
    if input_file is None:
        input_file = Config.DATA_PATH

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(Config.DATA_PATH), 'processed')

    print("=" * 60)
    print("金融新闻数据预处理")
    print("=" * 60)

    # 1. 加载并清洗数据
    df = load_and_clean_data(input_file)

    # 2. 数据分析
    analyze_data(df)

    # 3. 保存数据集
    save_dataset(df, output_dir)

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  - cleaned_data.csv  : 完整清洗后的数据")
    print("  - train.csv         : 训练集（80%）")
    print("  - val.csv           : 验证集（10%）")
    print("  - test.csv          : 测试集（10%）")
    print("  - train.jsonl       : 训练集（JSONL格式）")
    print("  - val.jsonl         : 验证集（JSONL格式）")
    print("  - test.jsonl        : 测试集（JSONL格式）")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    preprocess_data()
