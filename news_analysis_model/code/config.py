import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # ========== 数据路径 ==========
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'financial_news_2019.csv')

    # ========== 模型路径 ==========
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    BERT_MODEL_NAME = os.path.join(MODEL_DIR, 'bert-base-chinese')
    BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_sentiment')

    # ========== 训练参数 ==========
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_LABELS = 2

    # ========== 标签映射 ==========
    LABEL2ID = {'利空': 0, '利好': 1}
    ID2LABEL = {0: '利空', 1: '利好'}

    # ========== Flask API ==========
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000

    # ========== Gradio ==========
    GRADIO_SERVER_NAME = '0.0.0.0'
    GRADIO_SERVER_PORT = 7860
