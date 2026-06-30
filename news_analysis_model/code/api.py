"""
Flask API 服务
提供单条分类和批量分类接口
"""

from flask import Flask, request, jsonify
from predict import NewsPredictor
from config import Config

app = Flask(__name__)
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = NewsPredictor()
    return predictor


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    单条文本分类
    请求体: {"text": "新闻文本内容"}
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '请提供 text 字段'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': '文本内容不能为空'}), 400

    p = get_predictor()
    label, confidence = p.predict(text)
    return jsonify({
        'text': text[:100] + ('...' if len(text) > 100 else ''),
        'label': label,
        'confidence': round(confidence, 4)
    })


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量文本分类
    请求体: {"texts": ["新闻1", "新闻2", ...]}
    """
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': '请提供 texts 字段（列表）'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts 必须为非空列表'}), 400

    if len(texts) > 100:
        return jsonify({'error': '单次批量请求不能超过100条'}), 400

    p = get_predictor()
    results = p.predict_batch(texts)

    response = []
    bullish_count = 0
    bearish_count = 0
    for i, (text, (label, confidence)) in enumerate(zip(texts, results)):
        response.append({
            'index': i,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'label': label,
            'confidence': round(confidence, 4)
        })
        if label == '利好':
            bullish_count += 1
        else:
            bearish_count += 1

    return jsonify({
        'total': len(texts),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'results': response
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'bert-base-chinese'})


if __name__ == '__main__':
    get_predictor()
    print(f"Flask API 启动: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=False)
