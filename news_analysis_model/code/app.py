"""
Gradio Web 界面
黑白配色，简洁美观
支持单条分类和批量分类
"""

import gradio as gr
from predict import NewsPredictor
from config import Config

predictor = NewsPredictor()

CUSTOM_CSS = """
body { background-color: #111111; color: #f0f0f0; }
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background-color: #1a1a1a !important;
    border-radius: 16px !important;
    padding: 30px !important;
}
h1 {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-align: center !important;
}
h2 { color: #cccccc !important; font-size: 16px !important; font-weight: 400 !important; text-align: center !important; }
button {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
button:hover { background-color: #e0e0e0 !important; }
"""


def single_predict(text):
    if not text or not text.strip():
        return "请输入新闻文本", ""
    label, confidence = predictor.predict(text.strip())
    if label == '利好':
        emoji, desc = "📈", "该新闻被判定为【利好消息】，对市场和投资者偏正面"
    else:
        emoji, desc = "📉", "该新闻被判定为【利空消息】，对市场和投资者偏负面"
    return f"{emoji} {label}", f"{desc}\n置信度: {confidence:.2%}"


def batch_predict(texts):
    if not texts or not texts.strip():
        return "请在左侧输入框中输入多条新闻，每行一条"
    lines = [line.strip() for line in texts.strip().split('\n') if line.strip()]
    if not lines:
        return "请输入至少一条新闻"
    if len(lines) > 50:
        lines = lines[:50]

    results = predictor.predict_batch(lines)
    bullish = sum(1 for label, _ in results if label == '利好')
    bearish = len(lines) - bullish

    output_lines = [f"**共 {len(lines)} 条 | 利好 {bullish} 条 | 利空 {bearish} 条**", "---"]
    for i, (text, (label, conf)) in enumerate(zip(lines, results)):
        emoji = "📈" if label == '利好' else "📉"
        short = text[:60] + ('...' if len(text) > 60 else '')
        output_lines.append(f"{i+1}. {emoji} **{label}** ({conf:.1%}) | {short}")
    return '\n\n'.join(output_lines)


SAMPLE_TEXTS = [
    "长沙银行一季度营收42亿元，同比增长29.74%，净利润增长11.02%",
    "中信资本旗下多个私募产品逾期违约，涉及金额超30亿元",
    "宝马集团2018年在华销量创新高，同比增长7.7%",
]


def create_app():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
        gr.Markdown("# 📊 金融新闻文本分析系统\n## 基于 BERT 预训练模型 — 利好 / 利空 智能分类")

        with gr.Tabs():
            with gr.Tab("📝 单条新闻分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_text = gr.Textbox(label="输入新闻文本", placeholder="请输入一段金融新闻内容...", lines=6)
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 分析", variant="primary", scale=1)
                            clear_btn = gr.ClearButton(scale=1)
                    with gr.Column(scale=1):
                        label_output = gr.Textbox(label="分类结果", interactive=False, lines=2)
                        detail_output = gr.Textbox(label="详细说明", interactive=False, lines=4)
                gr.Examples(examples=[[s] for s in SAMPLE_TEXTS], inputs=[input_text], label="示例文本")
                analyze_btn.click(fn=single_predict, inputs=[input_text], outputs=[label_output, detail_output])
                clear_btn.add([input_text, label_output, detail_output])

            with gr.Tab("📋 批量新闻分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Textbox(label="输入多条新闻（每行一条）", placeholder="新闻1...\n新闻2...\n新闻3...", lines=10)
                        batch_btn = gr.Button("🚀 批量分析", variant="primary")
                    with gr.Column(scale=1):
                        batch_output = gr.Markdown(value="等待分析...")
                batch_btn.click(fn=batch_predict, inputs=[batch_input], outputs=[batch_output])

        gr.Markdown("---\n<div style='text-align:center;color:#666;font-size:12px;'>Powered by BERT-base-chinese | Flask + Gradio</div>")
    return demo


if __name__ == '__main__':
    demo = create_app()
    demo.launch(server_name=Config.GRADIO_SERVER_NAME, server_port=Config.GRADIO_SERVER_PORT, share=False)
