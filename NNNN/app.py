from flask import Flask, render_template, request, jsonify, session
import time
import importlib
import QAsystems.QA
importlib.reload(QAsystems.QA)
from QAsystems.QA import QASystem
import os
import uuid
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于session加密

# 存储对话历史的字典（在生产环境中应使用数据库）
conversations = {}

# 可用模型列表
AVAILABLE_MODELS = {
    "deepseek_local": "DeepSeek-R1-Distill-Qwen-7B (本地)",
    "deepseek_ollama": "DeepSeek-R1-Distill-Qwen-7B (Ollama)",
}

# 创建QA系统实例的工厂函数
def create_qa_system(model_choice="deepseek_local"):
    use_ollama = (model_choice == "deepseek_ollama")
    return QASystem(use_gpu=True, ollama_use=use_ollama)

@app.route('/')
def index():
    # 为新用户创建会话ID
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        conversations[session['conversation_id']] = {
            "history": [],
            "model": "deepseek_local"  # 默认模型
        }
    
    return render_template('index.html', 
                          models=AVAILABLE_MODELS,
                          current_model=conversations[session['conversation_id']]["model"])

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    model_choice = data.get('model', 'deepseek_local')
    conversation_id = session['conversation_id']
    
    # 更新对话的模型选择
    conversations[conversation_id]["model"] = model_choice
    
    # 创建或获取QA系统
    qa = create_qa_system(model_choice)
    
    start_time = time.time()
    
    # 调用QA系统获取答案
    reranked_scores, reranked_results, prompt_question, answer = qa.ask(query, k=5)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    # 构建响应数据
    response_data = {
        "answer": answer,
        "sources": reranked_results[:3],  # 仅返回前3个来源
        "source_scores": [float(score) for score in reranked_scores[:3]],  # 对应的分数
        "response_time": f"{response_time:.2f}秒"
    }
    
    # 更新对话历史
    conversations[conversation_id]["history"].append({
        "query": query,
        "response": response_data,
        "timestamp": time.time()
    })
    
    return jsonify(response_data)

@app.route('/api/history', methods=['GET'])
def get_history():
    conversation_id = session.get('conversation_id')
    if not conversation_id or conversation_id not in conversations:
        return jsonify({"history": []})
    
    return jsonify({"history": conversations[conversation_id]["history"]})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    conversation_id = session.get('conversation_id')
    if conversation_id and conversation_id in conversations:
        # 保留模型选择，但清空历史
        model_choice = conversations[conversation_id]["model"]
        conversations[conversation_id] = {
            "history": [],
            "model": model_choice
        }
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # 确保模板和静态文件目录存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 