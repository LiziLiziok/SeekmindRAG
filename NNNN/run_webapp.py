#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek RAG 多文档问答系统启动脚本
"""

import os
import sys
import time
import importlib
import subprocess

# 检查依赖
try:
    import flask
    from QAsystems.QA import QASystem
    from model_use.embedding_model import init_embedding_model
except ImportError as e:
    print(f"缺少必要的依赖: {e}")
    print("请先运行: pip install -r requirements.txt")
    sys.exit(1)

def check_model_files():
    """检查必要的模型文件是否存在"""
    # 检查向量索引
    if not os.path.exists("vector_store/faiss.index") or not os.path.exists("vector_store/doc_store.pkl"):
        print("未找到向量索引文件。请先运行索引构建脚本:")
        print("python index_add_or_create.py")
        return False
    
    # 检查BM25索引
    if not os.path.exists("BM25_store/bm25_cache.pkl"):
        print("未找到BM25索引文件。请先运行索引构建脚本:")
        print("python index_add_or_create.py")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("DeepSeek RAG 多文档问答系统启动程序")
    print("=" * 50)
    
    # 检查模型文件
    if not check_model_files():
        if input("是否继续启动应用程序？(y/n): ").lower() != 'y':
            return
    
    # 预加载QA系统，预热模型
    print("\n初始化QA系统中，这可能需要几分钟时间...\n")
    start_time = time.time()
    qa = QASystem(use_gpu=True, ollama_use=False)
    end_time = time.time()
    print(f"QA系统初始化完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 启动Flask应用
    print("\n启动Web应用...")
    print("请在浏览器中访问: http://localhost:5000")
    print("按 Ctrl+C 终止程序")
    print("-" * 50)
    
    # 启动应用
    subprocess.run(["python", "app.py"])

if __name__ == "__main__":
    main() 