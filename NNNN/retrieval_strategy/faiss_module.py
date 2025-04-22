import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import torch
from faiss import normalize_L2
import sys
# 获取当前文件所在目录的父目录（项目根目录）并添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import importlib
import model_use.embedding_model
importlib.reload(model_use.embedding_model)
from model_use.embedding_model import init_embedding_model


# class RetrievalStrategy:
#     def __init__(self):
#         self.sentence_transformer = SentenceTransformer("models/all-MiniLM-L6-v2")
#         self.index = None
#         self.doc_store = []  # 存储原始文本，与索引中的向量一一对应

#     def build_faiss_index(self, documents):
#         """
#         documents:就是文本分块的列表
#         """
#         embeddings = self.sentence_transformer.encode(documents, convert_to_numpy=True)
#         dim = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)
#         self.doc_store = documents

#         # 可选：保存
#         faiss.write_index(self.index, "vector_store/faiss.index")
#         with open("vector_store/doc_store.pkl", "wb") as f:
#             pickle.dump(self.doc_store, f)

#     def load_faiss_index(self):
#         self.index = faiss.read_index("vector_store/faiss.index")
#         with open("vector_store/doc_store.pkl", "rb") as f:
#             self.doc_store = pickle.load(f)
#         return self.index,self.doc_store
    
def build_faiss_index(documents, model_path="models/all-MiniLM-L6-v2",index_type="IndexHNSWFlat"):
    """构建一个崭新的 FAISS 索引
    index_type: 索引类型，可选值为 "IndexFlatL2"、"IndexFlatIP"、"IndexHNSWFlat"
    """
    # 确保向量存储目录存在
    os.makedirs("vector_store", exist_ok=True)
    
    # 初始化模型
    sentence_transformer = SentenceTransformer(model_path)
    
    # 生成嵌入向量
    embeddings = sentence_transformer.encode(documents, convert_to_numpy=True)
    dim = embeddings.shape[1]
    
    # 创建和添加索引
    if index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)  # 精确向量索引，基于L2距离
    elif index_type == "IndexFlatIP":
        faiss.normalize_L2(embeddings)  # 将向量归一化（单位化）
        index = faiss.IndexFlatIP(dim)  # 内积索引，用于计算余弦相似度
    elif index_type == "IndexHNSWFlat":  # 推荐使用HNSW，特别适合大规模数据集
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32为常见选择，影响索引的速度和精度
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
    
    # 将嵌入向量添加到索引
    index.add(embeddings)
    
    # 保存索引和文档存储
    faiss.write_index(index, "vector_store/faiss.index")
    with open("vector_store/doc_store.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    return index

def add_documents(documents, index, model_path="models/all-MiniLM-L6-v2"):
    """已有向量数据库，动态添加文档"""
    # 生成嵌入向量
    sentence_transformer = SentenceTransformer(model_path)
    embeddings = sentence_transformer.encode(documents, convert_to_numpy=True)
    # 加载已有索引
    index = faiss.read_index("vector_store/faiss.index")
    # 加载对应的文档内容（用来最终返回匹配文档）
    with open("vector_store/doc_store.pkl", "rb") as f:
        doc_store = pickle.load(f)

    # 添加新的向量到索引
    if type(index) == faiss.IndexHNSWFlat:
        index.add(embeddings)
    elif type(index) == faiss.IndexFlatL2:
        index.add(embeddings)
    elif type(index) == faiss.IndexFlatIP:
        normalize_L2(embeddings)
        index.add(embeddings)
    else:
        raise ValueError(f"不支持的索引类型: {type(index)}")
    
    # 添加新内容到文档存储
    doc_store.extend(documents)

    # 保存新的索引
    faiss.write_index(index, "vector_store/faiss.index")

    # 保存新的文档存储
    with open("vector_store/doc_store.pkl", "wb") as f:
        pickle.dump(doc_store, f)


def load_faiss_index():
    """加载 FAISS 索引"""
    if not os.path.exists("vector_store/faiss.index"):
        raise FileNotFoundError("找不到索引文件，请先构建索引")
        
    index = faiss.read_index("vector_store/faiss.index")
    with open("vector_store/doc_store.pkl", "rb") as f:
        doc_store = pickle.load(f)
    return index, doc_store

def search_documents(query, index, doc_store, model_path="models/all-MiniLM-L6-v2", k=5, use_gpu=False):
    """搜索文档"""
    # 使用GPU加速（推荐大规模文档时使用）：
    print(f"use_gpu: {use_gpu}")
    print(f"index_type: {type(index)}")
    if use_gpu:
        # 初始化 GPU 资源
        res = faiss.StandardGpuResources()
        # 将索引转换为 GPU 索引
        index = faiss.index_cpu_to_gpu(res, 0, index)

    sentence_transformer = SentenceTransformer(model_path)
    query_vector = sentence_transformer.encode([query], convert_to_numpy=True)

    D, I = index.search(query_vector, k)
    
    results = []
    for idx in I[0]:
        results.append(doc_store[idx])
    return D[0], results

if __name__ == "__main__":
    # 示例
    import processor.main_file_chunk
    importlib.reload(processor.main_file_chunk)
    from processor.main_file_chunk import main_file_chunk, txt_results_to_documents
    # 构建索引
    txt_results = main_file_chunk()
    documents, metadatas = txt_results_to_documents(txt_results)
    index = build_faiss_index(documents)
    # 动态添加
    txt_results = main_file_chunk(directory_path = "new_files")
    documents, metadatas = txt_results_to_documents(txt_results)
    add_documents(documents, index)
    # 搜索文档
    query = "中国现代化的定义"
    index, doc_store = load_faiss_index(directory_path = "new_files")
    D, results = search_documents(query, index, doc_store, use_gpu=True)
    print(D)
    print(results)

