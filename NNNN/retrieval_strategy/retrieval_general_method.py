import os
import pickle
from rank_bm25 import BM25Okapi
import hanlp
import numpy as np
from collections import OrderedDict
import importlib
import model_use.rerank_model
importlib.reload(model_use.rerank_model)
from model_use.rerank_model import init_rerank_model




class BM25Index:
    def __init__(self, cache_path='BM25_store/bm25_cache.pkl'):
        self.cache_path = cache_path
        self.documents = []
        self.tokenized_documents = []
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self.bm25 = None
        self._load()

    def _load(self):
        """从缓存加载 BM25 索引"""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.tokenized_documents = data['tokenized_documents']
                self._update_bm25()
                print(f"[+] Loaded BM25 cache from {self.cache_path}")

    def _save(self):
        """保存索引到本地文件"""
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'tokenized_documents': self.tokenized_documents
            }, f)
            print(f"[+] Saved BM25 index to {self.cache_path}")

    def _update_bm25(self):
        """更新 BM25 模型"""
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def build_from_txt_results(self, documents,tokenized_documents):
        """初次构建索引"""
        self.documents = documents
        self.tokenized_documents = tokenized_documents
        self._update_bm25()
        self._save()

    def retrieve(self, query, k=5):
        """检索接口"""
        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        # top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_k_idx = np.argsort(scores)[-k:][::-1]
        print(f"bm25 top {k} 个结果分数: {top_k_idx}")
        return top_k_idx,[self.documents[i] for i in top_k_idx]
    
    def add_documents(self, new_docs,new_tokens):
        """
        批量添加多个中文文档。
        new_docs:list[str] 要添加的文档列表
        new_tokens:list[list[str]] 要添加的文档列表的分词结果
        """

        if not new_docs:
            print("[!] 没有有效的新文档添加。")
            return

        self.documents.extend(new_docs)
        self.tokenized_documents.extend(new_tokens)

        self._update_bm25()  # 重新初始化 BM25 模型
        self._save()         # 保存更新后的数据

        print(f"[+] 成功添加 {len(new_docs)} 个新文档！")

import importlib
import retrieval_strategy.faiss_module
importlib.reload(retrieval_strategy.faiss_module)
from retrieval_strategy.faiss_module import search_documents
import re


# 混合检索部分
def hybrid_retrieval(query, faiss_index, doc_store, bm25_index, k=5, rerank_model=init_rerank_model()):
    """混合检索：结合密集检索和稀疏检索"""
    # 执行密集检索
    dense_scores,dense_results = search_documents(query, faiss_index, doc_store, k=k)
    
    # 执行稀疏检索
    sparse_scores,sparse_results = bm25_index.retrieve(query, k=k)

    passages = dense_results + sparse_results
    # 去除重复的passage
    # for i in passages:
    #     print(i)

    # 去除中文文本中的空格和标点符号
    def clean_text(text):
        # 去除多余的空格
        text = text.strip()
        # 使用正则去掉所有的非中文字符和空格
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 保留中文字符
        return text

    # 合并 dense 和 sparse 的结果
    passages = dense_results + sparse_results

    # 去除重复的passage
    combined = [("dense", p) for p in dense_results] + [("sparse", p) for p in sparse_results]

    # 创建一个OrderedDict来去重，保持插入顺序
    unique_dict = OrderedDict()

    # 去重并调试输出
    for source, p in combined:
        # 清理文本，去除无效字符
        cleaned_passage = clean_text(p)
        if cleaned_passage not in unique_dict:
            unique_dict[cleaned_passage] = source
        else:
            # print(f"重复发现: {cleaned_passage}")  # 输出重复的元素，帮助调试
            pass

    # 获取去重后的passages
    unique_passages = list(unique_dict.keys())  # 去重后的passage列表
    source_info = list(unique_dict.values())    # 来源（dense或sparse）

    # 打印去重后的结果
    # for passage in unique_passages:
    #     print(passage)

    # 使用rerank模型对检索结果进行重排序
    scores = rerank_model.predict([(query, passage) for passage in unique_passages])
    reranked_results = [unique_passages[i] for i in np.argsort(scores)[-k:][::-1]]
    reranked_scores = [scores[i] for i in np.argsort(scores)[-k:][::-1]]


    return reranked_results, reranked_scores