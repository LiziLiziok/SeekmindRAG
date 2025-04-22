# import importlib
# import retrieval_strategy.faiss_module
# import model_use.generate_model

# # 模块热重载（开发调试时用）
# importlib.reload(retrieval_strategy.faiss_module)
# importlib.reload(model_use.generate_model)

# from retrieval_strategy.faiss_module import load_faiss_index, search_documents
# from model_use.generate_model import init_deepseek_model

# class QASystem:
#     def __init__(self, use_gpu=True, ollama_use=True):
#         self.use_gpu = use_gpu
#         self.index, self.doc_store = load_faiss_index()
#         self.llm = init_deepseek_model(ollama_use=ollama_use)

#     def ask(self, query, k=5):
#         D, results = search_documents(query, self.index, self.doc_store, use_gpu=self.use_gpu, k=k)
#         context = "\n".join(results)
#         prompt_question = f"基于以下内容，回答问题：\n\n{context}\n\n问题：{query}"
#         answer = self.llm(prompt_question)
#         return D, results, prompt_question, answer
import importlib
import retrieval_strategy.faiss_module
import model_use.generate_model
import retrieval_strategy.retrieval_general_method
import model_use.rerank_model

# 模块热重载（开发调试时用）
importlib.reload(retrieval_strategy.faiss_module)
importlib.reload(model_use.generate_model)
importlib.reload(retrieval_strategy.retrieval_general_method)
importlib.reload(model_use.rerank_model)

from retrieval_strategy.faiss_module import load_faiss_index, search_documents
from model_use.generate_model import init_deepseek_model
from retrieval_strategy.retrieval_general_method import hybrid_retrieval,BM25Index
from model_use.rerank_model import init_rerank_model

class QASystem:
    def __init__(self, use_gpu=True, ollama_use=True):
        self.use_gpu = use_gpu
        self.index, self.doc_store = load_faiss_index()  # 加载FAISS索引
        self.llm = init_deepseek_model(ollama_use=ollama_use)  # 初始化深度学习模型
        self.rerank_model = init_rerank_model()  # 初始化rerank模型
        self.bm25_index = BM25Index()  # 初始化BM25索引

    def ask(self, query, k=5):
        # 使用混合检索方法
        reranked_results, reranked_scores = hybrid_retrieval(query, self.index, self.doc_store, self.bm25_index, k=k, rerank_model=self.rerank_model)
        
        # 打印并返回文档及相关信息
        context = "\n".join(reranked_results)

        prompt_question = (
        f"你是一个图书管理员，请参考以下书本语句，简洁直接地回答提问者的问题，不要输出思维链：【{context}】。\n\n"
        f"提问者：{query}\n"
        "图书管理员："
        )

        answer = self.llm(prompt_question)
        
        return reranked_scores, reranked_results, prompt_question, answer
