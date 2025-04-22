import importlib
from typing import List, Tuple
import numpy as np
import torch
import argparse
import os
import faiss
from sentence_transformers import SentenceTransformer
import hanlp

# 导入并重新加载必要的模块
import model_use.rerank_model
import model_use.generate_model
import model_use.embedding_model
importlib.reload(model_use.rerank_model)
importlib.reload(model_use.generate_model)
importlib.reload(model_use.embedding_model)

# 导入评估相关组件
from evaluators.base_evaluator import BaseEvaluator
from eval_project import evaluate_rag
from model_use.rerank_model import init_rerank_model
from model_use.generate_model import init_deepseek_model
from model_use.embedding_model import init_embedding_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估RAG系统')
    parser.add_argument('--dataset_path', type=str, default='data/multidoc_qa_2.jsonl',
                      help='数据集文件路径')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                      help='是否使用GPU')
    parser.add_argument('--initial_k', type=int, default=10,
                      help='初始检索的文档数量')
    return parser.parse_args()


def build_faiss_index_eval(documents, model_path="models/all-MiniLM-L6-v2",index_type="IndexHNSWFlat"):
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
    return index


def create_temp_retrieval_function(embedding_model: SentenceTransformer, rerank_model, device: torch.device, k: int = 10, index_type: str = "IndexHNSWFlat"):
    """
    创建基于临时向量索引的检索函数
    
    Args:
        embedding_model: 用于生成文档嵌入的模型
        rerank_model: 用于重排序的模型
        device: 计算设备
        k: 检索的文档数量
        index_type: 索引类型，可选值为 "IndexFlatL2"、"IndexFlatIP"、"IndexHNSWFlat"
    """
    def retrieval_function(documents: List[str], question: str) -> Tuple[List[str], List[float]]:
        try:
            if not documents:
                print("警告：输入文档列表为空")
                return [], []
                
            # 1. 为当前文档创建嵌入
            print(f"为当前文档创建嵌入，文档数量: {len(documents)}")
            # print(f"documents: ")
            # for doc in documents:
            #     print(f"doc[:10]: {doc[:10]}")
            with torch.cuda.amp.autocast():
                embeddings = embedding_model.encode(documents, convert_to_numpy=True)
            print(f"为当前文档创建嵌入成功")
            
            # 检查嵌入是否成功生成
            if isinstance(embeddings, tuple):
                print("警告：嵌入结果是元组类型，尝试转换")
                if len(embeddings) > 0 and hasattr(embeddings[0], 'shape'):
                    embeddings = embeddings[0]  # 取第一个元素
                else:
                    print("警告：无法转换元组为有效嵌入")
                    return [], []
                
            if hasattr(embeddings, 'size') and (embeddings.size == 0 or len(embeddings) == 0):
                print("警告：生成的嵌入为空")
                return [], []
            
            # 确保embeddings是numpy数组
            if not isinstance(embeddings, np.ndarray):
                print("警告：嵌入不是numpy数组，尝试转换")
                try:
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.detach().cpu().numpy()
                    else:
                        embeddings = np.array(embeddings)
                except Exception as e:
                    print(f"转换嵌入为numpy数组失败: {str(e)}")
                    return [], []
            
            # 确保embeddings是2D数组
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # 2. 创建临时FAISS索引（密集检索）
            dimension = embeddings.shape[1]
            if index_type == "IndexFlatL2":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IndexFlatIP":
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(dimension)
            elif index_type == "IndexHNSWFlat":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"不支持的索引类型: {index_type}")
            
            # 3. 将索引转移到GPU（如果可用）
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # 4. 添加文档到索引
            index.add(embeddings)
            
            # 5. 编码问题
            with torch.cuda.amp.autocast():
                query_embedding = embedding_model.encode(question, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)
            
            # 6. 密集检索相关文档
            D, I = index.search(query_embedding, min(k, len(documents)))
            
            # 获取密集检索结果
            dense_results = []
            dense_scores = []
            try:
                for i, idx in enumerate(I[0]):
                    if idx < len(documents):  # 确保索引有效
                        dense_results.append(documents[idx])
                        dense_scores.append(-D[0][i])  # 将距离转换为分数
            except IndexError as e:
                print(f"密集检索索引错误: {str(e)}")
                # 如果索引出错，尝试使用可用的数据
                if len(I) > 0 and len(I[0]) > 0:
                    valid_indices = [idx for idx in I[0] if idx < len(documents)]
                    dense_results = [documents[idx] for idx in valid_indices]
                    dense_scores = [-d for d in D[0][:len(valid_indices)]] if len(D) > 0 else [1.0] * len(valid_indices)
                
            if not dense_results:
                dense_scores = []
                
            print(f"密集检索结果数量: {len(dense_results)}")
            # print(f"密集检索结果: ")
            # for doc in dense_results:
            #     print(f"doc[:10]: {doc[:10]}")            
            
            # 7. 创建BM25检索（稀疏检索）
            # 这里我们简化实现，假设文档已经是小段落
            # 实际项目中可能需要使用完整的BM25Index类
            import re
            from rank_bm25 import BM25Okapi
            
            # 简单分词（实际项目可能使用hanlp等分词器）
            tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            tokenized_documents = [tokenizer(doc) for doc in documents]  # 修正：为每个文档分词
            query_tokens = tokenizer(question)  # 修正：对问题也使用相同分词器
            
            try:
                bm25 = BM25Okapi(tokenized_documents)
                sparse_scores = bm25.get_scores(query_tokens)
                top_k_idx = np.argsort(sparse_scores)[-min(k, len(documents)):][::-1]
                
                # 获取稀疏检索结果
                sparse_results = []
                for idx in top_k_idx:
                    if idx < len(documents):  # 确保索引有效
                        sparse_results.append(documents[idx])
                
                print(f"稀疏检索结果数量: {len(sparse_results)}")
                # print(f"稀疏检索结果: ")
                # for doc in sparse_results:
                #     print(f"doc[:10]: {doc[:10]}")  
            except Exception as e:
                print(f"稀疏检索错误: {str(e)}")
                sparse_results = []  # 如果稀疏检索失败，使用空列表
            
            # 8. 混合检索结果并去重
            # 去除中文文本中的空格和标点符号的函数
            def clean_text(text):
                # 确保text是字符串类型
                if not isinstance(text, str):
                    return ""
                # 去除多余的空格
                text = text.strip()
                # 使用正则去掉所有的非中文字符和空格
                text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 保留中文字符
                return text
            
            # 合并密集检索和稀疏检索的结果
            combined = [("dense", p) for p in dense_results] + [("sparse", p) for p in sparse_results]
            
            # 创建一个OrderedDict来去重，保持插入顺序
            from collections import OrderedDict
            unique_dict = OrderedDict()
            
            # 去重
            for source, p in combined:
                # 清理文本，去除无效字符
                cleaned_passage = clean_text(p)
                if cleaned_passage and cleaned_passage not in unique_dict:
                    unique_dict[cleaned_passage] = p  # 存储原始文本
            
            # 获取去重后的passages
            unique_passages = list(unique_dict.values())  # 去重后的passage列表(原始文本)
            
            print(f"混合检索去重后结果数量: {len(unique_passages)}")
            print(f"混合检索去重后结果: ")
            for doc in unique_passages:
                print(f"doc[:10]: {doc[:10]}")

            # 9. 如果结果为空，返回空列表
            if not unique_passages:
                print("警告：混合检索结果为空")
                return [], []
            
            # 10. 使用重排模型对结果进行重排序
            if rerank_model is not None and unique_passages:
                try:
                    # 准备输入对
                    pairs = [(question, doc) for doc in unique_passages]
                    # 使用重排模型进行预测
                    rerank_scores = rerank_model.predict(pairs)
                    # 根据分数排序
                    top_indices = np.argsort(rerank_scores)[-min(k, len(unique_passages)):][::-1]
                    
                    # 获取重排后的结果
                    reranked_results = [unique_passages[i] for i in top_indices if i < len(unique_passages)]
                    reranked_scores = [rerank_scores[i] for i in top_indices if i < len(unique_passages)]
                    
                    print(f"重排后结果数量: {len(reranked_results)}")
                    print(f"重排问题: {question}")
                    print(f"重排后结果: ")
                    for doc in reranked_results:
                        print(f"doc[:10]: {doc[:10]}")
                    return reranked_results, reranked_scores
                except Exception as e:
                    print(f"重排过程出错: {str(e)}")
                    # 如果重排失败，返回混合检索结果
                    return unique_passages, [1.0] * len(unique_passages)
            
            # 11. 清理资源
            if torch.cuda.is_available():
                del index
                del res
            
            # 如果没有重排或者重排失败，返回混合检索结果
            return unique_passages, [1.0] * len(unique_passages)
            
        except Exception as e:
            import traceback
            print(f"检索过程出错: {str(e)}")
            print(traceback.format_exc())  # 打印详细的堆栈跟踪
            return [], []
            
    return retrieval_function

def init_components(use_gpu: bool = True):
    """初始化所有必要的组件"""
    print("正在初始化评估所需组件...")
    
    # 设置设备
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化LLM模型
    llm = init_deepseek_model(ollama_use=True)
    print("- LLM模型初始化完成")
    
    # 初始化编码模型
    embedding_model =  SentenceTransformer("models/all-MiniLM-L6-v2")
    print("- 编码模型初始化完成")
    
    # 初始化重排模型
    rerank_model = init_rerank_model()
    # if use_gpu and torch.cuda.is_available():
    #     rerank_model = rerank_model.to(device)
    print("- 重排模型初始化完成")
    
    return llm, embedding_model, rerank_model, device

def create_llm_function(llm):
    """创建LLM调用函数"""
    def llm_function(prompt: str) -> str:
        return llm(prompt)
    return llm_function

def create_rerank_function(rerank_model, device):
    """创建重排函数"""
    def rerank_function(documents: List[str], question: str) -> Tuple[List[str], List[float]]:
        try:
            # 如果文档列表为空，直接返回空结果
            if not documents:
                print("重排函数收到空文档列表")
                return [], []
                
            # 准备输入对
            print(f"重排问题: {question}")
            print(f"重排文档数量: {len(documents)}")
            pairs = [(question, doc) for doc in documents]
            
            # 使用重排模型进行预测
            scores = rerank_model.predict(pairs)
            
            # 检查分数长度是否与文档长度匹配
            if len(scores) != len(documents):
                print(f"警告: 重排分数数量({len(scores)})与文档数量({len(documents)})不匹配")
                # 如果长度不匹配，使用可用的分数或者默认分数
                if len(scores) > 0:
                    scores = scores[:len(documents)] + [0.0] * (len(documents) - len(scores))
                else:
                    scores = [1.0] * len(documents)
            
            # 根据分数排序，取前k个结果
            top_k_idx = np.argsort(scores)[-len(documents):][::-1]
            
            # 确保索引在有效范围内
            valid_indices = [i for i in top_k_idx if i < len(documents)]
            sorted_docs = [documents[i] for i in valid_indices]
            sorted_scores = [scores[i] for i in valid_indices]
            
            return sorted_docs, sorted_scores
        except Exception as e:
            print(f"重排过程出错: {str(e)}")
            return documents, [1.0] * len(documents)  # 发生错误时返回原始顺序
    return rerank_function

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查数据集文件是否存在
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {args.dataset_path}")
    
    try:
        # 1. 初始化组件
        print("开始初始化评估组件...")
        llm, embedding_model, rerank_model, device = init_components(use_gpu=args.use_gpu)
        
        # 2. 创建必要的函数
        print("创建评估所需函数...")
        llm_fn = create_llm_function(llm)
        retrieval_fn = create_temp_retrieval_function(
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            device=device,
            k=args.initial_k
        )
        rerank_fn = create_rerank_function(rerank_model, device)
        
        # 3. 配置评估参数
        config = {
            "rerank_fn": rerank_fn,
            "device": device,
            "initial_k": args.initial_k
        }
        
        # 4. 运行评估
        print(f"\n开始评估数据集: {args.dataset_path}")
        evaluate_rag(
            dataset_path=args.dataset_path,  # 直接传入数据集路径
            llm_fn=llm_fn,
            retrieval_fn=retrieval_fn,
            rerank_fn=rerank_fn,
            config=config
        )
        
        # # 5. 打印结果
        # print("\n评估结果:")
        # for metric, value in results.items():
        #     print(f"{metric}: {value:.4f}")

   
    except Exception as e:
        print(f"\n评估过程中出现错误: {str(e)}")
        raise
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n评估完成")

if __name__ == "__main__":
    main()

# python eval_main.py --dataset_path data/multidoc_qa_test.csv