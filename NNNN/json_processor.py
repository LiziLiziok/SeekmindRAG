import importlib
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import argparse
import os
import faiss
from sentence_transformers import SentenceTransformer
import hanlp
from tqdm import tqdm
import time
import pandas as pd

# 导入并重新加载必要的模块
import model_use.rerank_model
import model_use.generate_model
import model_use.embedding_model
importlib.reload(model_use.rerank_model)
importlib.reload(model_use.generate_model)
importlib.reload(model_use.embedding_model)

# 导入评估相关组件
from model_use.rerank_model import init_rerank_model
from model_use.generate_model import init_deepseek_model
from model_use.embedding_model import init_embedding_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理JSON数据集并生成结果')
    parser.add_argument('--input_path', type=str, required=True,
                      help='输入JSON数据集文件路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='输出JSON结果文件路径')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                      help='是否使用GPU')
    parser.add_argument('--initial_k', type=int, default=10,
                      help='初始检索的文档数量')
    parser.add_argument('--prompt_template', type=str, default="根据以下信息回答问题:\n\n文档: {context}\n\n问题: {question}\n\n回答:",
                      help='提示词模板')
    return parser.parse_args()

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
            
            # 7. 创建BM25检索（稀疏检索）
            # 这里我们简化实现，假设文档已经是小段落
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
    print("- 重排模型初始化完成")
    
    return llm, embedding_model, rerank_model, device

def create_llm_function(llm):
    """创建LLM调用函数"""
    def llm_function(prompt: str) -> str:
        return llm(prompt)
    return llm_function

def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """使用pandas加载JSON数据集"""
    try:
        # 检查文件扩展名
        if file_path.endswith('.jsonl'):
            # 对于JSONL文件，使用lines=True参数
            df = pd.read_json(file_path, lines=True)
        else:
            # 对于标准JSON文件
            df = pd.read_json(file_path)
            
        # 转换为字典列表
        data = df.to_dict(orient='records')
        print(f"成功加载数据集，包含 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        raise

def save_json_results(data: List[Dict[str, Any]], output_path: str):
    """使用pandas保存JSON结果到文件"""
    try:
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 检查文件扩展名
        if output_path.endswith('.jsonl'):
            # 对于JSONL文件，使用lines=True参数
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        else:
            # 对于标准JSON文件
            df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")
        raise

def process_dataset(data: List[Dict[str, Any]], 
                    llm_fn, 
                    retrieval_fn, 
                    prompt_template: str = "根据以下信息回答问题:\n\n文档: {context}\n\n问题: {question}\n\n回答:") -> List[Dict[str, Any]]:
    """处理数据集并生成结果"""
    results = []
    
    # 使用tqdm显示进度
    for item in tqdm(data, desc="处理数据"):
        try:
            # 获取问题和文档
            question = item.get("question", "")
            documents = item.get("context", [])
            
            if not question or not documents:
                print(f"警告: 跳过无效数据项 - 问题: {bool(question)}, 文档: {len(documents) if documents else 0}")
                # 添加空结果
                item_result = item.copy()
                item_result["llm_response"] = ""
                item_result["retrieved_documents"] = []
                results.append(item_result)
                continue
            
            # 1. 检索相关文档
            print(f"\n处理问题: {question}")
            retrieved_docs, scores = retrieval_fn(documents, question)
            
            # 2. 合并检索到的文档
            context = "\n".join([f"文档 {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
            
            # 3. 准备提示词
            prompt = prompt_template.format(context=context, question=question)
            
            # 4. 获取LLM响应
            llm_response = llm_fn(prompt)
            
            # 5. 构建结果
            item_result = item.copy()
            item_result["llm_response"] = llm_response
            item_result["retrieved_documents"] = [
                {"content": doc, "score": float(score)} 
                for doc, score in zip(retrieved_docs, scores)
            ]
            
            results.append(item_result)
            
            # 简单的限流，防止API调用过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"处理数据项时出错: {str(e)}")
            # 添加错误信息
            item_result = item.copy()
            item_result["error"] = str(e)
            item_result["llm_response"] = ""
            item_result["retrieved_documents"] = []
            results.append(item_result)
    
    return results

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"找不到输入文件: {args.input_path}")
    
    try:
        # 1. 初始化组件
        print("开始初始化组件...")
        llm, embedding_model, rerank_model, device = init_components(use_gpu=args.use_gpu)
        
        # 2. 创建必要的函数
        print("创建所需函数...")
        llm_fn = create_llm_function(llm)
        retrieval_fn = create_temp_retrieval_function(
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            device=device,
            k=args.initial_k
        )
        
        # 3. 加载数据集
        print(f"加载数据集: {args.input_path}")
        dataset = load_json_dataset(args.input_path)
        
        # 4. 处理数据集
        print("开始处理数据集...")
        results = process_dataset(
            data=dataset,
            llm_fn=llm_fn,
            retrieval_fn=retrieval_fn,
            prompt_template=args.prompt_template
        )
        
        # 5. 保存结果
        print("保存处理结果...")
        save_json_results(results, args.output_path)
        
        print(f"处理完成! 结果已保存到: {args.output_path}")
            
    except Exception as e:
        print(f"\n处理过程中出现错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n处理完成")

if __name__ == "__main__":
    main()

# 使用示例:
# python json_processor.py --input_path data/multidoc_qa_2.jsonl --output_path data/output.json --initial_k 5 