from typing import Callable, Dict, Any, List
import json
import os
import pandas as pd
from evaluators.base_evaluator import BaseEvaluator
import numpy as np


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    加载数据集
    Args:
        dataset_path: 数据集文件的完整路径
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    print(f"正在加载数据集: {dataset_path}")
    # 读取CSV文件
    df  = pd.read_json(dataset_path,lines=True)
    print(f"数据集加载完成，共 {len(df)} 条样本")
    
    # 转换为所需的格式
    dataset = []
    for _, row in df.iterrows():
        item = {
            "question": row["question"],
            "answer": row["answer"],
            "positive_references": row["positive_references"],
            "referred_docs": row["referred_docs"],
            "len_positive_references": row["len_positive_references"]
        }
        
        dataset.append(item)
    print(dataset[0])
    
    return dataset

def evaluate_rag(dataset_path: str, llm_fn: Callable, retrieval_fn: Callable, 
                rerank_fn: Callable, config: Dict[str, Any]) -> Dict[str, float]:
    """评估RAG系统"""
    # 加载数据集
    dataset = load_dataset(dataset_path)
    
    # 初始化评估器
    evaluator = BaseEvaluator(llm_fn, retrieval_fn, config)
    
    # 评估每个样本
    results = []
    res_pred = []
    total = len(dataset)
    
    print(f"\n开始评估 {total} 个样本...")
    for idx, item in enumerate(dataset):
        print(f"\r评估进度: {idx+1}/{total}", end="")
        
        # 获取当前样本需要的文本块数量
        k = item["len_positive_references"]
        print(f"当前样本需要的文本块数量k: {k}")
        print(f"当前样本的question: {item['question']}")
        
        # 确保positive_references是列表
        if not isinstance(item["positive_references"], list):
            item["positive_references"] = [item["positive_references"]]
            
        # 确保referred_docs是列表
        if not isinstance(item["referred_docs"], list):
            item["referred_docs"] = [item["referred_docs"]]
            
        print(f"当前样本的positive_references: ")
        for doc in item["positive_references"]:
            print(f"doc[:10]: {doc[:10]}")
        print(f"当前样本的referred_docs: ")
        for doc in item["referred_docs"]:
            print(f"doc[:10]: {doc[:10]}")
        try:
            # 1. 检索相关文档
            retrieved_docs, retrieval_scores = retrieval_fn(item["referred_docs"], item["question"])
            print(f"检索到的文档数量: {len(retrieved_docs)}")

            # 2. 重排文档
            if rerank_fn:
                reranked_docs, rerank_scores = rerank_fn(retrieved_docs, item["question"])
                # 确保最终使用的文档数量等于len_positive_references
                final_docs = reranked_docs[:k] if len(reranked_docs) >= k else reranked_docs
                final_scores = rerank_scores[:k] if len(rerank_scores) >= k else rerank_scores
            else:
                # 确保最终使用的文档数量等于len_positive_references
                final_docs = retrieved_docs[:k] if len(retrieved_docs) >= k else retrieved_docs
                final_scores = retrieval_scores[:k] if len(retrieval_scores) >= k else retrieval_scores
            
            print(f"重排后结果数量: {len(final_docs)}")
            print(f"重排后结果: ")
            for doc in final_docs:
                print(f"doc[:10]: {doc[:10]}")
            
            # 3. 生成答案
            context = "\n".join(final_docs)
            prompt = evaluator.format_prompt(item["question"], context)
            prediction = llm_fn(prompt)
            print(f"预测答案: {prediction}")

            # 4. 计算指标
            # metrics = evaluator.evaluate_single(
            #     question=item["question"],
            #     ground_truths=[item["answer"]],
            #     documents=item["referred_docs"],
            #     positive_references=item["positive_references"],
            #     k=k
            # )
            # results.append(metrics)
            dataset[idx]["prediction"] = prediction
            dataset[idx]["retrieved_docs"] = final_docs
            dataset[idx]["scores"] = final_scores
        except Exception as e:
            print(f"\n处理样本 {idx+1} 时出错: {str(e)}")
        # 将检索结果加入
        # results.append(final_docs) 
        # res_pred.append(prediction)
    df = pd.DataFrame(dataset)
    df.to_json('data/multidoc_qa_2_output_2.json', orient='records', lines=True)

    print("\n评估完成，计算最终指标...")
    return 
    
    # # 计算平均指标
    # final_metrics = {
    #     "em": np.mean([r["metrics"]["em"] for r in results]),
    #     "f1": np.mean([r["metrics"]["f1"] for r in results]),
    #     "rouge": np.mean([r["metrics"]["rouge"] for r in results]),
    #     "retrieval_precision": np.mean([r["metrics"]["retrieval_precision"] for r in results]),
    #     "retrieval_recall": np.mean([r["metrics"]["retrieval_recall"] for r in results]),
    #     "ndcg": np.mean([r["metrics"]["ndcg"] for r in results])
    # }
    
    # return final_metrics

def main():
    """示例用法"""
    # 模拟LLM接口
    def mock_llm(prompt: str) -> str:
        return "这是一个示例答案"
    
    # 模拟检索接口
    def mock_retrieval(documents: List[str], question: str) -> List[str]:
        return documents[:3]  # 返回前3个文档
    
    # 模拟重排接口
    def mock_rerank(documents: List[str], question: str) -> List[str]:
        return documents  # 不进行重排
    
    # 配置
    config = {
        "task": "multidoc_qa",
        "topk": 3,
        "rerank_fn": mock_rerank
    }
    
    # 运行评估
    results = evaluate_rag(
        dataset_path="data/multidoc_qa_test.csv",
        llm_fn=mock_llm,
        retrieval_fn=mock_retrieval,
        rerank_fn=mock_rerank,
        config=config
    )
    
    # 打印结果
    print("\n评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
