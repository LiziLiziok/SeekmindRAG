from typing import Callable, Dict, Any, List, Tuple, Union
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from rouge_chinese import Rouge
import jieba

class BaseEvaluator:
    def __init__(self, llm_fn: Callable, retrieval_fn: Callable, config: Dict[str, Any]):
        """
        初始化评估器
        Args:
            llm_fn: LLM接口函数，输入问题返回答案
            retrieval_fn: 检索接口函数，输入文档列表和问题，返回相关文档和分数
            config: 配置信息
        """
        self.llm_fn = llm_fn
        self.retrieval_fn = retrieval_fn
        self.config = config
        self.rouge = Rouge()
        
    def format_prompt(self, question: str, context: str) -> str:
        """格式化输入提示"""
        prompt_question = (
        f"你是一个图书管理员，请参考以下书本语句，简洁直接地回答提问者的问题，不要输出思维链：【{context}】。\n\n"
        f"提问者：{question}\n"
        "图书管理员："
        )
        return prompt_question
        
    def evaluate_single(self, question: str, ground_truths: List[str], 
                       documents: List[str], positive_references: List[str] = None, k: int = None, **kwargs) -> Dict[str, Any]:
        """评估单个问题"""
        # 1. 检索相关文档
        retrieved_docs, retrieval_scores = self.retrieval_fn(documents, question)
        
        # 2. 重排文档（如果配置了重排函数）
        if "rerank_fn" in self.config and self.config["rerank_fn"]:
            reranked_docs, rerank_scores = self.config["rerank_fn"](retrieved_docs, question)
            final_docs = reranked_docs[:k] if k is not None else reranked_docs
            final_scores = rerank_scores[:k] if k is not None else rerank_scores
        else:
            final_docs = retrieved_docs[:k] if k is not None else retrieved_docs
            final_scores = retrieval_scores[:k] if k is not None else retrieval_scores
            
        # 3. 构造提示
        context = "\n".join(final_docs)
        prompt = self.format_prompt(question, context)
        
        # 4. 生成答案
        prediction = self.llm_fn(prompt)
        
        # 5. 计算指标
        metrics = {
            "em": self._exact_match(prediction, ground_truths),
            "f1": self._f1_score(prediction, ground_truths),
            "rouge": self._rouge_score(prediction, ground_truths),
            "retrieval_precision": self._calculate_retrieval_precision(final_docs, positive_references if positive_references else documents),
            "retrieval_recall": self._calculate_retrieval_recall(final_docs, positive_references if positive_references else documents),
            "ndcg": self._calculate_ndcg(final_scores, documents)
        }
        
        return {
            "question": question,
            "prediction": prediction,
            "ground_truths": ground_truths,
            "retrieved_docs": final_docs,
            "retrieval_scores": final_scores,
            "metrics": metrics,
            "k": k
        }
    
    def _calculate_ndcg(self, scores: List[float], ground_truth_docs: List[str]) -> float:
        """计算NDCG分数"""
        if not scores or not ground_truth_docs:
            return 0.0
        try:
            relevance = np.zeros(len(scores))
            for i, score in enumerate(scores):
                if i < len(ground_truth_docs):
                    relevance[i] = 1.0
            return ndcg_score([relevance], [scores])
        except:
            return 0.0
    
    def _calculate_retrieval_precision(self, retrieved_docs: List[str], reference_docs: List[str]) -> float:
        """计算检索精确率"""
        if not retrieved_docs:
            return 0.0
        retrieved_set = set(retrieved_docs)
        reference_set = set(reference_docs)
        return len(retrieved_set & reference_set) / len(retrieved_docs)
    
    def _calculate_retrieval_recall(self, retrieved_docs: List[str], reference_docs: List[str]) -> float:
        """计算检索召回率"""
        if not reference_docs:
            return 0.0
        retrieved_set = set(retrieved_docs)
        reference_set = set(reference_docs)
        return len(retrieved_set & reference_set) / len(reference_docs)
        
    def _exact_match(self, prediction: Union[str, List[str]], ground_truths: List[str]) -> float:
        """计算精确匹配分数"""
        if isinstance(prediction, list):
            # 如果预测结果是列表，检查是否与ground_truths中的任一元素精确匹配
            sorted_prediction = sorted(prediction)
            sorted_ground_truths = sorted(ground_truths)
            if sorted_prediction == sorted_ground_truths:
                return 1.0
            else:
                return 0.0       
        else:
            # 如果预测结果是字符串
            for truth in ground_truths:
                if prediction.strip() == truth.strip():
                    return 1.0
            return 0.0
        
    def _f1_score(self, prediction: Union[str, List[str]], ground_truths: List[str]) -> float:
        """计算F1分数"""
        if isinstance(prediction, list):
            # 如果预测结果是列表，计算每个预测与每个ground_truth的F1，取最高值
            best_f1 = 0.0
            for pred in prediction:
                for truth in ground_truths:
                    f1 = self._calculate_f1_for_strings(pred, truth)
                    best_f1 = max(best_f1, f1)
            return best_f1
        else:
            # 如果预测结果是字符串，计算与每个ground_truth的F1，取最高值
            best_f1 = 0.0
            for truth in ground_truths:
                f1 = self._calculate_f1_for_strings(prediction, truth)
                best_f1 = max(best_f1, f1)
            return best_f1
    
    def _calculate_f1_for_strings(self, pred: str, truth: str) -> float:
        """计算两个字符串之间的F1分数"""
        pred_tokens = list(jieba.cut(pred.strip()))
        truth_tokens = list(jieba.cut(truth.strip()))
        
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        common_tokens = set(pred_tokens) & set(truth_tokens)
        if not common_tokens:
            return 0.0
            
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
        
    def _rouge_score(self, prediction: Union[str, List[str]], ground_truths: List[str]) -> float:
        """计算ROUGE分数"""
        try:
            if isinstance(prediction, list):
                # 如果预测结果是列表，计算每个预测与每个ground_truth的ROUGE，取最高值
                best_rouge = 0.0
                for pred in prediction:
                    for truth in ground_truths:
                        scores = self.rouge.get_scores(pred, truth)
                        rouge_score = scores[0]["rouge-l"]["f"]
                        best_rouge = max(best_rouge, rouge_score)
                return best_rouge
            else:
                # 如果预测结果是字符串，计算与每个ground_truth的ROUGE，取最高值
                best_rouge = 0.0
                for truth in ground_truths:
                    scores = self.rouge.get_scores(prediction, truth)
                    rouge_score = scores[0]["rouge-l"]["f"]
                    best_rouge = max(best_rouge, rouge_score)
                return best_rouge
        except:
            return 0.0
        
    def evaluate_batch(self, eval_data: List[Dict]) -> Dict[str, Any]:
        """评估一批数据"""
        results = {
            "em": [],
            "f1": [],
            "rouge": [],
            "retrieval_precision": [],
            "retrieval_recall": [],
            "ndcg": []
        }
        
        for item in eval_data:
            # 1. 检索相关文档
            retrieved_docs, retrieval_scores = self.retrieval_fn(item["referred_docs"], item["question"])
            
            # 2. 计算检索指标
            retrieved_set = set(retrieved_docs)
            positive_set = set(item["positive_references"])
            
            # 计算检索精确率和召回率
            if len(retrieved_set) > 0:
                precision = len(retrieved_set & positive_set) / len(retrieved_set)
                recall = len(retrieved_set & positive_set) / len(positive_set)
            else:
                precision = 0
                recall = 0
                
            results["retrieval_precision"].append(precision)
            results["retrieval_recall"].append(recall)
            
            # 3. 生成答案
            prediction = self.llm_fn(item["question"], retrieved_docs)
            
            # 4. 计算答案评估指标
            em = self._exact_match(prediction, item["answer"] if isinstance(item["answer"], list) else [item["answer"]])
            f1 = self._f1_score(prediction, item["answer"] if isinstance(item["answer"], list) else [item["answer"]])
            rouge = self._rouge_score(prediction, item["answer"] if isinstance(item["answer"], list) else [item["answer"]])
            
            results["em"].append(em)
            results["f1"].append(f1)
            results["rouge"].append(rouge)
            
            # 5. 计算NDCG
            ndcg = self._calculate_ndcg(retrieval_scores, item["referred_docs"])
            results["ndcg"].append(ndcg)
            
        # 计算平均指标
        final_results = {
            "em": np.mean(results["em"]),
            "f1": np.mean(results["f1"]),
            "rouge": np.mean(results["rouge"]),
            "retrieval_precision": np.mean(results["retrieval_precision"]),
            "retrieval_recall": np.mean(results["retrieval_recall"]),
            "ndcg": np.mean(results["ndcg"])
        }
        
        return final_results 