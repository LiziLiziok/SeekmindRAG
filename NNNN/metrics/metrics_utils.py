import string
import regex
import numpy as np
from collections import Counter
from rouge_chinese import Rouge
import jieba
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = Rouge()

def normalize_answer(s: str) -> str:
    """标准化答案字符串"""
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truths):
    """计算完全匹配分数"""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    normalized_prediction = normalize_answer(prediction)
    return max([float(normalize_answer(gt) in normalized_prediction) for gt in ground_truths])

def f1_score(prediction, ground_truths):
    """计算F1分数"""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    prediction_tokens = ' '.join(jieba.cut(normalize_answer(prediction)))
    f1_scores = []
    
    for ground_truth in ground_truths:
        ground_truth_tokens = ' '.join(jieba.cut(normalize_answer(ground_truth)))
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0)
            continue
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return max(f1_scores)

def rouge_score(prediction, ground_truths):
    """计算ROUGE分数"""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    scores = []
    prediction_tokens = ' '.join(jieba.cut(prediction))
    
    for ground_truth in ground_truths:
        ground_truth_tokens = ' '.join(jieba.cut(ground_truth))
        try:
            score = rouge.get_scores(prediction_tokens, ground_truth_tokens, avg=True)
            scores.append((
                score["rouge-1"]["f"],
                score["rouge-2"]["f"],
                score["rouge-l"]["f"]
            ))
        except:
            scores.append((0.0, 0.0, 0.0))
    
    # 返回最高分数
    return tuple(map(max, zip(*scores)))

def calculate_metrics(prediction, ground_truths, metrics=None):
    """计算所有指标"""
    if metrics is None:
        metrics = ["em", "f1", "rouge", "bleu"]
        
    results = {}
    
    if "em" in metrics:
        results["exact_match"] = exact_match_score(prediction, ground_truths)
    
    if "f1" in metrics:
        results["f1"] = f1_score(prediction, ground_truths)
    
    if "rouge" in metrics:
        rouge1, rouge2, rougel = rouge_score(prediction, ground_truths)
        results.update({
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougel": rougel
        })
    
    if "bleu" in metrics:
        results["bleu"] = bleu_score(prediction, ground_truths)
    
    return results 

def calculate_retrieval_metrics(retrieved_docs: List[str], ground_truth_docs: List[str], 
                              retrieval_scores: List[float] = None) -> Dict[str, float]:
    """计算检索相关指标"""
    if not retrieved_docs or not ground_truth_docs:
        return {
            "precision": 0.0, 
            "recall": 0.0, 
            "f1": 0.0,
            "ndcg": 0.0 if retrieval_scores else None
        }
    
    # 计算基本指标
    retrieved_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_docs)
    
    precision = len(retrieved_set & ground_truth_set) / len(retrieved_set)
    recall = len(retrieved_set & ground_truth_set) / len(ground_truth_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 如果提供了检索分数，计算NDCG
    ndcg = None
    if retrieval_scores:
        try:
            relevance = np.zeros(len(retrieval_scores))
            for i, doc in enumerate(retrieved_docs):
                if doc in ground_truth_set:
                    relevance[i] = 1.0
            ndcg = ndcg_score([relevance], [retrieval_scores])
        except:
            ndcg = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg
    }

def bleu_score(prediction, ground_truths):
    """计算BLEU分数"""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    # 使用jieba分词
    prediction_tokens = list(jieba.cut(prediction))
    smoothing = SmoothingFunction().method1
    
    bleu_scores = []
    for ground_truth in ground_truths:
        reference_tokens = list(jieba.cut(ground_truth))
        try:
            # 计算BLEU-4分数
            score = sentence_bleu([reference_tokens], prediction_tokens, 
                                 weights=(0.25, 0.25, 0.25, 0.25),
                                 smoothing_function=smoothing)
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
    
    return max(bleu_scores) if bleu_scores else 0.0

def calculate_answer_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """计算答案相关指标"""
    # 初始化Rouge
    rouge_calculator = Rouge()
    
    # 使用jieba进行中文分词
    pred_tokens = list(jieba.cut(prediction.strip()))
    truth_tokens = list(jieba.cut(ground_truth.strip()))
    
    # 计算精确匹配
    em = 1.0 if prediction.strip() == ground_truth.strip() else 0.0
    
    # 计算F1分数（基于分词）
    if not pred_tokens or not truth_tokens:
        f1 = 0.0
    else:
        common_tokens = set(pred_tokens) & set(truth_tokens)
        if not common_tokens:
            f1 = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(truth_tokens)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算ROUGE分数
    try:
        prediction_text = ' '.join(pred_tokens)
        ground_truth_text = ' '.join(truth_tokens)
        rouge_scores = rouge_calculator.get_scores(prediction_text, ground_truth_text)
        rouge1 = rouge_scores[0]["rouge-1"]["f"]
        rouge2 = rouge_scores[0]["rouge-2"]["f"]
        rouge_l = rouge_scores[0]["rouge-l"]["f"]
    except:
        rouge1, rouge2, rouge_l = 0.0, 0.0, 0.0
    
    # 计算BLEU分数
    try:
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([truth_tokens], pred_tokens, 
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing)
    except:
        bleu = 0.0
    
    return {
        "em": em,
        "f1": f1,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rouge_l,
        "bleu": bleu
    }

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """聚合多个样本的指标"""
    if not metrics_list:
        return {}
    
    aggregated = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        aggregated[metric] = np.mean(values)
    
    return aggregated 