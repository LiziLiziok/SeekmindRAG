# 基于 DeepSeek 的多文档问答与评估系统

这是一个基于大语言模型的多文档问答与评估系统，支持多种文档格式处理、向量检索和评估方法。系统使用混合检索策略，结合了稠密向量检索和稀疏检索，通过重排序技术提升检索质量。

## 核心功能

### 1. 多文档问答处理
- **文档检索**：支持多种检索策略（稠密向量检索+稀疏检索）
- **上下文重组**：智能合并检索结果为上下文
- **LLM生成**：基于检索结果生成高质量答案
- **批量处理**：支持JSON/CSV数据集的批量处理

### 2. 检索技术
- **稠密向量检索**：使用FAISS向量数据库
  - 支持多种索引类型：IndexFlatL2、IndexFlatIP、IndexHNSWFlat
  - GPU加速支持
- **稀疏检索**：基于BM25算法
  - 使用HanLP进行中文分词
- **重排序**：使用交叉编码器对检索结果进行精确重排

### 3. 评估功能
- **检索评估**：精确率、召回率、F1和NDCG
- **回答评估**：精确匹配、F1、ROUGE和BLEU

## 技术架构

### 1. 模型组件
- **文本嵌入模型**：`all-MiniLM-L6-v2`
  - 用于文档和问题的向量表示
  - 支持中文文本向量化
- **重排序模型**：`cross-encoder/ms-marco-MiniLM-L6-v2`
  - 用于检索结果精确排序
- **生成模型**：
  - `DeepSeek-R1-Distill-Qwen-7B`（默认）
  - 支持Ollama本地部署模式

### 2. 检索流程
1. **双路检索**：
   - 稠密检索：使用向量相似度
   - 稀疏检索：使用BM25算法
2. **结果合并**：去重并合并两种检索结果
3. **重排序**：使用交叉编码器重新排序
4. **上下文构建**：将前K个结果合并为上下文

### 3. 问答流程
1. **问题处理**：分词、向量化
2. **文档检索**：获取相关文档
3. **提示词构建**：结合问题和上下文
4. **答案生成**：通过LLM生成最终答案

## 评估方法

### 1. 检索评估指标
- **精确率(Precision)**：检索结果中相关文档的比例
- **召回率(Recall)**：相关文档中被检索到的比例
- **F1分数**：精确率和召回率的调和平均
- **NDCG**：归一化折损累积增益，考虑排序质量

### 2. 回答评估指标
- **精确匹配(EM)**：预测答案与参考答案完全匹配的比例
- **F1分数**：基于分词的重叠度量
- **ROUGE-1/2/L**：评估生成文本与参考文本的重叠度
- **BLEU**：评估生成文本的翻译质量

## 技术实现

### 1. 文档处理
- 使用HanLP进行中文分词
- 支持多种文档格式处理

### 2. 向量检索
- FAISS索引类型：
  - IndexFlatL2：精确L2距离检索
  - IndexFlatIP：内积相似度（归一化后为余弦相似度）
  - IndexHNSWFlat：层次化可导航小世界图索引（大规模数据推荐）
- GPU加速：使用`faiss.index_cpu_to_gpu`将索引迁移至GPU

### 3. 混合检索策略
- 稠密检索：基于嵌入向量相似度
- 稀疏检索：基于BM25算法
- 结果去重：使用规范化文本进行去重

### 4. 重排序技术
- 使用CrossEncoder进行精确重排
- 对检索结果进行精确相关性评分

### 5. LLM集成
- 支持两种模式：
  - 本地Transformers模型（4bit量化）
  - Ollama API调用
- 可配置生成参数：温度、top_p等

## 依赖项

```
langchain>=0.1.0
langchain-community>=0.0.10
faiss-gpu>=1.7.2
torch>=2.0.0
transformers>=4.36.0
sentence-transformers>=2.2.2
hanlp>=2.1.0
python-docx>=0.8.11
PyPDF2>=3.0.0
numpy>=1.24.3
auto-gptq>=0.5.0
optimum>=1.16.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
```

## 使用方法

### 处理JSON数据集
```bash
python json_processor.py --input_path data/multidoc_qa.jsonl --output_path data/output.json --initial_k 5
```

### 评估系统性能
```bash 
python eval_main.py --dataset_path data/multidoc_qa_test.csv --initial_k 10
```

### 命令行参数
- `--input_path`：输入数据集路径
- `--output_path`：输出结果路径
- `--use_gpu`：是否使用GPU加速（默认：True）
- `--initial_k`：检索文档数量（默认：10）
- `--prompt_template`：提示词模板

## 硬件要求
- GPU：支持CUDA的显卡（建议8GB以上显存）
- CPU：建议8核心以上
- 内存：16GB以上

## 性能优化
- 4-bit量化：降低显存占用
- GPU加速：FAISS向量检索
- 批处理优化：提高处理效率

## 系统限制
- 显存要求：运行完整的7B模型需要至少8GB显存
- 检索限制：大规模文档集可能需要更多内存 