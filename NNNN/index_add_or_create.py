from processor.document_processor import DocumentProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from model_use.embedding_model import init_embedding_model
from model_use.generate_model import init_deepseek_model
from sentence_transformers import SentenceTransformer
import importlib
import processor.main_file_chunk
importlib.reload(processor.main_file_chunk)
from processor.main_file_chunk import main_file_chunk, txt_results_to_documents,txt_results_tokenize_documents
import retrieval_strategy.faiss_module
importlib.reload(retrieval_strategy.faiss_module)
from retrieval_strategy.faiss_module import build_faiss_index, load_faiss_index, search_documents,add_documents
import retrieval_strategy.retrieval_general_method
importlib.reload(retrieval_strategy.retrieval_general_method)
from retrieval_strategy.retrieval_general_method import BM25Index

def main(new_index=True):
    """如果new_index为True，则从“files”文件夹中构建新的索引以及全新的向量数据库
    如果new_index为False，则从“new_files”文件夹中动态添加新的文档到向量数据库中"""
    if new_index:
        # faiss构建索引
        txt_results = main_file_chunk()
        documents, metadatas = txt_results_to_documents(txt_results)
        index = build_faiss_index(documents)
        # bm25构建索引
        bm25_index = BM25Index()
        documents,tokenized_documents = txt_results_tokenize_documents(txt_results)
        bm25_index.build_from_txt_results(documents,tokenized_documents)
    else:
        # faiss动态添加
        index, doc_store = load_faiss_index()
        txt_results = main_file_chunk(directory_path = "new_files")
        documents, metadatas = txt_results_to_documents(txt_results)
        add_documents(documents, index)
        # bm25动态添加
        documents,tokenized_documents = txt_results_tokenize_documents(txt_results)
        bm25_index = BM25Index()
        bm25_index.add_documents(documents,tokenized_documents)

if __name__ == "__main__":
    main(new_index=True)





