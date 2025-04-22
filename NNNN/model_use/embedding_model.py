import torch
from langchain_huggingface import HuggingFaceEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"
# if device == "cuda":
#     print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 初始化嵌入模型
def init_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceEmbeddings(
            model_name="models/all-MiniLM-L6-v2",  # 使用本地模型路径
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    return embedding
if __name__ == "__main__":
    embedding = init_embedding_model()
    print(embedding)