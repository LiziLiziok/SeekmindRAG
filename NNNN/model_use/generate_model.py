from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from langchain.chains import RetrievalQA
import requests
# Ollama封装函数
def ollama_wrapper(prompt, model_name="erwan2/DeepSeek-R1-Distill-Qwen-1.5B"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
    )
    print("模型加载完成，当前显存占用：")
    print(f"GPU 显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    return response.json()["response"]


# 初始化模型（支持Ollama或transformers）
def init_deepseek_model(ollama_use=False):
    """
    初始化 DeepSeek 模型
    - 如果 ollama_use=True：使用 Ollama 本地模型
    - 否则：使用 transformers 加载的量化模型
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    if ollama_use:
        print("使用 Ollama 本地模型: DeepSeek-R1-Distill-Qwen-1.5B")
        return ollama_wrapper

    print("使用 Transformers 量化模型: DeepSeek-R1-Distill-Qwen-7B")
    model_name = "models/DeepSeek-R1-Distill-Qwen-7B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        torch_dtype=torch.float16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        },
        low_cpu_mem_usage=True,
        offload_folder="offload"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        device_map="auto",
        batch_size=1,
        use_cache=True
    )

    print("模型加载完成，当前显存占用：")
    print(f"GPU 显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    return HuggingFacePipeline(pipeline=pipe)



if __name__ == "__main__":
    llm = init_deepseek_model()
    # 测试问题
    test_questions = [
        "请简单介绍一下中国式现代化的特点",
        "什么是人类命运共同体？",
    ]

    for question in test_questions:
        print(f"\n问题：{question}")
        response = llm(question)
        print(f"回答：{response}")
        print(f"当前显存占用：{torch.cuda.memory_allocated()/1024**2:.2f} MB")
        

