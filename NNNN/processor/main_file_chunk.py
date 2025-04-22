from processor.document_processor import DocumentProcessor
import os
import shutil
def move_files(src_dir="new_files", dst_dir="files"):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历源目录下的所有文件
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 如果是文件，就移动
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")

def main_file_chunk(directory_path="files", chunk_size=512, overlap=52):
    """
    处理所有文件
    返回：txt_results：
    [
    {'filename':'aa.doc','extension': '.doc', 'content': '然而，中国的发展模式强调合作共赢，而不是零和博弈。', 'chunks': [['然而', '，', '中国', '的', '发展', '模式', '强调', '合作', '共赢', '，', '而不是', '零和', '博弈', '。']],'chunk_count': 1,'chunk_lengths': [20], 'average_chunk_length': 20},
    {'filename':'aa.doc','extension': '.doc', 'content': '然而，中国的发展模式强调合作共赢，而不是零和博弈。', 'chunks': [['然而', '，', '中国', '的', '发展', '模式', '强调', '合作', '共赢', '，', '而不是', '零和', '博弈', '。']],'chunk_count': 1,'chunk_lengths': [20], 'average_chunk_length': 20}
    ……]
    """
    # 1. 初始化文档处理器
    processor = DocumentProcessor(
        directory_path=directory_path, 
        chunk_size=chunk_size,   # 分块大小
        overlap=overlap         # 重叠大小
    )
    
    # 2. 获取目录中文件的统计信息
    stats = processor.get_file_stats()
    print("\n=== 文件统计信息 ===")
    print(f"总文件数: {stats['total_files']}")
    print("文件类型分布:")
    for ext, count in stats["extension_counts"].items():
        print(f"  {ext} ({processor.get_extension_type(ext)}): {count}文件")
    print(f"总文本长度: {stats['total_content_length']}字符")
    print(f"平均文件长度: {stats['average_file_length']:.2f}字符")
    
    # 3. 处理所有支持的文件
    print("\n=== 处理所有文件 ===")
    results = processor.process_directory()
    
    # 4. 显示处理结果
    for result in results:
        print(f"\n文件: {result['filename']}")
        print(f"类型: {processor.get_extension_type(result['extension'])}")
        print(f"内容长度: {result['content_length']}字符")
        
        if result.get("chunks"):
            print(f"分块数量: {result['chunk_count']}")
            print(f"平均分块长度: {result['average_chunk_length']:.2f}字符")
            
            # 打印前两个分块的内容示例
            print("\n分块内容示例:")
            for i, chunk in enumerate(result["chunks"][:2]):
                print(f"\n块 {i+1}:")
                print(''.join(chunk))
                if i >= 1:  # 只显示前两个块
                    print("...")
                    break
        else:
            print(f"分块失败: {result.get('chunk_error', '未知错误')}")

    # 5. 处理特定类型的文件
    print("\n=== 只处理特定类型文件 ===")
    txt_results = processor.process_directory(['.pdf', '.doc','.txt','.md','.docx','.csv','.json','.yaml','.yml'])  # 只处理这些类型的文件
    print(f"处理了 {len(txt_results)} 个文本文件")
    # print(txt_results)
    if directory_path == "new_files":
         move_files()

    return txt_results


def txt_results_to_documents(txt_results):
    """
    将txt_results转换为documents
    返回：
    texts：
    ['然而，中国的发展模式强调合作共赢，而不是零和博弈。', '然而，中国的发展模式强调合作共赢，而不是零和博弈。']
    metadatas：
    [{'filename': 'aa.doc', 'chunk_id': 0, 'extension': '.doc', 'chunk_length': 20},
    {'filename': 'aa.doc', 'chunk_id': 0, 'extension': '.doc', 'chunk_length': 20}]
    """
    results = txt_results
    texts = []
    metadatas = []

    for doc in results:
        filename = doc['filename']
        for i, tokens in enumerate(doc['chunks']):
            text = ''.join(tokens)  # 拼接成句子
            texts.append(text)
            metadatas.append({
                'filename': filename,
                'chunk_id': i,
                'extension': doc.get('extension'),
                'chunk_length': doc.get('chunk_lengths', [])[i] if i < len(doc.get('chunk_lengths', [])) else None,
            })
    return texts, metadatas

def txt_results_tokenize_documents(txt_results):
    documents = []
    tokenized_documents = []
    for i in txt_results:
        for j in i['chunks']:
            documents.append("".join(j))
            tokenized_documents.append(j)
    
    return documents,tokenized_documents

