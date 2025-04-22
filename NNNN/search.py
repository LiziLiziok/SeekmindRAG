
import time
import importlib
import QAsystems.QA
importlib.reload(QAsystems.QA)
from QAsystems.QA import QASystem



start_time = time.time()
qa = QASystem(use_gpu=True, ollama_use=True)
end_time = time.time()
print(f"初始化时间：{end_time - start_time:.4f} 秒")

questions = ["计算机的重要性是什么", 
             "世界的格局是什么", 
             "中国的未来如何发展",
             "中华人民共和国反间谍法第第六十四条是什么",
             "明知是间谍行为的涉案财物而窝藏、转移、收购、代为销售或者以其他方法掩饰、隐瞒的会受到什么处罚",
             "中华人民共和国反间谍法几月几日开始执行",
             ]
time_list = []
for i in range(len(questions)):
    # 第一次提问
    print("=== 第{}次提问 ===".format(i+1))
    start_time = time.time()
    query1 = questions[i]
    reranked_scores, reranked_results, prompt_question, answer = qa.ask(query1, k=5)
    for j in range(len(reranked_results)):
        print(f"第{j+1}个检索结果：{reranked_results[j]}")
        print(f"第{j+1}个检索结果得分：{reranked_scores[j]}")
    print("-"*100)
    end_time = time.time()
    print(answer)
    print(f"第{i+1}次回答时间：{end_time - start_time:.4f} 秒")
    time_list.append(end_time - start_time)

print(time_list)    
print(sum(time_list)/len(time_list))



