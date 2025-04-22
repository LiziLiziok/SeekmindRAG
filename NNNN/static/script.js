document.addEventListener('DOMContentLoaded', function() {
    // 获取元素
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const clearButton = document.getElementById('clear-btn');
    const modelSelect = document.getElementById('model-select');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // 初始化
    loadChatHistory();
    
    // 事件监听器
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    clearButton.addEventListener('click', clearChatHistory);
    
    // 获取对话历史
    function loadChatHistory() {
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                chatMessages.innerHTML = '<div class="system-message"><div class="message-content"><p>欢迎使用基于 DeepSeek 的多文档问答系统！请输入您的问题。</p></div></div>';
                
                if (data.history && data.history.length > 0) {
                    data.history.forEach(item => {
                        displayUserMessage(item.query);
                        displayAssistantMessage(item.response);
                    });
                    
                    // 滚动到底部
                    scrollToBottom();
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
            });
    }
    
    // 发送消息
    function handleSendMessage() {
        const query = userInput.value.trim();
        if (!query) return;
        
        // 显示用户消息
        displayUserMessage(query);
        
        // 清空输入框并滚动到底部
        userInput.value = '';
        scrollToBottom();
        
        // 显示加载动画
        loadingOverlay.classList.add('show');
        
        // 禁用发送按钮
        sendButton.disabled = true;
        userInput.disabled = true;
        
        // 调用后端API
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                model: modelSelect.value
            }),
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载动画
            loadingOverlay.classList.remove('show');
            
            // 显示助手回复
            displayAssistantMessage(data);
            
            // 恢复发送按钮
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
            
            // 滚动到底部
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.classList.remove('show');
            
            // 显示错误消息
            const errorMsg = {
                answer: "抱歉，处理您的请求时遇到了问题。请再试一次。",
                sources: [],
                source_scores: [],
                response_time: "N/A"
            };
            displayAssistantMessage(errorMsg);
            
            // 恢复发送按钮
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        });
    }
    
    // 显示用户消息
    function displayUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        
        const messageHeader = document.createElement('div');
        messageHeader.className = 'message-header';
        
        const messageSender = document.createElement('div');
        messageSender.className = 'message-sender';
        messageSender.textContent = '用户';
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = formatTime(new Date());
        
        messageHeader.appendChild(messageSender);
        messageHeader.appendChild(messageTime);
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = `<p>${escapeHTML(message)}</p>`;
        
        messageElement.appendChild(messageHeader);
        messageElement.appendChild(messageContent);
        
        chatMessages.appendChild(messageElement);
    }
    
    // 显示助手消息
    function displayAssistantMessage(data) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message assistant-message';
        
        const messageHeader = document.createElement('div');
        messageHeader.className = 'message-header';
        
        const messageSender = document.createElement('div');
        messageSender.className = 'message-sender';
        messageSender.textContent = '助手';
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = data.response_time ? `${data.response_time}` : formatTime(new Date());
        
        messageHeader.appendChild(messageSender);
        messageHeader.appendChild(messageTime);
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // 处理Markdown
        messageContent.innerHTML = marked.parse(data.answer);
        
        // 添加来源信息
        if (data.sources && data.sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            
            const sourcesToggle = document.createElement('div');
            sourcesToggle.className = 'sources-toggle';
            sourcesToggle.innerHTML = '<i class="fas fa-book"></i> 显示参考内容';
            
            const sourcesContent = document.createElement('div');
            sourcesContent.className = 'sources-content';
            
            data.sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                let score = '';
                if (data.source_scores && data.source_scores[index]) {
                    score = `<div class="source-score">相关度: ${data.source_scores[index].toFixed(2)}</div>`;
                }
                
                sourceItem.innerHTML = `
                    <div>${escapeHTML(source.substring(0, 200))}${source.length > 200 ? '...' : ''}</div>
                    ${score}
                `;
                
                sourcesContent.appendChild(sourceItem);
            });
            
            // 添加切换功能
            sourcesToggle.addEventListener('click', function() {
                if (sourcesContent.classList.contains('show')) {
                    sourcesContent.classList.remove('show');
                    sourcesToggle.innerHTML = '<i class="fas fa-book"></i> 显示参考内容';
                } else {
                    sourcesContent.classList.add('show');
                    sourcesToggle.innerHTML = '<i class="fas fa-book-open"></i> 隐藏参考内容';
                }
            });
            
            sourcesContainer.appendChild(sourcesToggle);
            sourcesContainer.appendChild(sourcesContent);
            messageContent.appendChild(sourcesContainer);
        }
        
        messageElement.appendChild(messageHeader);
        messageElement.appendChild(messageContent);
        
        chatMessages.appendChild(messageElement);
    }
    
    // 清空聊天历史
    function clearChatHistory() {
        if (confirm('确定要清空对话历史吗？')) {
            fetch('/api/clear', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 重新加载对话历史
                    loadChatHistory();
                }
            })
            .catch(error => {
                console.error('Error clearing chat history:', error);
            });
        }
    }
    
    // 辅助函数：滚动到底部
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // 辅助函数：格式化时间
    function formatTime(date) {
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    // 辅助函数：HTML转义
    function escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}); 