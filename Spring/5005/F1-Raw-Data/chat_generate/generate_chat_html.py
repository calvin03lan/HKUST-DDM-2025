import json

def parse_chat(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogues = []
    
    for req in data.get('requests', []):
        # User message
        user_msg = req.get('message', {}).get('text', '')
        if user_msg:
            dialogues.append({'role': 'user', 'content': user_msg})
        
        # Assistant context (including thinking and text fragments)
        assistant_content = ""
        for resp_item in req.get('response', []):
            kind = resp_item.get('kind')
            
            # Extract thinking process (save previous plain text buffer first)
            if kind == 'thinking' and 'value' in resp_item and resp_item['value'].strip():
                if assistant_content.strip():
                    dialogues.append({'role': 'assistant', 'content': assistant_content.strip()})
                    assistant_content = ""
                dialogues.append({'role': 'thinking', 'content': resp_item['value'].strip()})
                
            # Handle inline file references in the chat
            elif kind == 'inlineReference' and 'name' in resp_item:
                assistant_content += f"`{resp_item.get('name')}`"

            # Merge text/code chunks
            elif not kind and 'value' in resp_item:
                assistant_content += resp_item['value']
        
        # Add remaining text after loop finishes
        if assistant_content.strip():
            dialogues.append({'role': 'assistant', 'content': assistant_content.strip()})
            
    return dialogues

def create_html(dialogues):
    # We pass the dialogues as a JSON object to Javascript
    # and use marked.js in the browser to perfectly render the Markdown
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 对话记录</title>
    
    <!-- 引入 marked.js 用于渲染 markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- 引入 github-markdown-css 美化 markdown 样式 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
            background-color: #f5f5f5; 
            padding: 20px; 
            margin: 0;
        }
        .chat-container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: #fff; 
            padding: 30px; 
            border-radius: 12px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
        }
        .message { 
            margin-bottom: 20px; 
            display: flex; 
            flex-direction: column; 
        }
        .user { align-items: flex-end; }
        .assistant, .thinking { align-items: flex-start; }
        
        .bubble { 
            max-width: 85%; 
            padding: 15px 20px; 
            border-radius: 18px; 
            line-height: 1.6; 
            word-wrap: break-word; 
            font-size: 15px;
        }
        
        /* User bubble base */
        .user .bubble { 
            background-color: #007aff !important; 
            color: white !important; 
            border-bottom-right-radius: 4px; 
        }
        
        /* Override markdown-body defaults for user bubbles so text inside paragraphs is also white */
        .user .bubble p, 
        .user .bubble li, 
        .user .bubble code, 
        .user .bubble strong, 
        .user .bubble span, 
        .user .bubble a,
        .user .bubble h1, .user .bubble h2, .user .bubble h3, .user .bubble h4, .user .bubble h5, .user .bubble h6 {
            color: white !important;
        }
        
        .user .bubble code {
            background-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        .assistant .bubble { 
            background-color: #f6f8fa; 
            border: 1px solid #d0d7de;
            color: #24292e; 
            border-bottom-left-radius: 4px; 
        }
        
        .thinking .bubble { 
            background-color: #fcfdfd; 
            color: #57606a; 
            border: 1px dashed #d0d7de; 
            border-bottom-left-radius: 4px; 
        }
        .thinking .bubble::before {
            content: '💡 思考过程：';
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #8c959f;
        }
        
        .label { 
            font-size: 12px; 
            color: #888; 
            margin-bottom: 5px; 
            margin-top: 10px; 
        }

        /* Prevent markdown-body from resetting background globally */
        .markdown-body {
            background-color: transparent !important;
            font-family: inherit !important;
            font-size: inherit !important;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2 style="text-align: center; color: #333; margin-bottom: 30px;">AI 对话记录</h2>
        <div id="chat-box"></div>
    </div>

    <script>
        // 设置 marked 支持换行和 GFM（Github Flavored Markdown）
        marked.setOptions({
            gfm: true,
            breaks: true
        });

        // 注入 Python 解析好的 JSON 数据
        const dialogues = DIALOGUE_DATA_PLACEHOLDER;
        const chatBox = document.getElementById('chat-box');
        
        dialogues.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            
            let roleName = "GitHub Copilot";
            if (msg.role === "user") roleName = "User";
            else if (msg.role === "thinking") roleName = "Copilot 思考中";
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'label';
            labelDiv.textContent = roleName;
            
            const bubbleDiv = document.createElement('div');
            // 添加 markdown-body 应用更好的排版（如代码块、表格等）
            bubbleDiv.className = 'bubble markdown-body';
            bubbleDiv.innerHTML = marked.parse(msg.content);
            
            messageDiv.appendChild(labelDiv);
            messageDiv.appendChild(bubbleDiv);
            chatBox.appendChild(messageDiv);
        });
    </script>
</body>
</html>"""
    
    # 替换占位符
    html_content = html_template.replace("DIALOGUE_DATA_PLACEHOLDER", json.dumps(dialogues, ensure_ascii=False))
    return html_content

if __name__ == "__main__":
    dialogues = parse_chat('chat.json')
    html_content = create_html(dialogues)
    
    html_file = 'chat_with_agent.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f" HTML file generated: {html_file}")

