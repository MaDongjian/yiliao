from flask import Blueprint, request, send_file, jsonify, Response
from flask_apispec import use_kwargs
from utils.rest_response import success_response
from qwen_ask import qwen_ask_with_sources
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.answer_formatter import AnswerFormatter
from src.flexible_llm import flexible_ask, get_flexible_rag
from qa_integration import ask

quest_blueprint = Blueprint('quest', __name__)
formatter = AnswerFormatter()

# 全局变量追踪模型加载状态
_model_loaded = False


@quest_blueprint.route("/test", methods=["GET"])
@use_kwargs({}, location='querystring')
def test_func():
    question2 = "空气净化管理规范"
    result2 = qwen_ask_with_sources(question2)
    print(f"问题: {question2}")
    print(f"答案: {result2['answer']}")
    print(f"\n参考来源: {len(result2['sources'])} 个")
    for i, source in enumerate(result2['sources'], 1):
        print(f"  [{i}] {source['filename']} (相似度: {source['similarity']:.2f})")
    return success_response(data="cg")


@quest_blueprint.route("/ask", methods=["POST"])
def ask_question():
    """
    问答接口 - 支持多种返回格式

    Request JSON:
    {
        "question": "医院医疗废物相关标准",
        "format": "table"  // "table" 或 "paragraph"
    }

    Response:
    {
        "code": 200,
        "data": {
            "question": "...",
            "format_type": "table",
            "columns": [...],
            "rows": [...]
        }
    }
    """
    data = request.get_json()
    question = data.get('question', '')
    format_type = data.get('format', 'table')  # 默认表格格式

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 获取RAG结果
    result = qwen_ask_with_sources(question)

    # 格式化输出
    if format_type == 'table':
        formatted = formatter.format_table(question, result)
    elif format_type == 'paragraph':
        formatted = formatter.format_paragraph(question, result)
    else:
        return success_response(data={'error': f'不支持的格式: {format_type}'}), 400

    return success_response(data=formatted)


@quest_blueprint.route("/ask/html", methods=["POST"])
def ask_question_html():
    """
    问答接口 - 返回HTML格式

    Request JSON:
    {
        "question": "医院医疗废物相关标准",
        "format": "table"  // "table" 或 "paragraph"
    }

    Response: HTML字符串
    """
    data = request.get_json()
    question = data.get('question', '')
    format_type = data.get('format', 'table')

    if not question:
        return "<p>请提供问题</p>", 400

    # 获取RAG结果
    result = qwen_ask_with_sources(question)

    # 生成HTML
    if format_type == 'table':
        html = formatter.format_html_table(question, result)
    elif format_type == 'paragraph':
        html = formatter.format_html_paragraph(question, result)
    else:
        return f"<p>不支持的格式: {format_type}</p>", 400

    return html


@quest_blueprint.route("/ask/table", methods=["GET", "POST"])
def ask_table():
    """
    问答接口 - 表格格式 (GET/POST)

    GET: /quest/ask/table?question=xxx
    POST: /quest/ask/table with JSON body
    """
    if request.method == 'GET':
        question = request.args.get('question', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    result = qwen_ask_with_sources(question)
    formatted = formatter.format_table(question, result)

    return success_response(data=formatted)


@quest_blueprint.route("/ask/paragraph", methods=["GET", "POST"])
def ask_paragraph():
    """
    问答接口 - 段落格式 (GET/POST)

    GET: /quest/ask/paragraph?question=xxx
    POST: /quest/ask/paragraph with JSON body
    """
    if request.method == 'GET':
        question = request.args.get('question', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    result = qwen_ask_with_sources(question)
    formatted = formatter.format_paragraph(question, result)

    return success_response(data=formatted)


@quest_blueprint.route("/ask/chat", methods=["GET", "POST"])
def ask_chat():
    """
    灵活对话接口 - 更自然的回答

    GET: /quest/ask/chat?question=xxx&temperature=0.8
    POST: /quest/ask/chat with JSON body

    Request JSON:
    {
        "question": "问题",
        "temperature": 0.8,  // 可选，0.1-1.0，越高越灵活
        "stream": false       // 可选，是否流式输出
    }

    Response:
    {
        "question": "...",
        "answer": "...",
        "sources": [...]
    }
    """
    if request.method == 'GET':
        question = request.args.get('question', '')
        temperature = float(request.args.get('temperature', 0.8))
    else:
        data = request.get_json() or {}
        question = data.get('question', '')
        temperature = float(data.get('temperature', 0.8))

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    result = flexible_ask(question, stream=False, temperature=temperature)

    return success_response(data=result)


@quest_blueprint.route("/ask/stream", methods=["GET", "POST"])
def ask_stream():
    """
    流式对话接口 - 像DeepSeek一样逐字输出

    GET: /quest/ask/stream?question=xxx
    POST: /quest/ask/stream with JSON body

    返回Server-Sent Events (SSE)格式
    """
    if request.method == 'GET':
        question = request.args.get('question', '')
        temperature = float(request.args.get('temperature', 0.8))
    else:
        data = request.get_json() or {}
        question = data.get('question', '')
        temperature = float(data.get('temperature', 0.8))

    if not question:
        return "请提供问题", 400

    def generate():
        """生成SSE流"""
        try:
            for chunk in flexible_ask(question, stream=True, temperature=temperature):
                # SSE格式: data: <内容>\n\n
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [错误] {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')


@quest_blueprint.route("/ask/flexible", methods=["POST"])
def ask_flexible():
    """
    最灵活的对话接口 - 支持多种参数

    Request JSON:
    {
        "question": "问题",
        "temperature": 0.8,     // 随机性，0.1-1.0
        "top_k": 5,              // 检索文档数量
        "format": "table"        // 可选，返回格式化数据
    }
    """
    data = request.get_json() or {}
    question = data.get('question', '')
    temperature = float(data.get('temperature', 0.8))
    format_type = data.get('format', None)

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 获取答案
    result = flexible_ask(question, stream=False, temperature=temperature)

    # 如果需要格式化
    if format_type == 'table':
        formatted = formatter.format_table(question, result)
        return success_response(data=formatted)
    elif format_type == 'paragraph':
        formatted = formatter.format_paragraph(question, result)
        return success_response(data=formatted)
    else:
        return success_response(data=result)


# ============================================================
# 简化版 API 接口 - 直接使用 qa_integration.ask()
# ============================================================

@quest_blueprint.route("/api/ask", methods=["POST", "GET"])
def api_ask():
    """
    简化的问答接口 - 支持 GET 和 POST

    GET: /quest/api/ask?question=xxx
    POST: /quest/api/ask with JSON body {"question": "xxx"}

    Response:
    {
        "code": 200,
        "data": {
            "question": "...",
            "answer": "...",
            "sources": [...],
            "success": true
        }
    }
    """
    global _model_loaded

    # 获取问题
    if request.method == 'GET':
        question = request.args.get('question', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    try:
        result = ask(question)
        _model_loaded = True

        return success_response(data={
            'question': result['question'],
            'answer': result['answer'],
            'sources': result['sources'],
            'success': result['success']
        })
    except Exception as e:
        return success_response(data={'error': str(e)}), 500


@quest_blueprint.route("/api/health", methods=["GET"])
def api_health():
    """
    健康检查接口
    """
    return success_response(data={
        'status': 'ok',
        'model_loaded': _model_loaded,
        'service': 'RAG Q&A System'
    })


@quest_blueprint.route("/api/ask/html", methods=["POST", "GET"])
def api_ask_html():
    """
    返回 HTML 格式的问答 - 包含高亮样式

    GET: /quest/api/ask/html?question=xxx
    POST: /quest/api/ask/html with JSON body

    Response: HTML 字符串（带样式）
    """
    global _model_loaded

    if request.method == 'GET':
        question = request.args.get('question', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')

    if not question:
        return "<p>请提供问题</p>", 400

    try:
        result = ask(question)
        _model_loaded = True

        # 将 Markdown 转换为带样式的 HTML
        answer = result['answer']
        html = _markdown_to_html(answer, result['sources'])

        return html
    except Exception as e:
        return f"<p>错误: {str(e)}</p>", 500


def _markdown_to_html(answer, sources):
    """将 Markdown 答案转换为带样式的 HTML"""
    import re

    html = f'<div class="rag-answer">'

    # 处理表格
    lines = answer.split('\n')
    in_table = False
    table_rows = []

    for line in lines:
        # 表格行
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                html += '<table class="rag-table"><thead>'
            else:
                html += '<tr>'

            cells = [c.strip() for c in line.split('|')[1:-1]]
            for cell in cells:
                if cell == '---' or cell.startswith('---'):
                    html += '</thead><tbody>'
                else:
                    tag = 'th' if not in_table or '---' in line else 'td'
                    html += f'<{tag}>{cell}</{tag}>'

            if not ('---' in line):
                html += '</tr>'
        elif line.strip() == '':
            if in_table:
                html += '</tbody></table>'
                in_table = False
            html += '<br>'
        else:
            if in_table:
                html += '</tbody></table>'
                in_table = False

            # 处理粗体 **text**
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)

            # 处理列表
            if line.strip().startswith('- '):
                line = f'<li>{line.strip()[2:]}</li>'
                if not html.endswith('<ul>'):
                    html += '<ul>'
            elif line.strip().startswith('## '):
                line = f'<h3>{line.strip()[3:]}</h3>'
            elif line.strip().startswith('# '):
                line = f'<h2>{line.strip()[2:]}</h2>'
            else:
                line = f'<p>{line}</p>'

            html += line

    if in_table:
        html += '</tbody></table>'
    if html.endswith('<ul>'):
        html += '</ul>'

    # 添加来源
    if sources:
        html += '<div class="rag-sources"><h4>参考来源：</h4><ul>'
        for s in sources[:5]:
            html += f'<li>{s.get("filename", "未知")} (相似度: {s.get("similarity", 0):.2f})</li>'
        html += '</ul></div>'

    html += '</div>'

    # 添加样式
    styles = """
    <style>
    .rag-answer { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .rag-answer strong { color: #e74c3c; font-weight: bold; background: #fff3f3; padding: 2px 4px; border-radius: 3px; }
    .rag-answer table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    .rag-answer th { background: #3498db; color: white; padding: 10px; text-align: left; }
    .rag-answer td { border: 1px solid #ddd; padding: 8px; }
    .rag-answer tr:nth-child(even) { background: #f8f9fa; }
    .rag-answer h2, .rag-answer h3 { color: #2c3e50; margin-top: 15px; }
    .rag-answer ul { padding-left: 20px; }
    .rag-answer li { margin: 5px 0; }
    .rag-sources { margin-top: 20px; padding: 10px; background: #f0f8ff; border-left: 3px solid #3498db; }
    .rag-sources h4 { margin: 0 0 10px 0; color: #3498db; }
    </style>
    """

    return styles + html


@quest_blueprint.route("/api/ask/stream", methods=["POST", "GET"])
def api_ask_stream():
    """
    流式问答接口 - SSE 格式

    GET: /quest/api/ask/stream?question=xxx
    POST: /quest/api/ask/stream with JSON body
    """
    def generate():
        global _model_loaded

        if request.method == 'GET':
            question = request.args.get('question', '')
        else:
            data = request.get_json() or {}
            question = data.get('question', '')

        if not question:
            yield f"data: {{'error': '请提供问题'}}\n\n"
            return

        try:
            yield f"data: 正在思考: {question}\n\n"
            result = ask(question)
            _model_loaded = True

            # 分块发送答案
            answer = result['answer']
            chunk_size = 100
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"data: {chunk}\n\n"

            # 发送来源
            for source in result['sources']:
                yield f"data: 来源: {source['filename']} (相似度: {source['similarity']:.2f})\n\n"

        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')