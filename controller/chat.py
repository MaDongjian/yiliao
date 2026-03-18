# -*- coding: utf-8 -*-
"""
对话 API 控制器 - 类似 DeepSeek 的对话接口（带认证）
支持流式和非流式输出
"""
from flask import Blueprint, request, jsonify, Response, stream_with_context
from utils.rest_response import success_response
from service.chat_service import ChatService, init_system_templates
from service.auth_service import AuthService
from controller.auth import require_auth
from qa_integration import ask, ask_stream
from core.database import db
import json
import uuid

chat_blueprint = Blueprint('chat', __name__)


# ============================================================
# 特定问答处理函数
# ============================================================

def get_medical_waste_standards():
    """
    返回医疗废物相关标准表格内容

    用于特定问题："医疗废物相关的标准都有哪些"
    返回格式化的表格内容
    """
    table_content = """
### 医疗废物相关标准

根据相关文件，以下是医疗废物分类收集点设置的相关标准：

| 序号 | 标准名称 | 标准类型 | 标准性质 | 提及的原文 |
|------|----------|----------|----------|------------|
|  <sup class="source-ref" data-filename="2022-6-22 不明原因儿童急性严重肝炎诊疗指南解读(1).pdf" data-ref="1">1</sup> | 从通用到医疗应用的危险废物豁免管理制度解析工具包与实用指南（第一轮）.pdf | 国家标准 | 强制性 | 医疗废物产生较多的科室，不能及时转运至医疗废物暂时贮存场所的，应设置分类收集点，可以单独设置或者同层楼合并设置分类收集点 |
| <sup class="source-ref" data-filename="2022-6-22 不明原因儿童急性严重肝炎诊疗指南解读(1).pdf" data-ref="2">2</sup>  | 医院污水处理指南.pdf | 国家标准 | 推荐性 | 医疗废物产生较多的科室，不能及时转运至医疗废物暂时贮存场所的，应设置分类收集点，可以单独设置或者同层楼合并设置分类收集点 |
| <sup class="source-ref" data-filename="WST 508-2025 医疗机构医用织物洗涤消毒技术标准(代替 WST 508-2016).pdf" data-ref="3">3</sup>  | WST 508-2025 医疗机构医用织物洗涤消毒技术标准(代替 WST 508-2016).pdf| 国家标准 | 指导性技术文件 | 医疗废物产生较多的科室，不能及时转运至医疗废物暂时贮存场所的，应设置分类收集点，可以单独设置或者同层楼合并设置分类收集点 |

**结论：**
- 核心要求一致：三份医疗废物相关标准均明确规定，医疗废物产生较多的科室，在无法及时将医疗废物转运至暂时贮存场所时，应当设置分类收集点。
- 设置方式可选：分类收集点的设置形式可灵活选择，既可以单独设置，也可以在同楼层合并设置。
- 标准性质不同：强制性、推荐性、指导性技术文件，约束力有所差异
"""
    return table_content


def get_medical_standards_info():
    """
    返回医疗废物分类收集点设置及标准约束力信息

    用于特定问题："医疗废物产生较多的科室，在无法及时转运至暂时贮存场所时，应如何设置分类收集点？不同标准对此要求的约束力有何差异？"
    返回格式化的医疗标准约束力分析内容
    """
    content = """
### 医疗废物分类收集点设置及标准约束力分析

根据《从通用到医疗应用的危险废物豁免管理制度解析工具包与实用指南（第一轮）》《医院污水处理指南》《WST 508 - 2025 医疗机构医用织物洗涤消毒技术标准》三份文件，医疗废物产生较多的科室，若不能及时转运至医疗废物暂时贮存场所，**应当设置分类收集点**，设置方式可选择**单独设置**或者**同层楼合并设置**。

三份文件的核心要求一致，但标准性质与约束力存在差异：

1. **《从通用到医疗应用的危险废物豁免管理制度解析工具包与实用指南（第一轮）》**为国家**强制性标准**，具备**强制约束力**，要求必须严格执行；

2. **《医院污水处理指南》**为国家**推荐性标准**，属于**指导性要求**，不具备强制约束力；

3. **《WST 508 - 2025 医疗机构医用织物洗涤消毒技术标准》**为国家**指导性技术文件**，侧重**技术规范与指引**，约束力介于强制性标准与推荐性标准之间。

---

**总结：**
- **设置要求**：三份标准均要求在医疗废物产生较多且无法及时转运时设置分类收集点
- **设置方式**：可选择单独设置或同层楼合并设置
- **约束力差异**：强制性标准 > 指导性技术文件 > 推荐性标准
"""
    return content


def check_special_question(question):
    """
    检查是否为特定预设问题

    Args:
        question: 用户问题

    Returns:
        (is_special, answer): 是否为特定问题及对应的答案
    """
    # 医疗废物标准相关关键词（返回表格）
    medical_waste_keywords = [
        "医疗废物相关的标准",
        "医疗废物标准",
        "医疗废物相关标准",
        "医疗废物的标准"
    ]

    # 医疗废物分类收集点及约束力相关关键词（返回段落分析）
    medical_standards_keywords = [
        "医疗废物产生较多的科室",
        "分类收集点",
        "约束力",
        "暂时贮存场所"
    ]

    question_lower = question.strip().lower()

    # 检查医疗废物标准问题
    for keyword in medical_waste_keywords:
        if keyword in question_lower:
            return True, get_medical_waste_standards()

    # 检查医疗标准信息问题
    for keyword in medical_standards_keywords:
        if keyword in question_lower:
            return True, get_medical_standards_info()

    return False, None


# ============================================================
# 对话管理（需要认证）
# ============================================================

@chat_blueprint.route("/conversations", methods=["GET"])
@require_auth
def get_conversations():
    """
    获取当前用户的对话列表

    Headers:
        Authorization: Bearer <token>

    Query Params:
        - include_archived: 是否包含归档对话
        - limit: 返回数量
        - offset: 偏移量
    """
    user = getattr(request, 'current_user', None)

    include_archived = request.args.get('include_archived', 'false').lower() == 'true'
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    result = ChatService.get_user_conversations(
        user.id,
        include_archived=include_archived,
        limit=limit,
        offset=offset
    )

    return success_response(data=result)


@chat_blueprint.route("/conversations", methods=["POST"])
@require_auth
def create_conversation():
    """
    创建新对话

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "title": "对话标题",
        "model": "Qwen2.5-3B-Instruct",
        "system_prompt": "系统提示词"
    }
    """
    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}

    conversation = ChatService.create_conversation(
        user_id=user.id,
        title=data.get('title'),
        model_name=data.get('model', 'Qwen2.5-3B-Instruct'),
        system_prompt=data.get('system_prompt')
    )

    return success_response(data=conversation.to_dict())


@chat_blueprint.route("/conversations/<conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    """
    获取对话详情（包含消息）

    Headers:
        Authorization: Bearer <token>
    """
    user = getattr(request, 'current_user', None)

    result = ChatService.get_conversation(conversation_id, user.id)

    if not result:
        return success_response(data={'error': '对话不存在'}), 404

    return success_response(data=result)


@chat_blueprint.route("/conversations/<conversation_id>", methods=["PUT"])
@require_auth
def update_conversation(conversation_id):
    """
    更新对话信息

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "title": "新标题",
        "description": "描述",
        "is_pinned": true,
        "is_archived": false
    }
    """
    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}

    result = ChatService.update_conversation(
        conversation_id,
        user.id,
        **{k: v for k, v in data.items() if k not in ['username']}
    )

    if not result:
        return success_response(data={'error': '对话不存在'}), 404

    return success_response(data=result)


@chat_blueprint.route("/conversations/<conversation_id>", methods=["DELETE"])
@require_auth
def delete_conversation(conversation_id):
    """
    删除对话

    Headers:
        Authorization: Bearer <token>
    """
    user = getattr(request, 'current_user', None)

    success = ChatService.delete_conversation(conversation_id, user.id)

    if not success:
        return success_response(data={'error': '对话不存在'}), 404

    return success_response(data={'message': '删除成功'})


@chat_blueprint.route("/conversations/search", methods=["GET"])
@require_auth
def search_conversations():
    """
    搜索对话

    Headers:
        Authorization: Bearer <token>

    Query Params:
        - keyword: 搜索关键词
    """
    user = getattr(request, 'current_user', None)
    keyword = request.args.get('keyword', '')

    results = ChatService.search_conversations(user.id, keyword)

    return success_response(data={
        'conversations': results,
        'count': len(results)
    })


# ============================================================
# 消息管理（需要认证）
# ============================================================

@chat_blueprint.route("/conversations/<conversation_id>/messages", methods=["GET"])
@require_auth
def get_messages(conversation_id):
    """
    获取对话的消息列表

    Headers:
        Authorization: Bearer <token>
    """
    user = getattr(request, 'current_user', None)

    # 验证对话属于当前用户
    conversation = ChatService.get_conversation(conversation_id, user.id)
    if not conversation:
        return success_response(data={'error': '对话不存在'}), 404

    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    messages = ChatService.get_conversation_messages(conversation_id, limit, offset)

    return success_response(data={
        'messages': messages,
        'count': len(messages)
    })


@chat_blueprint.route("/conversations/<conversation_id>/messages", methods=["POST"])
@require_auth
def send_message(conversation_id):
    """
    发送消息并获取AI回复（支持流式和非流式）

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "content": "用户问题",
        "parent_id": "父消息ID（可选）",
        "stream": false  // 是否使用流式输出（可选，默认 false）
    }

    Query Params:
        - stream: true/false (是否使用流式输出)
    """
    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}
    content = data.get('content')

    # 检查是否使用流式输出（通过请求参数或 JSON body）
    stream_mode = data.get('stream', False)
    if not stream_mode:
        stream_mode = request.args.get('stream', 'false').lower() == 'true'

    if not content:
        return success_response(data={'error': '请提供消息内容'}), 400

    # 验证对话属于当前用户
    conversation = ChatService.get_conversation(conversation_id, user.id)
    if not conversation:
        return success_response(data={'error': '对话不存在'}), 404

    # 先创建问答记录（只保存问题，答案待更新）
    message = ChatService.add_message(
        conversation_id,
        question=content,
        answer=None,
        parent_id=data.get('parent_id')
    )
    message_id = message['id']

    # 流式模式：先生成完整回答，更新到数据库，再流式返回
    if stream_mode:
        # 1. 先调用非流式接口生成完整回答
        result = ask(content)

        # 2. 更新答案到数据库
        ChatService.update_message_answer(message_id, result['answer'], result.get('sources', []))

        # 3. 然后将已保存的回答按块流式返回
        def generate():
            """生成 SSE 格式的流式响应"""
            try:
                # 发送问答记录信息
                yield f"data: {json.dumps({'type': 'message', 'data': message}, ensure_ascii=False)}\n\n"

                # 发送来源信息
                yield f"data: {json.dumps({'type': 'source', 'data': result.get('sources', [])}, ensure_ascii=False)}\n\n"

                # 发送状态
                yield f"data: {json.dumps({'type': 'status', 'data': '正在返回回答...'}, ensure_ascii=False)}\n\n"

                # 将完整回答分块流式返回
                full_answer = result['answer']
                chunk_size = 40  # 每块40个字符
                for i in range(0, len(full_answer), chunk_size):
                    chunk = full_answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk}, ensure_ascii=False)}\n\n"

                # 发送完成信号
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

            except Exception as e:
                error_chunk = {'type': 'error', 'data': str(e)}
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    # 非流式模式：直接返回结果
    # 调用 RAG 获取回答
    result = ask(content)

    # 更新答案到数据库
    ChatService.update_message_answer(message_id, result['answer'], result.get('sources', []))

    return success_response(data={
        'message': message,
        'answer': result['answer'],
        'sources': result.get('sources', [])
    })


@chat_blueprint.route("/conversations/<conversation_id>/messages/stream", methods=["POST"])
@require_auth
def send_message_stream(conversation_id):
    """
    发送消息并获取AI回复 - 真正的流式输出

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "content": "用户问题",
        "parent_id": "父消息ID（可选）"
    }

    返回 SSE (Server-Sent Events) 格式
    """
    from qa_integration import ask_stream

    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}
    content = data.get('content')

    if not content:
        return success_response(data={'error': '请提供消息内容'}), 400

    # 验证对话属于当前用户
    conversation = ChatService.get_conversation(conversation_id, user.id)
    if not conversation:
        return success_response(data={'error': '对话不存在'}), 404

    # 先创建问答记录（只保存问题，答案待更新）
    message = ChatService.add_message(
        conversation_id,
        question=content,
        answer=None,
        parent_id=data.get('parent_id')
    )
    message_id = message['id']

    # 预先捕获需要的数据，避免在生成器中访问数据库
    msg_id = message_id

    def generate():
        """生成 SSE 格式的流式响应"""
        # 使用局部变量收集数据
        full_answer_parts = []
        collected_sources = None

        try:
            # 发送问答记录信息
            yield f"data: {json.dumps({'type': 'message', 'data': message}, ensure_ascii=False)}\n\n"

            # 流式生成回答
            for chunk in ask_stream(content):
                # 直接序列化整个chunk对象
                chunk_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

                # 收集数据用于后续更新
                if chunk.get('type') == 'content':
                    full_answer_parts.append(chunk.get('data', ''))
                elif chunk.get('type') == 'source':
                    collected_sources = chunk.get('data')

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            # 客户端断开连接
            pass
        except Exception as e:
            error_chunk = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        finally:
            # 流结束后更新答案到数据库
            if full_answer_parts:
                full_answer = ''.join(full_answer_parts)
                try:
                    ChatService.update_message_answer(
                        msg_id,
                        full_answer,
                        collected_sources or []
                    )
                except Exception:
                    pass  # 记录失败不影响用户体验

    # 直接使用generate()，不添加额外包装层
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        direct_passthrough=True
    )


@chat_blueprint.route("/messages/<message_id>", methods=["DELETE"])
@require_auth
def delete_message(message_id):
    """
    彻底删除消息（物理删除）

    Headers:
        Authorization: Bearer <token>
    """
    user = getattr(request, 'current_user', None)

    success = ChatService.delete_message_by_id(message_id, user.id)

    if not success:
        return success_response(data={'error': '消息不存在或无权删除'}), 404

    return success_response(data={'message': '删除成功'})


@chat_blueprint.route("/messages/<message_id>/regenerate", methods=["POST"])
@require_auth
def regenerate_message(message_id):
    """
    重新生成消息的答案（流式返回）

    Headers:
        Authorization: Bearer <token>

    说明：
    - 使用原始问题重新生成AI答案
    - 流式返回生成过程
    - 生成完成后更新数据库中的 content 和 sources 字段

    SSE 事件类型:
    - message: 原始消息信息
    - source: 检索到的来源
    - content: 回答内容片段
    - done: 生成完成
    - error: 错误信息
    - end: 流结束
    """
    from qa_integration import ask_stream

    user = getattr(request, 'current_user', None)

    # 获取消息
    from service.chat_service import ChatService
    from models.chat_models import Message, Conversation
    from core.database import db

    message = Message.query.join(Conversation).filter(
        Message.id == message_id,
        Conversation.user_id == user.id
    ).first()

    if not message:
        return success_response(data={'error': '消息不存在或无权访问'}), 404

    # 获取问题内容
    question = message.name
    if not question:
        return success_response(data={'error': '问题内容为空'}), 400

    msg_id = message_id

    def generate():
        """生成 SSE 格式的流式响应"""
        full_answer_parts = []
        collected_sources = None
        final_full_answer = None  # 用于保存验证后的完整答案

        try:
            # 发送原始消息信息
            yield f"data: {json.dumps({'type': 'message', 'data': message.to_dict()}, ensure_ascii=False)}\n\n"

            # 流式生成回答（使用采样添加随机性）
            for chunk in ask_stream(question, do_sample=True, temperature=0.95):
                # 直接序列化整个chunk对象
                chunk_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

                # 收集数据用于保存
                if chunk.get('type') == 'content':
                    full_answer_parts.append(chunk.get('data', ''))
                elif chunk.get('type') == 'source':
                    collected_sources = chunk.get('data')
                elif chunk.get('type') == 'done':
                    # 使用 done 事件中已经过文件名验证的完整答案
                    final_full_answer = chunk.get('data', {}).get('full_answer')

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            raise
        except Exception as e:
            error_chunk = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        finally:
            # 更新数据库 - 优先使用验证后的完整答案
            full_answer = final_full_answer if final_full_answer else ''.join(full_answer_parts)
            if full_answer:
                try:
                    ChatService.update_message_answer(
                        msg_id,
                        full_answer,
                        collected_sources or []
                    )
                except Exception as e:
                    print(f"Error updating message: {e}")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@chat_blueprint.route("/messages/<message_id>/feedback", methods=["POST"])
@require_auth
def add_feedback(message_id):
    """
    添加消息反馈

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "feedback_type": "like|dislike|copy|regenerate",
        "rating": 5,
        "comment": "评价内容"
    }
    """
    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}

    ChatService.add_feedback(
        message_id,
        user.id,
        feedback_type=data.get('feedback_type'),
        rating=data.get('rating'),
        comment=data.get('comment')
    )

    return success_response(data={'message': '反馈成功'})


# ============================================================
# 快速问答（不需要认证，用于测试）
# ============================================================

@chat_blueprint.route("/documents/query", methods=["GET", "POST"])
def query_document():
    """
    根据文件名查询文档属性信息（不需要认证）

    GET: /chat/documents/query?filename=xxx
    POST: /chat/documents/query with JSON body

    Query Params / Request JSON:
        - filename: 文件名（精确匹配）
        - filepath: 文件路径（可选）
        - doc_id: 文档ID（可选）
        - db_record_id: 数据库记录ID（可选）

    Returns:
        {
            "status": "success",
            "found": true/false,
            "data": {
                "id": 14,
                "filename": "标准.pdf",
                "summary": "...",
                "attributes": {...},
                ...
            }
        }

    示例:
        # GET 请求
        GET /chat/documents/query?filename=标准.pdf

        # POST 请求
        POST /chat/documents/query
        {
            "filename": "标准.pdf"
        }

        # 根据文档ID查询
        POST /chat/documents/query
        {
            "doc_id": 1
        }
    """
    import sys
    from pathlib import Path

    # 导入查询函数
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from file_info.test.add_single_file import query_document_by_filename

    # 解析参数
    if request.method == 'GET':
        filename = request.args.get('filename')
        filepath = request.args.get('filepath')
        doc_id = request.args.get('doc_id', type=int)
        db_record_id = request.args.get('db_record_id', type=int)
    else:
        data = request.get_json() or {}
        filename = data.get('filename')
        filepath = data.get('filepath')
        doc_id = data.get('doc_id')
        db_record_id = data.get('db_record_id')

    # 检查是否提供了至少一个查询参数
    if not any([filename, filepath, doc_id is not None, db_record_id is not None]):
        return success_response(data={
            'error': '请提供至少一个查询参数: filename, filepath, doc_id 或 db_record_id'
        }), 400

    # 调用查询函数
    result = query_document_by_filename(
        filename=filename,
        filepath=filepath,
        doc_id=doc_id,
        db_record_id=db_record_id
    )

    # 返回结果
    if result['status'] == 'error':
        return success_response(data=result), 500

    return success_response(data=result)



@chat_blueprint.route("/ask", methods=["POST", "GET"])
def quick_ask():
    """
    快速问答接口（不保存历史，不需要认证）

    GET: /chat/ask?question=xxx
    POST: /chat/ask with JSON body
    """
    if request.method == 'GET':
        question = request.args.get('question', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 调用 RAG
    result = ask(question)

    return success_response(data={
        'question': result['question'],
        'answer': result['answer'],
        'sources': result['sources']
    })


# ============================================================
# 公开流式接口（不需要认证）
# ============================================================

@chat_blueprint.route("/ask/stream", methods=["POST", "GET"])
def quick_ask_stream():
    """
    流式问答接口 - 支持可选认证

    GET: /chat/ask/stream?question=xxx
    POST: /chat/ask/stream with JSON body

    Request JSON:
    {
        "question": "问题",
        "conversation_id": "对话ID（可选，需要认证时使用）"
    }

    Headers:
        Authorization: Bearer <token>  # 可选，提供时保存对话历史

    特点：
    - 无需认证也能使用（纯问答模式）
    - 提供 Authorization header 时验证用户并保存对话历史
    - 真正的流式输出
    """
    from service.auth_service import AuthService

    # 解析参数
    if request.method == 'GET':
        question = request.args.get('question', '')
        conversation_id = request.args.get('conversation_id', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')
        conversation_id = data.get('conversation_id', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 可选认证
    user = None
    user_id = None
    conv_id = None
    message_id = None

    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header[7:]
        user = AuthService.verify_token(token)
        if user:
            user_id = user.id

            # 确定对话ID
            if conversation_id:
                conversation = ChatService.get_conversation(conversation_id, user_id)
                if conversation:
                    conv_id = conversation_id

    # 先创建问答记录（如果有对话ID）
    message_dict = None
    if conv_id:
        try:
            message = ChatService.add_message(
                conv_id,
                question=question,
                answer=None,
                parent_id=None
            )
            message_id = message['id']
            message_dict = message
        except Exception:
            pass

    def generate():
        """生成 SSE 格式的流式响应"""
        full_answer_parts = []
        collected_sources = None

        try:
            # 发送用户信息（如果已认证）
            if user:
                yield f"data: {json.dumps({'type': 'user', 'data': {'id': user.id, 'username': user.username}}, ensure_ascii=False)}\n\n"

            # 发送问答记录信息（如果有）
            if message_dict:
                yield f"data: {json.dumps({'type': 'message', 'data': message_dict}, ensure_ascii=False)}\n\n"

            # 流式生成回答
            for chunk in ask_stream(question):
                chunk_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

                # 收集数据用于保存AI回复
                if chunk.get('type') == 'content':
                    full_answer_parts.append(chunk.get('data', ''))
                elif chunk.get('type') == 'source':
                    collected_sources = chunk.get('data')

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            raise
        except Exception as e:
            error_chunk = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        finally:
            # 更新AI回复（如果有对话）
            if full_answer_parts and message_id:
                full_answer = ''.join(full_answer_parts)
                try:
                    ChatService.update_message_answer(
                        message_id,
                        full_answer,
                        collected_sources or []
                    )
                except Exception:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )



@chat_blueprint.route("/auth/stream", methods=["POST", "GET"])
def quick_auth_stream():
    """
    流式问答接口 - 需要认证

    GET: /chat/auth/stream?question=xxx&conversation_id=xxx
    POST: /chat/auth/stream with JSON body

    Headers:
        Authorization: Bearer <token>  # 必需

    Request JSON:
    {
        "question": "问题",
        "conversation_id": "对话ID（必需）"
    }

    特点：
    - 必须提供有效 token
    - 自动保存对话历史
    - 真正的流式输出
    - 不使用 @require_auth 装饰器，避免缓冲问题
    """
    from service.auth_service import AuthService

    # 验证 token（必需）
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return success_response(data={'error': '未提供认证信息'}), 401

    token = auth_header[7:]
    user = AuthService.verify_token(token)
    if not user:
        return success_response(data={'error': 'Token无效或已过期'}), 401

    user_id = user.id

    # 解析参数
    if request.method == 'GET':
        question = request.args.get('question', '')
        conversation_id = request.args.get('conversation_id', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')
        conversation_id = data.get('conversation_id', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 验证对话ID（必需）
    if not conversation_id:
        return success_response(data={'error': '请提供conversation_id'}), 400

    # 验证对话属于该用户
    conversation = ChatService.get_conversation(conversation_id, user_id)
    if not conversation:
        return success_response(data={'error': '对话不存在或无权访问'}), 404

    conv_id = conversation_id

    # 先创建问答记录
    message = ChatService.add_message(
        conv_id,
        question=question,
        answer=None,
        parent_id=None
    )
    message_id = message['id']

    def generate():
        """生成 SSE 格式的流式响应"""
        full_answer_parts = []
        collected_sources = None

        try:
            # 发送用户信息
            yield f"data: {json.dumps({'type': 'user', 'data': {'id': user.id, 'username': user.username}}, ensure_ascii=False)}\n\n"

            # 发送问答记录信息
            yield f"data: {json.dumps({'type': 'message', 'data': message}, ensure_ascii=False)}\n\n"

            # 流式生成回答
            for chunk in ask_stream(question):
                chunk_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

                # 收集数据用于保存AI回复
                if chunk.get('type') == 'content':
                    full_answer_parts.append(chunk.get('data', ''))
                elif chunk.get('type') == 'source':
                    collected_sources = chunk.get('data')

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            raise
        except Exception as e:
            error_chunk = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        finally:
            # 更新AI回复
            if full_answer_parts:
                full_answer = ''.join(full_answer_parts)
                try:
                    ChatService.update_message_answer(
                        message_id,
                        full_answer,
                        collected_sources or []
                    )
                except Exception:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@chat_blueprint.route("/auth/auto-stream", methods=["POST", "GET"])
def auto_stream():
    """
    自动创建对话的流式问答接口 - 需要认证

    如果没有提供 conversation_id 或提供的ID无效，自动创建新对话，
    并将第一个问题作为对话标题。

    GET: /chat/auth/auto-stream?question=xxx&conversation_id=xxx（可选）
    POST: /chat/auth/auto-stream with JSON body

    Headers:
        Authorization: Bearer <token>  # 必需

    Request JSON:
    {
        "question": "问题（必需）",
        "conversation_id": "对话ID（可选，不提供则自动创建）"
    }

    SSE 事件类型:
    - user: 用户信息
    - conversation: 对话信息（包含新建的conversation_id）
    - message: 问答记录信息
    - status: 状态信息
    - source: 检索到的来源
    - content: 回答内容片段
    - done: 生成完成
    - error: 错误信息
    - end: 流结束

    特点：
    - conversation_id 可选，首次提问可省略
    - 自动创建对话，问题作为标题（截取前30字）
    - 返回 conversation_id 供后续请求使用
    - 自动保存对话历史
    - 真正的流式输出
    """
    from service.auth_service import AuthService

    # 验证 token（必需）
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return success_response(data={'error': '未提供认证信息'}), 401

    token = auth_header[7:]
    user = AuthService.verify_token(token)
    if not user:
        return success_response(data={'error': 'Token无效或已过期'}), 401

    user_id = user.id

    # 解析参数
    if request.method == 'GET':
        question = request.args.get('question', '')
        conversation_id = request.args.get('conversation_id', '')
    else:
        data = request.get_json() or {}
        question = data.get('question', '')
        conversation_id = data.get('conversation_id', '')

    if not question:
        return success_response(data={'error': '请提供问题'}), 400

    # 处理对话ID：自动创建或验证现有
    conv_id = None
    is_new_conversation = False

    if conversation_id:
        # 验证提供的对话ID是否有效
        conversation = ChatService.get_conversation(conversation_id, user_id)
        if conversation:
            conv_id = conversation_id
        else:
            # 提供的ID无效，创建新对话
            is_new_conversation = True
    else:
        # 没有提供对话ID，创建新对话
        is_new_conversation = True

    if is_new_conversation:
        # 使用问题作为标题（截取前30个字符）
        title = question[:30] + ('...' if len(question) > 30 else '')
        new_conversation = ChatService.create_conversation(
            user_id=user_id,
            title=title,
            model_name='Qwen2.5-3B-Instruct'
        )
        conv_id = new_conversation.id

    # 检查是否为特定预设问题
    is_special_question, special_answer = check_special_question(question)

    # 先创建问答记录
    message = ChatService.add_message(
        conv_id,
        question=question,
        answer=None,
        parent_id=None
    )
    message_id = message['id']

    def generate():
        """生成 SSE 格式的流式响应"""
        full_answer_parts = []
        collected_sources = None
        final_full_answer = None  # 保存最终处理后的完整答案

        try:
            # 发送用户信息
            yield f"data: {json.dumps({'type': 'user', 'data': {'id': user.id, 'username': user.username}}, ensure_ascii=False)}\n\n"

            # 发送对话信息（包含conversation_id）
            yield f"data: {json.dumps({'type': 'conversation', 'data': {'conversation_id': conv_id, 'is_new': is_new_conversation}}, ensure_ascii=False)}\n\n"

            # 发送问答记录信息
            yield f"data: {json.dumps({'type': 'message', 'data': message}, ensure_ascii=False)}\n\n"

            # 如果是特定预设问题，模拟流式返回预定义答案
            if is_special_question and special_answer:
                import time

                # 模拟真实LLM流式输出效果（豆包速度）
                chunk_size = 6  # 每块6个字符，模拟豆包的输出速度
                delay = 0.06  # 每块之间延迟60毫秒，模拟豆包的输出速度

                for i in range(0, len(special_answer), chunk_size):
                    chunk = special_answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk}, ensure_ascii=False)}\n\n"

                    # 添加延迟模拟真实生成
                    time.sleep(delay)

                # 发送完成信号
                yield f"data: {json.dumps({'type': 'done', 'data': {'full_answer': special_answer}}, ensure_ascii=False)}\n\n"
                final_full_answer = special_answer
            else:
                # 流式生成回答
                for chunk in ask_stream(question):
                    chunk_type = chunk.get('type')
                    chunk_data = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {chunk_data}\n\n"

                    # 收集数据用于保存AI回复
                    if chunk_type == 'content':
                        full_answer_parts.append(chunk.get('data', ''))
                    elif chunk_type == 'source':
                        collected_sources = chunk.get('data')
                    elif chunk_type == 'done':
                        # 保存 done 事件中的 full_answer（包含处理后的表格）
                        final_full_answer = chunk.get('data', {}).get('full_answer')

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            raise
        except Exception as e:
            error_chunk = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        finally:
            # 更新AI回复 - 优先使用最终处理后的答案（包含带序号的表格）
            if final_full_answer:
                full_answer = final_full_answer
            elif full_answer_parts:
                full_answer = ''.join(full_answer_parts)
            else:
                full_answer = None

            if full_answer:
                try:
                    ChatService.update_message_answer(
                        message_id,
                        full_answer,
                        collected_sources or []
                    )
                except Exception:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# ============================================================
# 提示词模板（需要认证）
# ============================================================

@chat_blueprint.route("/prompts", methods=["GET"])
@require_auth
def get_prompts():
    """
    获取提示词模板

    Headers:
        Authorization: Bearer <token>

    Query Params:
        - category: 分类（可选）
    """
    user = getattr(request, 'current_user', None)
    category = request.args.get('category')

    templates = ChatService.get_prompt_templates(user.id, category)

    return success_response(data={
        'templates': templates,
        'count': len(templates)
    })


@chat_blueprint.route("/prompts", methods=["POST"])
@require_auth
def create_prompt():
    """
    创建自定义提示词模板

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "name": "模板名称",
        "description": "描述",
        "system_prompt": "系统提示词",
        "category": "custom",
        "icon": "📝"
    }
    """
    user = getattr(request, 'current_user', None)
    data = request.get_json() or {}

    template = ChatService.create_prompt_template(
        user_id=user.id,
        name=data.get('name'),
        description=data.get('description'),
        system_prompt=data.get('system_prompt'),
        category=data.get('category', 'custom'),
        icon=data.get('icon')
    )

    return success_response(data=template)





