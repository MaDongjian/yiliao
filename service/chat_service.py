# -*- coding: utf-8 -*-
"""
对话服务层 - 处理对话历史业务逻辑
"""
from models.chat_models import User, Conversation, Message, MessageFeedback, ConversationShare, PromptTemplate
from core.database import db
from datetime import datetime
from sqlalchemy import and_, or_
import json


class ChatService:
    """对话服务"""

    @staticmethod
    def get_or_create_user(username, user_info=None):
        """获取或创建用户（从 auth_user 同步）"""
        user = User.query.filter_by(username=username).first()

        if not user:
            # 从 auth_user 获取用户信息
            from models.auth_models import AuthUser
            auth_user_obj = AuthUser.query.filter_by(username=username).first()

            if auth_user_obj:
                # 如果 auth_user 存在，使用它的信息
                user = User(
                    id=auth_user_obj.id,  # 使用相同的 ID
                    username=username,
                    nickname=auth_user_obj.nickname or auth_user_obj.real_name,
                    avatar=auth_user_obj.avatar
                )
            else:
                # 兼容：如果 auth_user 不存在，创建新用户
                user = User(
                    username=username,
                    nickname=user_info.get('nickname') if user_info else username,
                    avatar=user_info.get('avatar') if user_info else None,
                    email=user_info.get('email') if user_info else None,
                    phone=user_info.get('phone') if user_info else None,
                )

            db.session.add(user)
            db.session.commit()

        # 更新最后活跃时间
        user.last_active_at = datetime.utcnow()
        db.session.commit()

        return user

    @staticmethod
    def create_conversation(user_id, title=None, model_name='Qwen2.5-3B-Instruct', system_prompt=None):
        """创建新对话"""
        # 确保 chat_user 表中有该用户
        user = User.query.get(user_id)
        if not user:
            # 从 auth_user 同步用户
            from models.auth_models import AuthUser
            auth_user_obj = AuthUser.query.get(user_id)

            if auth_user_obj:
                user = User(
                    id=auth_user_obj.id,
                    username=auth_user_obj.username,
                    nickname=auth_user_obj.nickname or auth_user_obj.real_name,
                    avatar=auth_user_obj.avatar
                )
                db.session.add(user)
                db.session.commit()
            else:
                raise Exception(f'用户不存在: {user_id}')

        conversation = Conversation(
            user_id=user_id,
            title=title,
            model_name=model_name,
            system_prompt=system_prompt
        )
        db.session.add(conversation)
        db.session.commit()
        return conversation

    @staticmethod
    def get_user_conversations(user_id, include_archived=False, limit=50, offset=0):
        """获取用户的对话列表"""
        query = Conversation.query.filter_by(user_id=user_id)

        if not include_archived:
            query = query.filter_by(is_archived=False)

        # 按创建时间倒序（置顶的对话仍然排在最前面）
        query = query.order_by(
            Conversation.is_pinned.desc(),
            Conversation.created_at.desc()
        )

        conversations = query.limit(limit).offset(offset).all()
        total = query.count()

        return {
            'conversations': [c.to_dict() for c in conversations],
            'total': total,
            'limit': limit,
            'offset': offset
        }

    @staticmethod
    def get_conversation(conversation_id, user_id=None):
        """获取对话详情"""
        query = Conversation.query.filter_by(id=conversation_id)

        if user_id:
            query = query.filter_by(user_id=user_id)

        conversation = query.first()

        if not conversation:
            return None

        return conversation.to_dict(include_messages=True)

    @staticmethod
    def update_conversation(conversation_id, user_id, **kwargs):
        """更新对话信息"""
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()

        if not conversation:
            return None

        # 可更新的字段
        for key in ['title', 'description', 'system_prompt', 'is_pinned', 'is_archived']:
            if key in kwargs:
                setattr(conversation, key, kwargs[key])

        db.session.commit()
        return conversation.to_dict()

    @staticmethod
    def delete_conversation(conversation_id, user_id):
        """删除对话"""
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()

        if not conversation:
            return False

        db.session.delete(conversation)
        db.session.commit()
        return True

    @staticmethod
    def add_message(conversation_id, question, answer=None, sources=None, parent_id=None, model=None):
        """添加问答消息（一条记录同时保存问题和答案）

        Args:
            conversation_id: 对话ID
            question: 问题内容（存储在 name 字段）
            answer: AI回答内容（存储在 content 字段，可为空）
            sources: RAG来源信息
            parent_id: 父消息ID
            model: 使用的模型
        """
        print(f"[DEBUG] add_message called: conv_id={conversation_id}, question={question[:50]}, answer={answer[:50] if answer else None}")
        conversation = Conversation.query.get(conversation_id)

        if not conversation:
            return None

        message = Message(
            conversation_id=conversation_id,
            role='user',
            name=question,
            content=answer,
            sources=sources,
            parent_id=parent_id,
            model=model or conversation.model_name
        )

        db.session.add(message)

        # 更新对话的消息数量和更新时间
        conversation.message_count += 1
        conversation.updated_at = datetime.utcnow()

        # 自动生成标题（用第一条问题）
        if not conversation.title:
            conversation.title = question[:50] + ('...' if len(question) > 50 else '')

        db.session.commit()
        print(f"[DEBUG] Message created: id={message.id}, role={message.role}, name={message.name[:50]}, content={message.content[:50] if message.content else None}")
        return message.to_dict()

    @staticmethod
    def update_message_answer(message_id, answer, sources=None):
        """更新消息的答案内容

        Args:
            message_id: 消息ID
            answer: AI回答内容
            sources: RAG来源信息
        """
        message = Message.query.get(message_id)
        if not message:
            return False

        message.content = answer
        if sources:
            message.sources = sources

        # 更新所属对话的 updated_at，确保对话列表按最近更新时间排序
        conversation = Conversation.query.get(message.conversation_id)
        if conversation:
            conversation.updated_at = datetime.utcnow()

        db.session.commit()
        return True

    @staticmethod
    def regenerate_message_answer(message_id, user_id=None):
        """重新生成消息的答案

        Args:
            message_id: 消息ID
            user_id: 用户ID（用于权限验证）

        Returns:
            dict: 包含新答案和来源的字典，如果失败返回 None
        """
        # 获取消息
        message = Message.query.get(message_id)
        if not message:
            return None

        # 验证权限（如果提供了 user_id）
        if user_id:
            conversation = Conversation.query.get(message.conversation_id)
            if not conversation or conversation.user_id != user_id:
                return None

        # 获取问题内容
        question = message.name
        if not question:
            return None

        try:
            # 调用 RAG 生成新答案
            from qa_integration import ask
            result = ask(question)

            # 更新消息的答案
            message.content = result['answer']
            if result.get('sources'):
                message.sources = result['sources']

            db.session.commit()

            return {
                'message': message.to_dict(),
                'answer': result['answer'],
                'sources': result.get('sources', [])
            }
        except Exception as e:
            print(f"Error regenerating answer: {e}")
            return None

    @staticmethod
    def delete_message_by_id(message_id, user_id):
        """彻底删除消息（物理删除）

        Args:
            message_id: 消息ID
            user_id: 用户ID（用于权限验证）

        Returns:
            bool: 删除是否成功
        """
        # 获取消息并验证权限
        message = Message.query.join(Conversation).filter(
            Message.id == message_id,
            Conversation.user_id == user_id
        ).first()

        if not message:
            return False

        try:
            db.session.delete(message)

            # 更新对话的消息数量
            conversation = Conversation.query.get(message.conversation_id)
            if conversation and conversation.message_count > 0:
                conversation.message_count -= 1

            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error deleting message: {e}")
            return False

    @staticmethod
    def get_conversation_messages(conversation_id, limit=100, offset=0):
        """获取对话的消息列表"""
        messages = Message.query.filter_by(
            conversation_id=conversation_id,
            is_deleted=False
        ).order_by(Message.created_at).limit(limit).offset(offset).all()

        return [m.to_dict() for m in messages]

    @staticmethod
    def delete_message(message_id, user_id):
        """删除消息（软删除）"""
        message = Message.query.join(Conversation).filter(
            Message.id == message_id,
            Conversation.user_id == user_id
        ).first()

        if not message:
            return False

        message.is_deleted = True
        db.session.commit()
        return True

    @staticmethod
    def add_feedback(message_id, user_id, feedback_type, rating=None, comment=None):
        """添加消息反馈"""
        # 检查是否已有反馈
        existing = MessageFeedback.query.filter_by(
            message_id=message_id,
            user_id=user_id
        ).first()

        if existing:
            # 更新
            existing.feedback_type = feedback_type
            existing.rating = rating
            existing.comment = comment
        else:
            feedback = MessageFeedback(
                message_id=message_id,
                user_id=user_id,
                feedback_type=feedback_type,
                rating=rating,
                comment=comment
            )
            db.session.add(feedback)

        db.session.commit()
        return True

    @staticmethod
    def search_conversations(user_id, keyword, limit=20):
        """搜索对话"""
        conversations = Conversation.query.filter(
            Conversation.user_id == user_id,
            Conversation.is_archived == False,
            or_(
                Conversation.title.like(f'%{keyword}%'),
                Conversation.description.like(f'%{keyword}%')
            )
        ).order_by(Conversation.updated_at.desc()).limit(limit).all()

        return [c.to_dict() for c in conversations]

    @staticmethod
    def get_chat_history(conversation_id, user_id=None):
        """获取对话历史（用于多轮对话上下文）"""
        query = Message.query.filter_by(
            conversation_id=conversation_id,
            is_deleted=False
        ).order_by(Message.created_at)

        messages = query.all()

        # 构建类似 ChatGPT 的历史格式
        history = []
        for msg in messages:
            history.append({
                'role': msg.role,
                'content': msg.content
            })

        return history

    @staticmethod
    def create_prompt_template(user_id, name, description, system_prompt, category='custom', icon=None):
        """创建提示词模板"""
        template = PromptTemplate(
            user_id=user_id,
            name=name,
            description=description,
            icon=icon,
            category=category,
            system_prompt=system_prompt
        )
        db.session.add(template)
        db.session.commit()
        return template.to_dict()

    @staticmethod
    def get_prompt_templates(user_id=None, category=None):
        """获取提示词模板"""
        query = PromptTemplate.query.filter_by(is_active=True)

        if user_id:
            # 获取系统模板 + 用户自定义模板
            query = query.filter(
                or_(
                    PromptTemplate.is_system == True,
                    PromptTemplate.user_id == user_id
                )
            )
        else:
            # 只获取系统模板
            query = query.filter_by(is_system=True)

        if category:
            query = query.filter_by(category=category)

        templates = query.order_by(PromptTemplate.sort_order, PromptTemplate.usage_count.desc()).all()
        return [t.to_dict() for t in templates]


# 初始化系统提示词模板
def init_system_templates():
    """初始化系统提示词模板"""
    existing = PromptTemplate.query.filter_by(is_system=True).count()
    if existing > 0:
        return

    templates = [
        {
            'name': '医疗标准助手',
            'description': '专业的医疗标准知识问答助手',
            'icon': '🏥',
            'category': 'medical',
            'system_prompt': '你是一个专业的医疗标准知识助手。请根据参考信息准确回答问题，使用Markdown格式输出，灵活使用表格、列表等形式。',
            'sort_order': 1
        },
        {
            'name': '通用助手',
            'description': '通用的AI助手',
            'icon': '🤖',
            'category': 'general',
            'system_prompt': '你是一个有用的AI助手，请用Markdown格式回答问题。',
            'sort_order': 2
        },
        {
            'name': '文档分析',
            'description': '分析文档内容并总结',
            'icon': '📄',
            'category': 'analysis',
            'system_prompt': '你是一个文档分析专家，请仔细分析文档内容，提供详细的总结和要点。',
            'sort_order': 3
        },
        {
            'name': '表格生成',
            'description': '将信息整理成表格',
            'icon': '📊',
            'category': 'format',
            'system_prompt': '请将回答以表格形式呈现，使用Markdown表格格式。',
            'sort_order': 4
        }
    ]

    for t in templates:
        template = PromptTemplate(
            name=t['name'],
            description=t['description'],
            icon=t['icon'],
            category=t['category'],
            system_prompt=t['system_prompt'],
            sort_order=t['sort_order'],
            is_system=True
        )
        db.session.add(template)

    db.session.commit()
    print("系统提示词模板初始化完成")
