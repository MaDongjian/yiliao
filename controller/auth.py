# -*- coding: utf-8 -*-
"""
用户认证 API 控制器
"""
from flask import Blueprint, request, jsonify
from utils.rest_response import success_response
from service.auth_service import AuthService, init_default_admin
from functools import wraps

auth_blueprint = Blueprint('auth', __name__)


def require_auth(f):
    """认证装饰器 - 验证 Token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 从 Header 获取 Token
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return success_response(data={'error': '未提供认证信息'}), 401

        if not auth_header.startswith('Bearer '):
            return success_response(data={'error': '认证格式错误'}), 401

        token = auth_header[7:]  # 移除 'Bearer ' 前缀

        # 验证 Token
        user = AuthService.verify_token(token)
        if not user:
            return success_response(data={'error': 'Token无效或已过期'}), 401

        # 将用户信息添加到请求对象
        request.current_user = user
        request.current_token = token

        return f(*args, **kwargs)
    return decorated_function


def require_admin(f):
    """管理员权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'current_user'):
            return success_response(data={'error': '未认证'}), 401

        if not request.current_user.is_admin:
            return success_response(data={'error': '需要管理员权限'}), 403

        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# 认证接口
# ============================================================

@auth_blueprint.route("/register", methods=["POST"])
def register():
    """
    用户注册

    Request JSON:
    {
        "username": "user123",
        "password": "password123",
        "real_name": "真实姓名",
        "nickname": "昵称",
        "email": "邮箱",
        "phone": "手机号",
        "department": "部门"
    }
    """
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return success_response(data={'error': '请提供用户名和密码'}), 400

    if len(username) < 3:
        return success_response(data={'error': '用户名至少3个字符'}), 400

    if len(password) < 6:
        return success_response(data={'error': '密码至少6个字符'}), 400

    result = AuthService.register(
        username=username,
        password=password,
        real_name=data.get('real_name'),
        nickname=data.get('nickname'),
        email=data.get('email'),
        phone=data.get('phone'),
        department=data.get('department')
    )

    if isinstance(result, dict) and 'error' in result:
        return success_response(data=result), 400

    return success_response(data={
        'user': result.to_dict(),
        'message': '注册成功'
    })


@auth_blueprint.route("/login", methods=["POST"])
def login():
    """
    用户登录

    Request JSON:
    {
        "username": "user123",
        "password": "password123"
    }

    Response:
    {
        "code": 0,
        "data": {
            "user": {...},
            "token": "jwt_token_here"
        }
    }
    """
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return success_response(data={'error': '请提供用户名和密码'}), 400

    # 获取客户端信息
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')[:255]

    user, token, error = AuthService.login(username, password, ip_address, user_agent)

    if error:
        return success_response(data={'error': error}), 401

    return success_response(data={
        'user': user.to_dict(),
        'token': token
    })


@auth_blueprint.route("/logout", methods=["POST"])
@require_auth
def logout():
    """
    用户登出

    Headers:
        Authorization: Bearer <token>
    """
    token = getattr(request, 'current_token', None)
    if token:
        AuthService.logout(token)

    return success_response(data={'message': '登出成功'})


@auth_blueprint.route("/me", methods=["GET"])
@require_auth
def get_current_user():
    """
    获取当前用户信息

    Headers:
        Authorization: Bearer <token>
    """
    user = getattr(request, 'current_user', None)
    if user:
        return success_response(data=user.to_dict())

    return success_response(data={'error': '未认证'}), 401


@auth_blueprint.route("/me", methods=["PUT"])
@require_auth
def update_current_user():
    """
    更新当前用户信息

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "real_name": "新姓名",
        "nickname": "新昵称",
        "email": "新邮箱",
        "phone": "新手机号",
        "department": "新部门"
    }
    """
    user = getattr(request, 'current_user', None)
    if not user:
        return success_response(data={'error': '未认证'}), 401

    data = request.get_json() or {}

    result = AuthService.update_user(user.id, **data)

    if result:
        return success_response(data={'user': result.to_dict(), 'message': '更新成功'})

    return success_response(data={'error': '更新失败'}), 400


@auth_blueprint.route("/change-password", methods=["POST"])
@require_auth
def change_password():
    """
    修改密码

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "old_password": "旧密码",
        "new_password": "新密码"
    }
    """
    user = getattr(request, 'current_user', None)
    if not user:
        return success_response(data={'error': '未认证'}), 401

    data = request.get_json() or {}
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    if not old_password or not new_password:
        return success_response(data={'error': '请提供旧密码和新密码'}), 400

    if len(new_password) < 6:
        return success_response(data={'error': '新密码至少6个字符'}), 400

    result = AuthService.change_password(user.id, old_password, new_password)

    if 'error' in result:
        return success_response(data=result), 400

    return success_response(data={'message': '密码修改成功'})


# ============================================================
# 管理员接口
# ============================================================

@auth_blueprint.route("/admin/users", methods=["GET"])
@require_admin
def get_users():
    """
    获取用户列表（管理员）

    Query Params:
        - limit: 返回数量
        - offset: 偏移量
        - keyword: 搜索关键词
    """
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    keyword = request.args.get('keyword')

    result = AuthService.get_user_list(limit=limit, offset=offset, keyword=keyword)
    return success_response(data=result)


@auth_blueprint.route("/admin/users/<user_id>", methods=["PUT"])
@require_admin
def update_user_admin(user_id):
    """
    更新用户信息（管理员）

    Headers:
        Authorization: Bearer <token>

    Request JSON:
    {
        "is_active": true,
        "is_admin": false,
        "real_name": "姓名"
    }
    """
    data = request.get_json() or {}

    result = AuthService.update_user(user_id, **data)

    if result:
        return success_response(data={'user': result.to_dict(), 'message': '更新成功'})

    return success_response(data={'error': '用户不存在'}), 404


@auth_blueprint.route("/admin/users/<user_id>", methods=["DELETE"])
@require_admin
def delete_user(user_id):
    """
    删除用户（管理员）

    Headers:
        Authorization: Bearer <token>
    """
    from models.auth_models import AuthUser
    from core.database import db

    user = AuthUser.query.get(user_id)
    if not user:
        return success_response(data={'error': '用户不存在'}), 404

    if user.is_admin:
        return success_response(data={'error': '不能删除管理员账户'}), 400

    db.session.delete(user)
    db.session.commit()

    return success_response(data={'message': '删除成功'})


@auth_blueprint.route("/admin/login-logs", methods=["GET"])
@require_admin
def get_login_logs():
    """
    获取登录日志（管理员）

    Query Params:
        - user_id: 用户ID（可选）
        - limit: 返回数量
    """
    user_id = request.args.get('user_id')
    limit = int(request.args.get('limit', 50))

    logs = AuthService.get_login_logs(user_id=user_id, limit=limit)
    return success_response(data={'logs': logs, 'count': len(logs)})


# ============================================================
# 初始化
# ============================================================

@auth_blueprint.route("/init-admin", methods=["POST"])
def init_admin():
    """
    初始化默认管理员账户

    注意：生产环境请在安全的环境下执行，完成后删除或禁用此接口
    """
    try:
        admin = init_default_admin()
        return success_response(data={
            'message': '管理员账户创建成功',
            'user': admin.to_dict()
        })
    except Exception as e:
        return success_response(data={'error': str(e)}), 500
