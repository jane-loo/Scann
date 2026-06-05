from functools import wraps
from flask import abort, jsonify
from flask_login import current_user

def login_required_api(f):
    """必须登录（访客未登录不可用）。"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['sysadmin', 'labadmin']:
            return jsonify({'error': '需要管理员权限 (sysadmin 或 labadmin)'}), 403
        return f(*args, **kwargs)
    return decorated_function

def sysadmin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'sysadmin':
            return jsonify({'error': '需要系统管理员权限 (sysadmin)'}), 403
        return f(*args, **kwargs)
    return decorated_function

def expert_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['expert', 'labadmin', 'sysadmin']:
            return jsonify({'error': '需要资深专家权限'}), 403
        return f(*args, **kwargs)
    return decorated_function
