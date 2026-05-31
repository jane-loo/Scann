from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from ..models import db, User

auth_bp = Blueprint('auth', __name__)

# ----------------------------------------------------------------
# 最小实现，供成员 A 联调测试使用。
# 完整的登录/注册/管理员路由由成员 B 在 feature/auth 分支实现。
# ----------------------------------------------------------------

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip()
    password = data.get('password', '')
    if not all([username, email, password]):
        return jsonify({'error': '参数不完整'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已存在'}), 400
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': '注册成功', 'id': user.id}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    user = User.query.filter_by(username=data.get('username')).first()
    if not user or not user.check_password(data.get('password', '')):
        return jsonify({'error': '用户名或密码错误'}), 401
    login_user(user, remember=True)
    return jsonify({'message': '登录成功', 'role': user.role})


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': '已退出'})


@auth_bp.route('/me', methods=['GET'])
def get_me():
    """获取当前登录用户信息"""
    if not current_user.is_authenticated:
        return jsonify({'error': '未登录'}), 401
    return jsonify({
        'id': current_user.id,
        'username': current_user.username,
        'email': current_user.email,
        'role': current_user.role
    })
