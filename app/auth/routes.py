from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, current_user
from ..models import db, User
from ..permissions import user_to_dict

auth_bp = Blueprint('auth', __name__)


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
    if User.query.filter_by(email=email).first():
        return jsonify({'error': '邮箱已存在'}), 400
    user = User(username=username, email=email, role='normal')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': '注册成功', 'user': user_to_dict(user)}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    user = User.query.filter_by(username=data.get('username', '').strip()).first()
    if not user or not user.is_active or not user.check_password(data.get('password', '')):
        return jsonify({'error': '用户名或密码错误'}), 401
    login_user(user, remember=True)
    return jsonify({'message': '登录成功', 'user': user_to_dict(user)})


@auth_bp.route('/logout', methods=['POST'])
def logout():
    if current_user.is_authenticated:
        logout_user()
    return jsonify({'message': '已退出', 'user': user_to_dict(None, is_guest=True)})


@auth_bp.route('/me', methods=['GET'])
def get_me():
    """未登录时返回访客身份，供前端默认展示演示数据。"""
    if not current_user.is_authenticated:
        return jsonify(user_to_dict(None, is_guest=True))
    return jsonify(user_to_dict(current_user))
