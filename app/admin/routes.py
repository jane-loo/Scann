from flask import request, jsonify
from flask_login import login_required
from ..models import db, User
from ..decorators import admin_required
from . import admin_bp

@admin_bp.route('/users', methods=['GET'])
@login_required
@admin_required
def get_users():
    """获取所有用户列表"""
    users = User.query.all()
    user_list = [{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'role': u.role,
        'is_active': u.is_active,
        'created_at': u.created_at.isoformat()
    } for u in users]
    return jsonify(user_list)

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@login_required
@admin_required
def update_user(user_id):
    """修改用户角色或状态"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
        
    data = request.get_json() or {}
    if 'role' in data:
        if data['role'] not in ['user', 'admin']:
            return jsonify({'error': '角色值不合法'}), 400
        user.role = data['role']
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
        
    db.session.commit()
    return jsonify({'message': '更新成功'})

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    """删除用户"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
        
    if user.id == 1: # 保护初始管理员
        return jsonify({'error': '无法删除系统管理员'}), 400
        
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': '用户已删除'})
