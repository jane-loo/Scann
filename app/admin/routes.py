from flask import request, jsonify
from flask_login import login_required
from ..models import db, User
from ..decorators import sysadmin_required
from . import admin_bp

@admin_bp.route('/users', methods=['GET'])
@login_required
@sysadmin_required
def get_users():
    """获取所有用户列表 (仅限 sysadmin)"""
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
@sysadmin_required
def update_user(user_id):
    """修改用户角色或状态 (仅限 sysadmin)"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
        
    data = request.get_json() or {}
    if 'role' in data:
        valid_roles = ['visitor', 'normal', 'expert', 'labadmin', 'sysadmin']
        if data['role'] not in valid_roles:
            return jsonify({'error': f'角色值不合法，必须是 {valid_roles} 之一'}), 400
        user.role = data['role']
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
        
    db.session.commit()
    return jsonify({'message': '更新成功', 'user': {'id': user.id, 'role': user.role, 'is_active': user.is_active}})

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@login_required
@sysadmin_required
def delete_user(user_id):
    """物理删除用户及其关联数据 (仅限 sysadmin)"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
        
    if user.id == 1 or user.role == 'sysadmin': # 保护初始管理员和 sysadmin
        return jsonify({'error': '无法删除系统管理员'}), 400
        
    # TODO: 这里理想流程是调用成员 A 的删除接口清理该用户的所有数据集和索引文件
    # 目前先实现物理删除用户记录
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': '用户已彻底删除'})
