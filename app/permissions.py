"""RBAC 权限辅助：未登录用户视为访客 (visitor)。"""
from flask import jsonify
from flask_login import current_user

from .models import Dataset


def effective_role() -> str:
    if current_user.is_authenticated:
        return current_user.role or 'normal'
    return 'visitor'


def is_authenticated_user() -> bool:
    return bool(current_user.is_authenticated)


def visible_datasets_query():
    """按角色返回可见数据集查询。"""
    query = Dataset.query
    role = effective_role()

    if role == 'visitor':
        return query.filter(Dataset.upload_by.is_(None))
    if role in ('normal', 'expert'):
        return query.filter(
            (Dataset.upload_by == current_user.id) | (Dataset.upload_by.is_(None))
        )
    return query


def can_access_dataset(dataset: Dataset) -> bool:
    if dataset is None:
        return False
    role = effective_role()
    if role in ('sysadmin', 'labadmin'):
        return True
    if role == 'visitor':
        return dataset.upload_by is None
    if role in ('normal', 'expert'):
        return dataset.upload_by is None or dataset.upload_by == current_user.id
    return False


def get_accessible_dataset(dataset_id: int):
    """获取数据集；无权访问时返回 (None, response, status_code)。"""
    dataset = Dataset.query.get(dataset_id)
    if dataset is None:
        return None, jsonify({'error': '数据集不存在'}), 404
    if not can_access_dataset(dataset):
        return None, jsonify({'error': '无权访问该数据集'}), 403
    return dataset, None, None


def can_manage_data() -> bool:
    """上传、构建索引等写操作。"""
    return effective_role() in ('normal', 'expert', 'labadmin', 'sysadmin')


def user_to_dict(user, is_guest: bool = False) -> dict:
    if is_guest:
        return {
            'id':       None,
            'username': '访客',
            'email':    None,
            'role':     'visitor',
            'is_guest': True,
        }
    return {
        'id':       user.id,
        'username': user.username,
        'email':    user.email,
        'role':     user.role,
        'is_guest': False,
    }
