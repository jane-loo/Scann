import os

from sqlalchemy import inspect, text

from ..models import db


def ensure_evaluation_report_schema():
    """为已有 SQLite 数据库补全 evaluation_report 新列（幂等）。"""
    try:
        inspector = inspect(db.engine)
        if 'evaluation_report' not in inspector.get_table_names():
            return
        existing = {c['name'] for c in inspector.get_columns('evaluation_report')}
        migrations = [
            ('index_size_bytes', 'FLOAT'),
            ('exact_index_size_bytes', 'FLOAT'),
        ]
        for col, col_type in migrations:
            if col not in existing:
                db.session.execute(
                    text(f'ALTER TABLE evaluation_report ADD COLUMN {col} {col_type}')
                )
        db.session.commit()
    except Exception:
        db.session.rollback()


def index_file_size_bytes(ann_index) -> int:
    path = getattr(ann_index, 'index_file', None)
    if path and os.path.exists(path):
        return os.path.getsize(path)
    return 0


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return '0 B'
    units = ['B', 'KB', 'MB', 'GB']
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{num_bytes} B'
