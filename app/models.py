from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(64), unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    role       = db.Column(db.String(16), default='normal')
    is_active  = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw):
        self.password = generate_password_hash(raw)

    def check_password(self, raw):
        return check_password_hash(self.password, raw)


class Dataset(db.Model):
    __tablename__ = 'dataset'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    file_path   = db.Column(db.String(512), nullable=False)
    n_cells     = db.Column(db.Integer)
    n_genes     = db.Column(db.Integer)
    n_dims      = db.Column(db.Integer)       # PCA 维度数
    cell_types  = db.Column(db.Text)          # JSON 列表
    obs_columns = db.Column(db.Text)          # JSON 列表：可用元数据列名
    vector_meta = db.Column(db.Text)          # JSON：PCA 向量化来源与流水线说明
    upload_by   = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    indexes = db.relationship('AnnIndex', backref='dataset',
                              cascade='all, delete-orphan')


class AnnIndex(db.Model):
    __tablename__ = 'ann_index'
    id          = db.Column(db.Integer, primary_key=True)
    dataset_id  = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    index_type  = db.Column(db.String(32), nullable=False)  # 'hnsw'|'ivf_flat'|'ivf_pq'|'exact'
    metric      = db.Column(db.String(16), default='l2')    # 'l2'|'cosine'
    params      = db.Column(db.Text)          # JSON：构建参数
    index_file  = db.Column(db.String(512))   # 索引文件路径
    status      = db.Column(db.String(16), default='building')  # 'building'|'ready'|'error'
    build_time  = db.Column(db.Float)         # 构建耗时（秒）
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class QueryHistory(db.Model):
    __tablename__ = 'query_history'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'))
    dataset_id  = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    query_type  = db.Column(db.String(16))    # 'cell_id'|'vector'|'random'
    query_input = db.Column(db.Text)
    index_type  = db.Column(db.String(32))
    top_k       = db.Column(db.Integer)
    result_ids  = db.Column(db.Text)          # JSON 列表
    query_time  = db.Column(db.Float)         # 毫秒
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class EvaluationReport(db.Model):
    __tablename__ = 'evaluation_report'
    id          = db.Column(db.Integer, primary_key=True)
    dataset_id  = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    index_id    = db.Column(db.Integer, db.ForeignKey('ann_index.id'))
    recall_at_k = db.Column(db.Float)
    qps         = db.Column(db.Float)
    avg_latency = db.Column(db.Float)
    n_queries   = db.Column(db.Integer)
    k           = db.Column(db.Integer)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    index = db.relationship('AnnIndex', backref='reports')
