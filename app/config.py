import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'scann.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    INDEX_FOLDER  = os.path.join(BASE_DIR, 'data', 'indexes')
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 最大上传 2 GB
