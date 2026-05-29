from flask import Flask
from flask_login import LoginManager
from flask_cors import CORS
from .models import db, User
from .config import Config

login_manager = LoginManager()


def create_app(test_config: dict = None):
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')
    app.config.from_object(Config)
    
    # 开启跨域支持
    CORS(app, supports_credentials=True)

    # 测试时在 db.init_app 之前覆盖配置，确保 URI 生效
    if test_config:
        app.config.update(test_config)

    # 初始化扩展
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # 注册蓝图（Blueprint）——暂时只注册已实现的，其余成员完成后逐步添加
    from .auth.routes   import auth_bp
    from .data.routes   import data_bp
    from .index.routes  import index_bp
    from .search        import search_bp
    from .admin         import admin_bp
    from .evaluate      import evaluate_bp

    app.register_blueprint(auth_bp,    url_prefix='/auth')
    app.register_blueprint(data_bp,    url_prefix='/api/datasets')
    app.register_blueprint(index_bp,   url_prefix='/api')
    app.register_blueprint(search_bp,  url_prefix='/api/search')
    app.register_blueprint(admin_bp,   url_prefix='/admin')
    app.register_blueprint(evaluate_bp, url_prefix='/api/evaluate')

    # 健康检查路由
    @app.route('/ping')
    def ping():
        return {'status': 'ok'}

    # 7.1 启动时预加载已有数据集到内存缓存（TESTING 模式跳过）
    if not app.config.get('TESTING', False):
        _preload_datasets(app)

    return app


def _preload_datasets(app) -> None:
    """
    应用启动时，将数据库中所有数据集的向量预加载到内存缓存。
    仅在非测试模式下执行，文件不存在时静默跳过。
    """
    with app.app_context():
        try:
            from .models import Dataset
            from .data.loader import load_dataset, cache_dataset
            import os

            datasets = Dataset.query.all()
            for ds in datasets:
                try:
                    if os.path.exists(ds.file_path):
                        data = load_dataset(ds.file_path)
                        cache_dataset(ds.id, data)
                        print(f'[启动] 预加载数据集 "{ds.name}" '
                              f'({ds.n_cells} 细胞, {ds.n_dims} 维)')
                except Exception as e:
                    print(f'[启动] 数据集 {ds.id} 预加载失败: {e}')
        except Exception as e:
            # 首次启动时表可能不存在，静默处理
            print(f'[启动] 预加载跳过: {e}')
