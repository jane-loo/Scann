from app import create_app
from app.models import db, User

app = create_app()

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('已创建默认管理员账号: admin / admin123')
    else:
        print('管理员账号已存在，跳过创建')
    print('数据库初始化完成')
