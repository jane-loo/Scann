from app import create_app
from app.models import db, User

app = create_app()

with app.app_context():
    db.create_all()
    
    # 初始化 5 个不同职位的用户
    test_users = [
        {'username': 'visitor', 'email': 'visitor@example.com', 'role': 'visitor'},
        {'username': 'normal', 'email': 'normal@example.com', 'role': 'normal'},
        {'username': 'expert', 'email': 'expert@example.com', 'role': 'expert'},
        {'username': 'labadmin', 'email': 'labadmin@example.com', 'role': 'labadmin'},
        {'username': 'sysadmin', 'email': 'sysadmin@example.com', 'role': 'sysadmin'},
    ]

    for user_info in test_users:
        if not User.query.filter_by(username=user_info['username']).first():
            user = User(
                username=user_info['username'], 
                email=user_info['email'], 
                role=user_info['role']
            )
            user.set_password('pass123')
            db.session.add(user)
            print(f"已创建账号: {user_info['username']} / pass123")
    
    # 兼容原有的 admin 账号
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', role='sysadmin')
        admin.set_password('admin123')
        db.session.add(admin)
        print('已创建默认管理员账号: admin / admin123')

    # 将第一个数据集设为演示数据（允许所有人可见）
    from app.models import Dataset
    first_ds = Dataset.query.first()
    if first_ds:
        first_ds.upload_by = None
        print(f"已将数据集 {first_ds.name} 设为演示数据")

    db.session.commit()
    print('数据库初始化完成')
