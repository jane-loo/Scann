"""初始化数据库、演示数据与测试账号（等价于 reset_all.py）。"""
import runpy

if __name__ == '__main__':
    runpy.run_path('reset_all.py', run_name='__main__')
