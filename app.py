"""
Flask + Socket.IO 服务入口。
"""

from backend.app_factory import create_app


app, socketio = create_app()


if __name__ == "__main__":
    """本地开发启动入口。"""
    socketio.run(app, debug=True, port=5001)
