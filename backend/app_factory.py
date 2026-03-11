"""
Flask + Socket.IO 应用工厂。
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO

from celestialchess import ChessGame, MinimaxAI, MCTSAI, MonkyAI
from backend.game_session import GameSession
from backend.http_routes import register_http_routes
from backend.socket_events import register_socket_events


def create_app():
    """
    创建并装配应用对象。

    返回值:
        tuple[Flask, SocketIO]:
            - Flask 应用实例
            - 绑定到该应用的 SocketIO 实例
    """
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    socketio = SocketIO(app, async_mode="threading")
    executor = ThreadPoolExecutor(max_workers=3)

    chess_state = ((11, 11), 3)

    game = ChessGame(*chess_state)
    game.init_cfunc()
    game.init_history()

    minimax_ai = MinimaxAI(2, False)
    minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
    mcts_ai = MCTSAI(1000)
    monky_ai = MonkyAI()

    ai_map = {
        "minimax": minimax_ai,
        "mcts": mcts_ai,
        "monky": monky_ai,
    }

    session = GameSession(socketio, game, ai_map, executor)
    register_http_routes(app, session)
    register_socket_events(socketio, session)

    return app, socketio
