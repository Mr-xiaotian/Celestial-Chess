"""
后端会话控制器。

该模块只保留会话组装与共享状态定义，具体能力拆分为多个 mixin：
1. lifecycle_mixin: 角色/模式/分析调度/状态快照；
2. gameplay_mixin: 对局推进/AI 执棋/配置变更；
3. command_mixin: CMD 命令解析与分发。
"""

from typing import Dict, Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from flask_socketio import SocketIO

from celestialchess import ChessGame, BaseAI
from .session import SessionLifecycleMixin, SessionGameplayMixin, SessionCommandMixin


class GameSession(SessionLifecycleMixin, SessionGameplayMixin, SessionCommandMixin):
    """
    单局游戏会话控制器。

    该类聚合了游戏状态、AI 策略、线程池与 Socket 推送能力，
    为 app.py 提供统一的后端业务入口。
    """

    def __init__(
        self,
        socketio: SocketIO,
        game: ChessGame,
        ai_map: Dict[str, BaseAI],
        executor: ThreadPoolExecutor,
    ):
        """
        初始化会话对象。

        参数:
            socketio: 用于事件推送的 SocketIO 实例。
            game: 当前棋局对象。
            ai_map: 可用 AI 映射表，键为 AI 名称。
            executor: 异步任务执行器（用于 AI 执棋与观战循环）。
        """
        self.socketio = socketio
        self.game = game
        self.ai_map = ai_map
        self.executor = executor
        self.auto_mode: Optional[str] = None
        self.ai_thinking = False
        self.lock = Lock()
        self.spectator_mode = False
        self.spectator_sleep = 0.0
        self.side_role_config = {
            "Blue": self.normalize_role_side_config({"role": "human"}),
            "Red": self.normalize_role_side_config({"role": "human"}),
        }
        self.side_ai_map: Dict[str, Optional[BaseAI]] = {"Blue": None, "Red": None}
        self.analysis_iter = 1500
        self.analysis_seq = 0

    def set_auto_mode(self, mode: Optional[str] = None):
        """
        设置自动对弈模式。

        参数:
            mode: 目标 AI 名称；传 None 表示关闭自动模式。
        """
        self.auto_mode = mode
