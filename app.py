"""
Flask + Socket.IO 服务入口。

该模块负责：
1. 创建 Flask 应用与 Socket.IO 实例；
2. 初始化棋局与内置 AI；
3. 绑定 HTTP 路由（页面渲染、初始化状态、配置变更）；
4. 绑定 Socket 事件（落子、悔棋、AI 执棋、CMD 输入、观战控制）；
5. 将具体业务逻辑委派给 GameSession。
"""

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor

from celestialchess import ChessGame, MinimaxAI, MCTSAI, MonkyAI #, DeepLearningAI
import utils_backend


def create_app():
    """
    创建并装配应用对象。

    返回值:
        tuple[Flask, SocketIO]:
            - Flask 应用实例
            - 绑定到该应用的 SocketIO 实例
    """
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading")
    executor = ThreadPoolExecutor(max_workers=3)

    chess_state = ((11, 11), 3)

    game = ChessGame(*chess_state)
    game.init_cfunc()
    game.init_history()

    minimax_ai = MinimaxAI(2, False)
    minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
    mcts_ai = MCTSAI(1000, complate_mode=True)
    monky_ai = MonkyAI()

    ai_map = {
        "minimax": minimax_ai,
        "mcts": mcts_ai,
        "monky": monky_ai
    }

    session = utils_backend.GameSession(socketio, game, ai_map, executor)

    @app.route("/")
    def index():
        """渲染主页面。"""
        return render_template("index.html")

    @app.route("/init_state", methods=["GET"])
    def init_state():
        """返回前端初始化所需的完整棋局状态。"""
        session.schedule_human_analysis_if_needed()
        return jsonify(session.get_init_state())

    @app.route("/configure", methods=["POST"])
    def configure():
        """
        更新棋盘配置。

        请求体:
            {
                "row_len": int,
                "col_len": int,
                "power": int
            }

        返回:
            - 成功: 更新后的初始化状态
            - 失败: {"error": "invalid config"}, 400
        """
        data = request.get_json(silent=True) or {}
        try:
            row_len = int(data.get("row_len"))
            col_len = int(data.get("col_len"))
            power = int(data.get("power"))
            if row_len <= 0 or col_len <= 0 or power <= 1:
                raise ValueError("invalid config")
        except Exception:
            return jsonify({"error": "invalid config"}), 400

        return jsonify(session.configure_game(row_len, col_len, power, "panel"))

    @socketio.on("play_move")
    def handle_play_move(data):
        """处理前端落子事件。"""
        session.handle_play_move(data)

    @socketio.on("undo_move")
    def handle_undo_move():
        """处理悔棋事件。"""
        session.handle_undo_move()

    @socketio.on("redo_move")
    def handle_redo_move():
        """处理重做事件。"""
        session.handle_redo_move()

    @socketio.on("restart_game")
    def handle_restart_game():
        """处理重开棋局事件。"""
        session.handle_restart_game()

    def make_move_handler(ai_name):
        """
        生成指定 AI 的单步执棋事件处理函数。

        参数:
            ai_name (str): AI 名称键（如 minimax / mcts / monky）。
        """
        def handler():
            """执行单步 AI 执棋，观战模式下拒绝执行。"""
            if not session.ensure_not_spectator():
                return
            session.handle_ai_move(session.ai_map[ai_name])
        return handler

    def make_auto_handler(ai_name):
        """
        生成指定 AI 的自动对弈事件处理函数。

        参数:
            ai_name (str): AI 名称键（如 minimax / mcts / monky）。
        """
        def handler():
            """切换自动模式并立即触发 AI 回合。"""
            if not session.ensure_not_spectator():
                return
            session.set_auto_mode(ai_name)
            session.handle_ai_auto(session.ai_map[ai_name])
        return handler

    for name in ai_map:
        socketio.on(f"{name}_move")(make_move_handler(name))
        socketio.on(f"{name}_auto")(make_auto_handler(name))

    @socketio.on("cmd_input")
    def handle_cmd_input(data):
        """处理 CMD 输入事件。"""
        session.handle_cmd_input(data)

    @socketio.on("start_spectator")
    def handle_start_spectator(data):
        """
        处理启动观战事件。

        参数:
            data (dict): 包含 blue/red AI 配置及 sleep 间隔。
        """
        data = data or {}
        session.start_spectator(data.get("blue"), data.get("red"), data.get("sleep"))

    @socketio.on("stop_spectator")
    def handle_stop_spectator():
        """处理停止观战事件。"""
        session.stop_spectator()

    @socketio.on("apply_roles")
    def handle_apply_roles(data):
        """
        应用红蓝方角色配置（PVP / PVE / AI 对战）。

        参数:
            data (dict): 包含 blue/red 角色配置与 sleep。
        """
        data = data or {}
        session.apply_role_config(data.get("blue"), data.get("red"), data.get("sleep"), "panel")

    return app, socketio


app, socketio = create_app()


if __name__ == "__main__":
    """本地开发启动入口。"""
    socketio.run(app, debug=True, port=5001)
