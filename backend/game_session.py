"""
后端会话层与命令调度工具。

该模块封装了棋局会话的核心控制逻辑，负责在 Flask-SocketIO 环境中：
1. 维护单局游戏状态与线程安全访问；
2. 构建与切换 AI 实例；
3. 处理前端 Socket 指令与 CMD 文本命令；
4. 推送棋盘更新、配置变化与观战状态；
5. 管理自动模式与观战循环。
"""

import shlex
import time
from typing import Dict, Optional
from flask_socketio import SocketIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from celestialchess import ChessGame, BaseAI, MinimaxAI, MCTSAI, MonkyAI

from .session_utils import parse_options, prepare_board_for_json


class GameSession:
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
        self.analysis_iter = 800
        self.analysis_seq = 0

    def set_auto_mode(self, mode: Optional[str] = None):
        """
        设置自动对弈模式。

        参数:
            mode: 目标 AI 名称；传 None 表示关闭自动模式。
        """
        self.auto_mode = mode

    def normalize_role_side_config(self, config: Optional[Dict]):
        """
        归一化单侧角色配置。

        参数:
            config: 角色配置，可为 human 或具体 AI 类型配置。

        返回:
            dict: 标准结构 {"role": str, "ai": dict}。
        """
        role = "human"
        ai_config = self.normalize_ai_config(None)
        ai_types = {"minimax", "mcts", "monky"}
        if isinstance(config, dict):
            raw_role = str(config.get("role") or "human").lower()
            if raw_role in ai_types:
                role = raw_role
            elif raw_role == "ai":
                role = "mcts"
            elif raw_role == "human":
                role = "human"
            raw_ai_config = config.get("ai") if isinstance(config.get("ai"), dict) else config
            ai_config = self.normalize_ai_config(raw_ai_config)
            if role != "human":
                ai_config["type"] = role
        return {"role": role, "ai": ai_config}

    def detect_mode(self):
        """
        根据红蓝两侧角色推断当前对局模式。

        返回:
            str: pvp / pve / spectator。
        """
        blue_ai = self.side_role_config["Blue"]["role"] != "human"
        red_ai = self.side_role_config["Red"]["role"] != "human"
        if blue_ai and red_ai:
            return "spectator"
        if blue_ai or red_ai:
            return "pve"
        return "pvp"

    def get_role_status_payload(self):
        """
        构造角色状态广播载荷。

        返回:
            dict: 包含 mode、blue、red、sleep。
        """
        return {
            "mode": self.detect_mode(),
            "blue": self.side_role_config["Blue"],
            "red": self.side_role_config["Red"],
            "sleep": self.spectator_sleep,
        }

    def emit_role_status(self):
        """向前端广播角色状态。"""
        self.socketio.emit("role_status", self.get_role_status_payload())

    def refresh_side_ai_map(self):
        """根据角色配置重建红蓝方 AI 实例缓存。"""
        self.side_ai_map = {"Blue": None, "Red": None}
        for side in ("Blue", "Red"):
            side_config = self.side_role_config[side]
            if side_config["role"] != "human":
                self.side_ai_map[side] = self.create_ai_from_config(side_config["ai"])

    def schedule_role_ai_turn_if_needed(self):
        """
        在当前回合调度“AI 执棋或人类分析”。

        规则:
            - AI 回合：提交 AI 执棋任务；
            - Human 回合：提交局面分析任务。
        """
        if self.ai_thinking or self.game.is_game_over():
            return
        side = "Blue" if self.game.get_color() == 1 else "Red"
        if self.side_role_config[side]["role"] == "human":
            self.schedule_human_analysis_if_needed()
            return
        ai = self.side_ai_map.get(side)
        if ai is None:
            ai = self.create_ai_from_config(self.side_role_config[side]["ai"])
            self.side_ai_map[side] = ai
        self.executor.submit(self.handle_ai_move, ai)

    def schedule_human_analysis_if_needed(self):
        """
        当轮到人类方时异步触发 MCTS 分析。

        会先发送 pending 状态，再在后台线程完成分析并推送结果。
        """
        if self.ai_thinking or self.game.is_game_over():
            self.socketio.emit("analysis_update", {"status": "idle"})
            return
        side = "Blue" if self.game.get_color() == 1 else "Red"
        if self.side_role_config[side]["role"] != "human":
            self.socketio.emit("analysis_update", {"status": "idle"})
            return
        with self.lock:
            game_snapshot = self.game.copy()
        self.analysis_seq += 1
        seq = self.analysis_seq
        self.socketio.emit("analysis_update", {"status": "pending", "turn": side})
        self.executor.submit(self.run_human_analysis, seq, game_snapshot, side)

    def run_human_analysis(self, seq: int, game_snapshot: ChessGame, side: str):
        """
        执行单次人类回合分析任务并推送 analysis_update。

        参数:
            seq: 分析序号，用于丢弃过期结果。
            game_snapshot: 分析用棋局快照。
            side: 当前轮到的执棋方（Blue/Red）。
        """
        payload = {"status": "idle"}
        try:
            if game_snapshot.is_game_over():
                payload = {"status": "idle"}
            else:
                analyzer = MCTSAI(self.analysis_iter, complate_mode=False)
                analysis = analyzer.analyze_position(game_snapshot)
                payload = {
                    "status": "ready",
                    "turn": side,
                    "current_win_rate": analysis.get("current_win_rate"),
                    "moves": analysis.get("moves", []),
                }
        except Exception as e:
            payload = {"status": "error", "message": str(e)}
        if seq != self.analysis_seq:
            return
        self.socketio.emit("analysis_update", payload)

    def apply_role_config(self, blue_config: Optional[Dict], red_config: Optional[Dict], sleep: Optional[float], source: str):
        """
        应用红蓝双方角色配置并刷新模式状态。

        参数:
            blue_config: 蓝方角色配置。
            red_config: 红方角色配置。
            sleep: AI 对战回合间隔秒数。
            source: 触发来源标识。
        """
        normalized_blue = self.normalize_role_side_config(blue_config)
        normalized_red = self.normalize_role_side_config(red_config)
        try:
            normalized_sleep = float(sleep) if sleep is not None else 0.0
        except (TypeError, ValueError):
            normalized_sleep = 0.0
        self.side_role_config["Blue"] = normalized_blue
        self.side_role_config["Red"] = normalized_red
        self.spectator_sleep = max(normalized_sleep, 0.0)
        self.spectator_mode = self.detect_mode() == "spectator"
        self.refresh_side_ai_map()
        if self.spectator_mode:
            with self.lock:
                self.game.restart()
            self.sendDataToBackend()
        self.emit_role_status()
        blue_summary = "human" if normalized_blue["role"] == "human" else self.format_ai_config_summary(normalized_blue["ai"])
        red_summary = "human" if normalized_red["role"] == "human" else self.format_ai_config_summary(normalized_red["ai"])
        self.cmd_print(f"角色已更新: 蓝方={blue_summary}, 红方={red_summary}, 模式={self.detect_mode()}, 等待={self.spectator_sleep}s")
        self.schedule_role_ai_turn_if_needed()

    def get_init_state(self):
        """
        获取前端初始化所需的完整状态快照。

        返回:
            dict: 包含棋盘尺寸、棋力、棋盘内容、最近一步、分数、步数。
        """
        row_len, col_len = self.game.board_range
        power = self.game.power
        prepared_board = prepare_board_for_json(self.game.chessboard)
        move = self.game.get_current_move()
        score = self.game.get_score()
        step = self.game.step
        return {
            "power": power,
            "row_len": row_len,
            "col_len": col_len,
            "board": prepared_board,
            "move": move,
            "score": score,
            "step": step,
            "mode": self.detect_mode(),
            "blue": self.side_role_config["Blue"],
            "red": self.side_role_config["Red"],
            "sleep": self.spectator_sleep,
        }

    def get_update_board(self):
        """
        获取棋盘增量广播数据。

        返回:
            dict: 包含棋盘、分数、当前步、游戏结束与胜者信息。
        """
        prepared_board = prepare_board_for_json(self.game.chessboard)
        score = self.game.get_score()
        move = self.game.get_current_move()

        game_over = self.game.is_game_over()
        winner = self.game.who_is_winner() if game_over else None

        return {
            "board": prepared_board,
            "move": move,
            "score": score,
            "step": self.game.step,
            "game_over": game_over,
            "winner": winner,
        }

    def cmd_print(self, msg: str, msg_type: str = "system"):
        """
        向前端 CMD 面板发送日志。

        参数:
            msg: 文本内容。
            msg_type: 日志类型（用于前端样式分类）。
        """
        self.socketio.emit("cmd_log", {"msg": msg, "type": msg_type})

    def resolve_ai_msg_type(self, ai: BaseAI, color: int):
        """
        根据 AI 名称与执棋方生成前端日志类型。

        参数:
            ai: AI 实例。
            color: 执棋方（1=蓝方，-1=红方）。

        返回:
            str: 例如 ai-minimax-blue / ai-mcts-red。
        """
        side = "blue" if color == 1 else "red"
        ai_name = ai.name.lower()
        if "minimax" in ai_name:
            return f"ai-minimax-{side}"
        if "mcts" in ai_name:
            return f"ai-mcts-{side}"
        if "monky" in ai_name:
            return f"ai-monky-{side}"
        if "deep" in ai_name or "learning" in ai_name:
            return f"ai-deeplearning-{side}"
        return f"ai-generic-{side}"

    def emit_config_changed(self, source: str):
        """
        广播“配置已变化”事件。

        参数:
            source: 触发来源标识（如 panel / cmd）。
        """
        payload = self.get_init_state()
        payload["source"] = source
        self.socketio.emit("config_changed", payload)

    def emit_spectator_status(self, status: str, blue_config=None, red_config=None, sleep=None):
        """
        广播观战模式状态。

        参数:
            status: 状态字符串（start / stop）。
            blue_config: 蓝方 AI 配置（可选）。
            red_config: 红方 AI 配置（可选）。
            sleep: 观战回合间隔（可选）。
        """
        payload = {"status": status}
        if blue_config is not None:
            payload["blue"] = blue_config
        if red_config is not None:
            payload["red"] = red_config
        if sleep is not None:
            payload["sleep"] = sleep
        self.socketio.emit("spectator_status", payload)

    def sendDataToBackend(self):
        """
        向前端广播棋盘更新。

        如果构造更新数据时发生断言异常，会回传错误消息。
        """
        try:
            response = self.get_update_board()
            self.socketio.emit("update_board", response)
        except AssertionError as e:
            self.socketio.emit("update_board", {"error": str(e)})

    def ensure_not_spectator(self):
        """
        校验当前是否处于观战模式。

        返回:
            bool:
                - False: 观战中，不允许执行手动操作；
                - True: 非观战状态，可继续执行。
        """
        if self.spectator_mode:
            self.cmd_print("观战中，无法执行该操作。")
            return False
        return True

    def build_ai_map(self, chess_state, minimax_depth=2, mcts_iter=1000):
        """
        构建标准 AI 映射表。

        参数:
            chess_state: 棋局状态定义，供部分 AI 初始化使用。
            minimax_depth: Minimax 搜索深度。
            mcts_iter: MCTS 迭代次数。
        """
        minimax_ai = MinimaxAI(minimax_depth, False)
        minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
        mcts_ai = MCTSAI(mcts_iter, complate_mode=True)
        monky_ai = MonkyAI()
        return {
            "minimax": minimax_ai,
            "mcts": mcts_ai,
            "monky": monky_ai
        }

    def configure_game(self, row_len: int, col_len: int, power: int, source: str):
        """
        重新配置并重建棋局。

        行为:
            1. 停止观战；
            2. 新建游戏对象并初始化历史；
            3. 重建 AI 映射；
            4. 广播配置变更与棋盘状态。

        返回:
            dict: 新棋局初始化状态。
        """
        with self.lock:
            self.game = ChessGame((row_len, col_len), power)
            self.game.init_cfunc()
            self.game.init_history()

            chess_state = ((row_len, col_len), power)
            self.ai_map = self.build_ai_map(chess_state)
            self.auto_mode = None
            self.spectator_mode = self.detect_mode() == "spectator"
            self.refresh_side_ai_map()

        self.emit_config_changed(source)
        self.emit_role_status()
        self.sendDataToBackend()
        self.cmd_print(f"配置已更新: 棋盘={row_len}x{col_len}, power={power}, 来源={source}")
        self.schedule_role_ai_turn_if_needed()
        return self.get_init_state()

    def normalize_ai_config(self, config: Optional[Dict]):
        """
        归一化 AI 配置，补齐缺省值并转换类型。

        参数:
            config: 原始配置字典（可为空）。

        返回:
            dict: 标准化后的配置。
        """
        ai_type = "mcts"
        minimax_depth = 2
        mcts_iter = 1000
        if isinstance(config, dict):
            if config.get("type"):
                ai_type = str(config.get("type")).lower()
            if config.get("minimax_depth") is not None:
                minimax_depth = int(config.get("minimax_depth"))
            if config.get("mcts_iter") is not None:
                mcts_iter = int(config.get("mcts_iter"))
        return {
            "type": ai_type,
            "minimax_depth": minimax_depth,
            "mcts_iter": mcts_iter
        }

    def create_ai_from_config(self, config: Dict):
        """
        根据标准化配置创建 AI 实例。

        参数:
            config: AI 配置字典。

        返回:
            BaseAI: 对应的 AI 对象。
        """
        row_len, col_len = self.game.board_range
        chess_state = ((row_len, col_len), self.game.power)
        if config["type"] == "minimax":
            ai = MinimaxAI(config["minimax_depth"], False)
            ai.set_transposition_mode(chess_state, r"transposition_table/")
            return ai
        if config["type"] == "mcts":
            return MCTSAI(config["mcts_iter"], complate_mode=True)
        if config["type"] == "monky":
            return MonkyAI()
        return MCTSAI(config["mcts_iter"], complate_mode=True)

    def run_spectator_loop(self, blue_config: Dict, red_config: Dict):
        """
        观战循环主逻辑。

        根据当前执棋方轮流调用蓝方或红方 AI，直到游戏结束
        或观战模式被外部停止。
        """
        ai_blue = self.create_ai_from_config(blue_config)
        ai_red = self.create_ai_from_config(red_config)

        while self.spectator_mode and not self.game.is_game_over():
            ai = ai_blue if self.game.get_color() == 1 else ai_red
            self.handle_ai_move(ai)

        if self.spectator_mode:
            self.spectator_mode = False
            self.emit_spectator_status("stop")

    def start_spectator(self, blue_config: Optional[Dict], red_config: Optional[Dict], sleep: Optional[float] = None):
        """
        启动观战模式。

        参数:
            blue_config: 蓝方 AI 配置。
            red_config: 红方 AI 配置。
            sleep: 每步最小间隔秒数（可选）。
        """
        self.set_auto_mode()
        blue_ai = self.normalize_ai_config(blue_config)
        red_ai = self.normalize_ai_config(red_config)
        blue_role = {"role": blue_ai["type"], "ai": blue_ai}
        red_role = {"role": red_ai["type"], "ai": red_ai}
        self.apply_role_config(blue_role, red_role, sleep, "spectator")
        self.emit_spectator_status("start", blue_role["ai"], red_role["ai"], self.spectator_sleep)

    def stop_spectator(self):
        """停止观战模式并广播状态。"""
        if self.detect_mode() == "spectator":
            self.apply_role_config({"role": "human"}, {"role": "human"}, 0.0, "spectator")
            self.cmd_print("观战已停止")
            self.emit_spectator_status("stop")

    def format_ai_config_summary(self, config: Dict):
        """
        生成人类可读的 AI 配置摘要。

        参数:
            config: AI 配置字典。
        """
        if config["type"] == "minimax":
            return f"minimax(depth={config['minimax_depth']})"
        if config["type"] == "mcts":
            return f"mcts(iter={config['mcts_iter']})"
        if config["type"] == "monky":
            return "monky"
        return config["type"]

    def apply_move_and_update(self, row: int, col: int, color: int):
        """
        在游戏对象中应用落子并推送更新。

        参数:
            row: 落子行索引。
            col: 落子列索引。
            color: 执棋颜色（1/-1）。

        返回:
            bool: 是否在该步后进入终局。
        """
        self.game.update_chessboard(row, col, color)
        self.game.update_history(row, col)

        self.sendDataToBackend()

        msg = f"(Move = ({row}, {col}), Score = {self.game.get_score()}, Color = {color})"
        self.cmd_print(msg)

        if self.game.is_game_over():
            self.cmd_print(f"Game Over. Winner = {self.game.who_is_winner()}")
            return True

        return False

    def handle_ai_move(self, ai: BaseAI):
        """
        执行一次 AI 回合。

        该方法负责：
            - 打印 AI 思考日志；
            - 调用 AI 搜索落子；
            - 根据观战延时策略补齐等待；
            - 应用落子并处理终局收尾。
        """
        if self.game.is_game_over():
            return

        self.ai_thinking = True
        self.socketio.emit("ai_thinking", {"status": "start"})
        try:
            color = self.game.get_color()
            ai_msg_type = self.resolve_ai_msg_type(ai, color)
            self.cmd_print(f"({ai.name} thinking...)", ai_msg_type)
            start_time = time.perf_counter()
            move = ai.find_best_move(self.game)
            self.cmd_print(f"{ai.name}: {ai.msg}", ai_msg_type)

            if self.spectator_mode and self.spectator_sleep > 0:
                elapsed = time.perf_counter() - start_time
                remaining = self.spectator_sleep - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                if not self.spectator_mode:
                    return

            with self.lock:
                finished = self.apply_move_and_update(move[0], move[1], color)

            if finished:
                ai.end_model()
        finally:
            self.ai_thinking = False
            self.socketio.emit("ai_thinking", {"status": "stop"})
            self.schedule_role_ai_turn_if_needed()

    def handle_ai_auto(self, ai: BaseAI):
        """
        进入与指定 AI 的自动对弈流程。

        参数:
            ai: 对手 AI 实例。
        """
        self.cmd_print(f"已开启与 {ai.name} 对弈模式。")
        self.handle_ai_move(ai)

    def handle_play_move(self, data):
        """
        处理前端手动落子请求。

        参数:
            data: 包含 row/col/color 的字典。
        """
        if not self.ensure_not_spectator():
            return
        if self.ai_thinking:
            self.cmd_print("AI 思考中，棋盘已锁定，不能下子。")
            return

        row = data["row"]
        col = data["col"]
        color = data["color"]
        side = "Blue" if color == 1 else "Red"
        if self.side_role_config[side]["role"] != "human":
            self.cmd_print(f"{side} 方由 AI 执棋，当前不允许手动落子。")
            return

        if not self.game.is_move_valid(row, col):
            self.cmd_print(f"非法落子位置: ({row}, {col})")
            return

        with self.lock:
            finished = self.apply_move_and_update(row, col, color)

        if finished:
            self.set_auto_mode()
            minimax_ai = self.ai_map.get("minimax")
            if minimax_ai is not None:
                minimax_ai.end_model()
            return

        if self.auto_mode in self.ai_map:
            self.executor.submit(self.handle_ai_move, self.ai_map[self.auto_mode])
            return
        self.schedule_role_ai_turn_if_needed()

    def handle_undo_move(self):
        """处理悔棋请求并广播更新。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.undo()
        self.sendDataToBackend()
        self.cmd_print("已悔棋。")
        self.schedule_role_ai_turn_if_needed()

    def handle_redo_move(self):
        """处理重做请求并广播更新。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.redo()
        self.sendDataToBackend()
        self.cmd_print("已重悔。")
        self.schedule_role_ai_turn_if_needed()

    def handle_restart_game(self):
        """处理重开棋局请求并清空自动模式。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.restart()
        self.sendDataToBackend()
        self.set_auto_mode()
        self.cmd_print("已重开棋局。")
        self.schedule_role_ai_turn_if_needed()

    def handle_cmd_input(self, data):
        """
        处理 CMD 文本输入并分派子命令。

        支持命令:
            play / undo / redo / restart / ai / auto / config / help
        """
        text = data.get("text", "").strip()

        if not text:
            self.cmd_print("空命令。")
            return

        try:
            args = shlex.split(text)
        except ValueError as e:
            self.socketio.emit("cmd_output", {
                "success": False,
                "message": f"命令解析错误: {e}"
            })
            return

        if args and args[0] == "celestial":
            args = args[1:]

        if not args:
            self.cmd_print("未指定子命令。")
            return

        cmd = args[0]
        rest = args[1:]

        if cmd == "play":
            opts = parse_options(rest)

            try:
                if "row" not in opts or "col" not in opts:
                    raise ValueError("缺少 --row 或 --col")

                row = int(opts["row"])
                col = int(opts["col"])
                color = int(opts.get("color", self.game.get_color()))
            except Exception as e:
                self.cmd_print(f"play 参数错误: {e}")
                return

            self.handle_play_move({"row": row, "col": col, "color": color})
            return

        elif cmd == "undo":
            self.handle_undo_move()
            return

        elif cmd == "redo":
            self.handle_redo_move()
            return

        elif cmd == "restart":
            self.handle_restart_game()
            return

        elif cmd == "ai":
            if not rest:
                self.cmd_print("ai 后需要指定类型: minimax / mcts / monky")
                return

            ai_type = rest[0]

            if self.game.is_game_over():
                self.cmd_print("游戏已结束，无法继续 AI 执棋。")
                return

            if ai_type == "minimax":
                self.executor.submit(self.handle_ai_move, self.ai_map["minimax"])
            elif ai_type == "mcts":
                self.executor.submit(self.handle_ai_move, self.ai_map["mcts"])
            elif ai_type == "monky":
                self.executor.submit(self.handle_ai_move, self.ai_map["monky"])
            else:
                self.cmd_print(f"未知 AI 类型: {ai_type}")
                return

            return

        elif cmd == "auto":
            if not rest:
                self.cmd_print("auto 后需要指定类型: minimax / mcts / monky")
                return

            ai_type = rest[0]

            if ai_type == "minimax":
                self.set_auto_mode("minimax")
                self.handle_ai_auto(self.ai_map["minimax"])
            elif ai_type == "mcts":
                self.set_auto_mode("mcts")
                self.handle_ai_auto(self.ai_map["mcts"])
            elif ai_type == "monky":
                self.set_auto_mode("monky")
                self.handle_ai_auto(self.ai_map["monky"])
            else:
                self.cmd_print(f"未知 auto 类型: {ai_type}")

            return

        elif cmd == "config":
            if not rest:
                self.cmd_print("config 用法: config board --row N --col N --power N | config spectator --start/--stop")
                return

            mode = rest[0] if rest[0] and not rest[0].startswith("--") else "board"
            opts = parse_options(rest[1:] if mode != "board" else rest)

            if mode == "board":
                try:
                    row = int(opts.get("row"))
                    col = int(opts.get("col"))
                    power = int(opts.get("power"))
                    if row <= 0 or col <= 0 or power <= 1:
                        raise ValueError("invalid config")
                except Exception:
                    self.cmd_print("配置参数错误: 需要 --row/--col/--power")
                    return

                self.configure_game(row, col, power, "cmd")
                return

            if mode == "spectator":
                if "stop" in opts or "stop" in rest:
                    self.stop_spectator()
                    return

                blue = {
                    "type": opts.get("blue") or opts.get("blue-type") or "mcts",
                    "minimax_depth": opts.get("blue-depth") or opts.get("blue_depth"),
                    "mcts_iter": opts.get("blue-iter") or opts.get("blue_iter"),
                }
                red = {
                    "type": opts.get("red") or opts.get("red-type") or "mcts",
                    "minimax_depth": opts.get("red-depth") or opts.get("red_depth"),
                    "mcts_iter": opts.get("red-iter") or opts.get("red_iter"),
                }
                sleep = opts.get("sleep") or opts.get("delay")
                self.start_spectator(blue, red, sleep)
                return

            self.cmd_print("config 用法: config board --row N --col N --power N | config spectator --start/--stop")
            return

        elif cmd == 'help':
            self.cmd_print(
"""可用命令:
play --row N --col N --color C  播放移动 (N: 行/列索引, C: 颜色 0/1)
undo  撤销上一步
redo  重做上一步
restart  重新开始游戏
ai minimax/mcts/monky  执行指定 AI 类型的移动
auto minimax/mcts/monky  自动执行指定 AI 类型的移动
config board --row N --col N --power N  配置游戏板 (N: 行/列数, P: 棋力)
config spectator --start/--stop  启动/停止观众模式
help  显示帮助信息"""
            )
            return

        else:
            self.cmd_print(f"未知命令: {cmd}, 请输入 'help' 查看可用命令")
            return
