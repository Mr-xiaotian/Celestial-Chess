# utils_backend.py
import shlex
import time
from typing import List, Dict, Optional
from flask_socketio import SocketIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from celestialchess import ChessGame, BaseAI, MinimaxAI, MCTSAI, MonkyAI


def parse_options(tokens: List[str]):
    """
    简单解析类似 Linux 风格参数:
    --row 3 --col 4 --color 1  -> {"row": "3", "col": "4", "color": "1"}
    """
    opts = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[key] = tokens[i + 1]
                i += 2
            else:
                opts[key] = True
                i += 1
        else:
            i += 1
    return opts


def convert_inf_to_string(value):
    if value == float("inf"):
        return "inf"
    else:
        return value


def prepare_board_for_json(board):
    return [
        [[convert_inf_to_string(cell[0]), cell[1]] for cell in row]
        for row in board
    ]


class GameSession:
    def __init__(
        self,
        socketio: SocketIO,
        game: ChessGame,
        ai_map: Dict[str, BaseAI],
        executor: ThreadPoolExecutor,
    ):
        self.socketio = socketio
        self.game = game
        self.ai_map = ai_map
        self.executor = executor
        self.auto_mode: Optional[str] = None
        self.ai_thinking = False
        self.lock = Lock()
        self.spectator_mode = False
        self.spectator_sleep = 0.0

    def set_auto_mode(self, mode: Optional[str] = None):
        self.auto_mode = mode

    def get_init_state(self):
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
        }

    def get_update_board(self):
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

    def cmd_print(self, msg: str):
        self.socketio.emit("cmd_log", {"msg": msg})

    def emit_config_changed(self, source: str):
        payload = self.get_init_state()
        payload["source"] = source
        self.socketio.emit("config_changed", payload)

    def emit_spectator_status(self, status: str, blue_config=None, red_config=None, sleep=None):
        payload = {"status": status}
        if blue_config is not None:
            payload["blue"] = blue_config
        if red_config is not None:
            payload["red"] = red_config
        if sleep is not None:
            payload["sleep"] = sleep
        self.socketio.emit("spectator_status", payload)

    def sendDataToBackend(self):
        try:
            response = self.get_update_board()
            self.socketio.emit("update_board", response)
        except AssertionError as e:
            self.socketio.emit("update_board", {"error": str(e)})

    def ensure_not_spectator(self):
        if self.spectator_mode:
            self.cmd_print("观战中，无法执行该操作。")
            return False
        return True

    def build_ai_map(self, chess_state, minimax_depth=2, mcts_iter=1000):
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
        self.stop_spectator()
        with self.lock:
            self.game = ChessGame((row_len, col_len), power)
            self.game.init_cfunc()
            self.game.init_history()

            chess_state = ((row_len, col_len), power)
            self.ai_map = self.build_ai_map(chess_state)
            self.auto_mode = None

        self.emit_config_changed(source)
        self.sendDataToBackend()
        self.cmd_print(f"配置已更新: 棋盘={row_len}x{col_len}, power={power}, 来源={source}")
        return self.get_init_state()

    def normalize_ai_config(self, config: Optional[Dict]):
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
        ai_blue = self.create_ai_from_config(blue_config)
        ai_red = self.create_ai_from_config(red_config)

        while self.spectator_mode and not self.game.is_game_over():
            ai = ai_blue if self.game.get_color() == 1 else ai_red
            self.handle_ai_move(ai)

        if self.spectator_mode:
            self.spectator_mode = False
            self.emit_spectator_status("stop")

    def start_spectator(self, blue_config: Optional[Dict], red_config: Optional[Dict], sleep: Optional[float] = None):
        self.set_auto_mode()
        self.spectator_mode = True
        normalized_blue = self.normalize_ai_config(blue_config)
        normalized_red = self.normalize_ai_config(red_config)
        self.spectator_sleep = float(sleep) if sleep is not None else 0.0

        with self.lock:
            self.game.restart()
        self.sendDataToBackend()
        blue_summary = self.format_ai_config_summary(normalized_blue)
        red_summary = self.format_ai_config_summary(normalized_red)
        self.cmd_print(f"观战已开始: 蓝方={blue_summary}, 红方={red_summary}, 等待={self.spectator_sleep}s")
        self.emit_spectator_status("start", normalized_blue, normalized_red, self.spectator_sleep)
        self.executor.submit(self.run_spectator_loop, normalized_blue, normalized_red)

    def stop_spectator(self):
        if self.spectator_mode:
            self.spectator_mode = False
            self.spectator_sleep = 0.0
            self.cmd_print("观战已停止")
            self.emit_spectator_status("stop")

    def format_ai_config_summary(self, config: Dict):
        if config["type"] == "minimax":
            return f"minimax(depth={config['minimax_depth']})"
        if config["type"] == "mcts":
            return f"mcts(iter={config['mcts_iter']})"
        if config["type"] == "monky":
            return "monky"
        return config["type"]

    def apply_move_and_update(self, row: int, col: int, color: int):
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
        if self.game.is_game_over():
            return

        self.ai_thinking = True
        try:
            self.cmd_print(f"({ai.name} thinking...)")
            start_time = time.perf_counter()
            move = ai.find_best_move(self.game)
            self.cmd_print(f"{ai.name}: {ai.msg}")

            if self.spectator_mode and self.spectator_sleep > 0:
                elapsed = time.perf_counter() - start_time
                remaining = self.spectator_sleep - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                if not self.spectator_mode:
                    return

            color = self.game.get_color()
            with self.lock:
                finished = self.apply_move_and_update(move[0], move[1], color)

            if finished:
                ai.end_model()
        finally:
            self.ai_thinking = False

    def handle_ai_auto(self, ai: BaseAI):
        self.cmd_print(f"已开启与 {ai.name} 对弈模式。")
        self.handle_ai_move(ai)

    def handle_play_move(self, data):
        if not self.ensure_not_spectator():
            return
        if self.ai_thinking:
            self.cmd_print("AI 思考中，棋盘已锁定，不能下子。")
            return

        row = data["row"]
        col = data["col"]
        color = data["color"]

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

    def handle_undo_move(self):
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.undo()
        self.sendDataToBackend()
        self.cmd_print("已悔棋。")

    def handle_redo_move(self):
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.redo()
        self.sendDataToBackend()
        self.cmd_print("已重悔。")

    def handle_restart_game(self):
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.restart()
        self.sendDataToBackend()
        self.set_auto_mode()
        self.cmd_print("已重开棋局。")

    def handle_cmd_input(self, data):
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
                self.set_auto_mode("minimax")
                self.handle_ai_auto(self.ai_map["mcts"])
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

        else:
            self.cmd_print(f"未知命令: {cmd}")
            return
