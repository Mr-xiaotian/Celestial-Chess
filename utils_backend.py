# utils_backend.py
import shlex
from typing import List, Dict, Optional
from flask_socketio import SocketIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from celestialchess import ChessGame, BaseAI


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

    def set_auto_mode(self, mode: Optional[str] = None):
        self.auto_mode = mode

    def get_init_state(self, row_len: int, col_len: int, power: int):
        prepared_board = prepare_board_for_json(self.game.chessboard)
        return {
            "power": power,
            "row_len": row_len,
            "col_len": col_len,
            "board": prepared_board,
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

    def sendDataToBackend(self):
        try:
            response = self.get_update_board()
            self.socketio.emit("update_board", response)
        except AssertionError as e:
            self.socketio.emit("update_board", {"error": str(e)})

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
            move = ai.find_best_move(self.game)
            self.cmd_print(f"{ai.name}: {ai.msg}")

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
        with self.lock:
            self.game.undo()
        self.sendDataToBackend()
        self.cmd_print("已悔棋。")

    def handle_redo_move(self):
        with self.lock:
            self.game.redo()
        self.sendDataToBackend()
        self.cmd_print("已重悔。")

    def handle_restart_game(self):
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

        else:
            self.cmd_print(f"未知命令: {cmd}")
            return
