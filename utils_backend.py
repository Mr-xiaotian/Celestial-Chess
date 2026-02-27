# utils_backend.py
from typing import List
from flask_socketio import SocketIO
from threading import Lock
from celestialchess import ChessGame, BaseAI


ai_thinking = False    # 全局变量

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
            # 有值且不是下一个选项
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[key] = tokens[i + 1]
                i += 2
            else:
                # 纯 flag，如 --foo
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


def get_update_board(game: ChessGame):
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    move = game.get_current_move()

    game_over = game.is_game_over()
    winner = game.who_is_winner() if game_over else None

    return {
        "board": prepared_board,
        "move": move,
        "score": score,
        "step": game.step,
        "game_over": game_over,
        "winner": winner,
    }


def handle_ai_move(socketio: SocketIO, ai: BaseAI, game: ChessGame):
    global ai_thinking
    
    # Step 0：设置冻结标志
    ai_thinking = True

    # Step 2：AI 在副本上思考
    cmd_print(socketio, f"({ai.name} thinking...)")
    move = ai.find_best_move(game)
    cmd_print(socketio, f"{ai.name}: {ai.msg}")

    # Step 3：写回到真实棋局
    color = game.get_color()
    finished = apply_move_and_update(
        socketio, game, move[0], move[1], color
    )

    if finished:
        ai.end_model()

    # Step 4：解除冻结标志
    ai_thinking = False


def handle_ai_auto(socketio: SocketIO, ai: BaseAI, game: ChessGame):
    # 通知前端进入对弈模式
    cmd_print(socketio, f"已开启与 {ai.name} 对弈模式。")
    handle_ai_move(socketio, ai, game)


def cmd_print(socketio: SocketIO, msg: str):
    socketio.emit("cmd_log", {"msg": msg})


def apply_move_and_update(socketio: SocketIO, game: ChessGame, row: int, col: int, color: int):
    """
    统一的落子更新流程：更新棋盘、历史、前端、输出日志、检查结束。
    source = "player" 或 "ai"
    """
    game.update_chessboard(row, col, color)
    game.update_history(row, col)

    sendDataToBackend(socketio, game)

    # 打印输出
    msg = f"(Move = ({row}, {col}), Score = {game.get_score()}, Color = {color})"
    cmd_print(socketio, msg)

    # 判断游戏是否结束
    if game.is_game_over():
        cmd_print(socketio, f"Game Over. Winner = {game.who_is_winner()}")
        return True

    return False   # 游戏未结束


def sendDataToBackend(socketio: SocketIO, game):
    try:
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)})