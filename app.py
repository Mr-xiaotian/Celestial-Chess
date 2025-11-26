import shlex
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor

import utils_backend
from celestialchess import ChessGame, MinimaxAI, MCTSAI, MonkyAI #, DeepLearningAI
from utils_backend import sendDataToBackend, prepare_board_for_json, handle_ai_move, handle_ai_auto, cmd_print, parse_options


app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# 创建线程池
executor = ThreadPoolExecutor(max_workers=3)

chess_state = ((11, 11), 3)
(row_len, col_len), power = chess_state  # 棋盘大小，power

game = ChessGame(*chess_state)
game.init_cfunc()
game.init_history()

minimax_ai = MinimaxAI(2, False)
minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
mcts_ai = MCTSAI(1000, complate_mode=True)
monky_ai = MonkyAI()
# dl_ai = DeepLearningAI(r"train\data\models\2025-11-21\dl_model(11-16)(133927).pth", complete_mode=True)

AI_MAP = {
    "minimax": minimax_ai,
    "mcts": mcts_ai,
    "monky": monky_ai
}

minimax_auto = False
mcts_auto = False
monky_auto = False


def set_auto_mode(mode=None):
    global minimax_auto, mcts_auto, monky_auto
    minimax_auto = False
    mcts_auto = False
    monky_auto = False
    if mode == "minimax":
        minimax_auto = True
    elif mode == "mcts":
        mcts_auto = True
    elif mode == "monky":
        monky_auto = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/init_state", methods=["GET"])
def init_state():
    # 获取初始棋局状态
    prepared_board = prepare_board_for_json(game.chessboard)
    
    return jsonify(
        {
            "power": power,
            "row_len": row_len,
            "col_len": col_len,
            "board": prepared_board,
        }
    )


@socketio.on("play_move")
def handle_play_move(data):
    if utils_backend.ai_thinking:
        cmd_print(socketio, "AI 思考中，棋盘已锁定，不能下子。")
        return
    
    row = data["row"]
    col = data["col"]
    color = data["color"]  # 前端发送了颜色信息

    if not game.is_move_valid(row, col):
        cmd_print(socketio, f"非法落子位置: ({row}, {col})")
        return

    game.update_chessboard(row, col, color)
    game.update_history(row, col)
    sendDataToBackend(socketio, game)

    cmd_print(socketio, f"(Move = ({row}, {col}), Score = {game.get_score()}, Color = {color})")

    if game.is_game_over():
        set_auto_mode()
        minimax_ai.end_model()

        cmd_print(socketio, f"Game Over. Winner = {game.who_is_winner()}")
        return

    if minimax_auto:
        executor.submit(handle_ai_move, socketio, minimax_ai, game)
    elif mcts_auto:
        executor.submit(handle_ai_move, socketio, mcts_ai, game)
    elif monky_auto:
        executor.submit(handle_ai_move, socketio, monky_ai, game)


@socketio.on("undo_move")
def handle_undo_move():
    game.undo()
    sendDataToBackend(socketio, game)
    cmd_print(socketio, "已悔棋。")


@socketio.on("redo_move")
def handle_redo_move():
    game.redo()
    sendDataToBackend(socketio, game)
    cmd_print(socketio, "已重悔。")


@socketio.on("restart_game")
def handle_restart_game():
    game.restart()
    sendDataToBackend(socketio, game)
    set_auto_mode()
    cmd_print(socketio, "已重开棋局。")


def make_move_handler(ai_name):
    def handler():
        handle_ai_move(socketio, AI_MAP[ai_name], game)
    return handler


def make_auto_handler(ai_name):
    def handler():
        set_auto_mode(ai_name)
        handle_ai_auto(socketio, AI_MAP[ai_name], game)
    return handler


for name in AI_MAP:
    socketio.on(f"{name}_move")(make_move_handler(name))
    socketio.on(f"{name}_auto")(make_auto_handler(name))


@socketio.on("cmd_input")
def handle_cmd_input(data):
    """
    前端发来：socket.emit('cmd_input', { text: 'celestial play --row 3 --col 4 --color 1' })
    在这里解析并执行相应操作。
    """
    text = data.get("text", "").strip()

    if not text:
        cmd_print(socketio, "空命令。")
        return

    try:
        args = shlex.split(text)
    except ValueError as e:
        socketio.emit("cmd_output", {
            "success": False,
            "message": f"命令解析错误: {e}"
        })
        return

    # 可选的前缀，例如 "celestial"
    if args and args[0] == "celestial":
        args = args[1:]

    if not args:
        cmd_print(socketio, "未指定子命令。")
        return

    cmd = args[0]
    rest = args[1:]

    # --------- play: 落子 ----------
    if cmd == "play":
        opts = parse_options(rest)

        try:
            # 必填: row, col；color 默认为当前行动方
            if "row" not in opts or "col" not in opts:
                raise ValueError("缺少 --row 或 --col")

            row = int(opts["row"])
            col = int(opts["col"])
            color = int(opts.get("color", game.get_color()))
        except Exception as e:
            cmd_print(socketio, f"play 参数错误: {e}")
            return

        handle_play_move({"row": row, "col": col, "color": color})
        return

    # --------- 基础操作：undo / redo / restart ----------
    elif cmd == "undo":
        handle_undo_move()
        return

    elif cmd == "redo":
        handle_redo_move()
        return

    elif cmd == "restart":
        handle_restart_game()
        return

    # --------- 单步 AI 执棋：ai minimax/mcts/monky ----------
    elif cmd == "ai":
        if not rest:
            cmd_print(socketio, "ai 后需要指定类型: minimax / mcts / monky")
            return

        ai_type = rest[0]

        if game.is_game_over():
            cmd_print(socketio, "游戏已结束，无法继续 AI 执棋。")
            return

        if ai_type == "minimax":
            executor.submit(handle_ai_move, (socketio, minimax_ai, game))
        elif ai_type == "mcts":
            executor.submit(handle_ai_move, (socketio, mcts_ai, game))
        elif ai_type == "monky":
            executor.submit(handle_ai_move, (socketio, monky_ai, game))
        else:
            cmd_print(socketio, f"未知 AI 类型: {ai_type}")
            return

        return

    # --------- 自动对弈：auto minimax/mcts/monky ----------
    elif cmd == "auto":
        if not rest:
            cmd_print(socketio, "auto 后需要指定类型: minimax / mcts / monky")
            return

        ai_type = rest[0]

        if ai_type == "minimax":
            set_auto_mode("minimax")
            handle_ai_auto(socketio, minimax_ai, game)
        elif ai_type == "mcts":
            set_auto_mode("mcts")
            handle_ai_auto(socketio, mcts_ai, game)
        elif ai_type == "monky":
            set_auto_mode("minimax")
            handle_ai_auto(socketio, mcts_ai, game)
        else:
            cmd_print(socketio, f"未知 auto 类型: {ai_type}")

        return

    # --------- 未知命令 ----------
    else:
        cmd_print(socketio, f"未知命令: {cmd}")
        return

    # 你想要的逻辑都可以写在这里
    # 比如调用 AI、查看棋盘、运行 debug 命令、执行脚本等

    # 也可以回消息
    # cmd_print(socketio, f"Echo: {text}")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
