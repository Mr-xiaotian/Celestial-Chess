import shlex
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor

from celestialchess import ChessGame, BaseAI, MinimaxAI, MCTSAI, MonkyAI #, DeepLearningAI

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


def handle_ai_move(ai: BaseAI, game: ChessGame):
    # MCTSAI执棋
    cmd_print(f"({ai.name} thinking...)")

    color = game.get_color()
    move = ai.find_best_move(game)
    cmd_print(f"{ai.name}: {ai.msg}")

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)


def convert_inf_to_string(value):
    if value == float("inf"):
        return "inf"
    else:
        return value


def prepare_board_for_json(board):
    return [
        [[convert_inf_to_string(cell[0]), cell[1]] for cell in row] for row in board
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


def cmd_print(msg: str):
    socketio.emit("cmd_log", {"msg": msg})


def parse_options(tokens):
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


def sendDataToBackend(game: ChessGame):
    try:
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


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
    row = data["row"]
    col = data["col"]
    color = data["color"]  # 前端发送了颜色信息

    if not game.is_move_valid(row, col):
        cmd_print(f"非法落子位置: ({row}, {col})")
        return

    game.update_chessboard(row, col, color)
    game.update_history(row, col)
    sendDataToBackend(game)

    cmd_print(f"(Move = ({row}, {col}), Score = {game.get_score()})")

    if game.is_game_over():
        set_auto_mode()
        minimax_ai.end_model()

        cmd_print(f"Game Over. Winner = {game.who_is_winner()}")
        return

    if minimax_auto:
        executor.submit(handle_minimax_move)
    elif mcts_auto:
        executor.submit(handle_mcts_move)
    elif monky_auto:
        executor.submit(handle_monky_move)


@socketio.on("undo_move")
def handle_undo_move():
    game.undo()
    sendDataToBackend(game)
    cmd_print("已悔棋。")


@socketio.on("redo_move")
def handle_redo_move():
    game.redo()
    sendDataToBackend(game)
    cmd_print("已重悔。")


@socketio.on("restart_game")
def handle_restart_game():
    game.restart()
    sendDataToBackend(game)
    set_auto_mode()
    cmd_print("已重开棋局。")


@socketio.on("minimax_move")
def handle_minimax_move():
    # MinimaxAI执棋
    handle_ai_move(minimax_ai, game)


@socketio.on("mcts_move")
def handle_mcts_move():
    # MCTSAI执棋
    handle_ai_move(mcts_ai, game)


@socketio.on("monky_move")
def handle_monky_move():
    # MonkyAI执棋
    handle_ai_move(monky_ai, game)


# @socketio.on("dl_move")
# def handle_dl_move():
#     # DeepLearningAI执棋
#     handle_ai_move(dl_ai, game)


@socketio.on("minimax_auto")
def handle_minimax_auto():
    # 与Minimax对弈
    cmd_print("已开启与 Minimax 对弈模式。")
    set_auto_mode("minimax")
    handle_minimax_move()


@socketio.on("mcts_auto")
def handle_mcts_auto():
    # 与MCTS对弈
    cmd_print("已开启与 MCTS 对弈模式。")
    set_auto_mode("mcts")
    handle_mcts_move()


@socketio.on("monky_auto")
def handle_monky_auto():
    # 与Monky对弈
    cmd_print("已开启与 Monky 对弈模式。")
    set_auto_mode("monky")
    handle_monky_move()


# @socketio.on("al_auto")
# def handle_al_auto():
#     # 与DeepLearningAI对弈
#     set_auto_mode("dl")
#     handle_dl_move()


@socketio.on("cmd_input")
def handle_cmd_input(data):
    """
    前端发来：socket.emit('cmd_input', { text: 'celestial play --row 3 --col 4 --color 1' })
    在这里解析并执行相应操作。
    """
    text = data.get("text", "").strip()

    if not text:
        cmd_print("空命令。")
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
        cmd_print("未指定子命令。")
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
            cmd_print(f"play 参数错误: {e}")
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
            cmd_print("ai 后需要指定类型: minimax / mcts / monky")
            return

        ai_type = rest[0]

        if game.is_game_over():
            cmd_print("游戏已结束，无法继续 AI 执棋。")
            return

        if ai_type == "minimax":
            executor.submit(handle_minimax_move)
        elif ai_type == "mcts":
            executor.submit(handle_mcts_move)
        elif ai_type == "monky":
            executor.submit(handle_monky_move)
        else:
            cmd_print(f"未知 AI 类型: {ai_type}")
            return

        return

    # --------- 自动对弈：auto minimax/mcts/monky ----------
    elif cmd == "auto":
        if not rest:
            cmd_print("auto 后需要指定类型: minimax / mcts / monky")
            return

        ai_type = rest[0]

        if ai_type == "minimax":
            handle_minimax_auto()
        elif ai_type == "mcts":
            set_auto_mode("mcts")
            handle_mcts_auto()
        elif ai_type == "monky":
            handle_monky_auto()
        else:
            cmd_print(f"未知 auto 类型: {ai_type}")

        return

    # --------- 未知命令 ----------
    else:
        cmd_print(f"未知命令: {cmd}")
        return

    # 你想要的逻辑都可以写在这里
    # 比如调用 AI、查看棋盘、运行 debug 命令、执行脚本等

    # 也可以回消息
    # cmd_print(f"Echo: {text}")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
