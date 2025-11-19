from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor

from celestialchess import ChessGame, MinimaxAI, MCTSAI, MonkyAI

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# 创建线程池
executor = ThreadPoolExecutor(max_workers=3)

chess_state = ((11, 11), 3)
(row_len, col_len), power = chess_state # 棋盘大小，power

game = ChessGame(*chess_state)
game.init_cfunc()
game.init_history()

minimax_ai = MinimaxAI(5, chess_state)
mcts_ai = MCTSAI(10000)
monky_ai = MonkyAI()

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

def convert_inf_to_string(value):
    if value == float("inf"):
        return "inf"
    elif value == float("-inf"):
        return "-inf"
    else:
        return value

def prepare_board_for_json(board):
    return [[[convert_inf_to_string(cell[0]),cell[1]] for cell in row] for row in board]

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
        "winner": winner
    }

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
    score = game.get_score()
    # move = game.get_current_move()
    return jsonify({"power": power, "row_len": row_len, "col_len": col_len,  
                    "board": prepared_board, "score": score, "step": game.step})


@socketio.on("play_move")
def handle_play_move(data):
    row = data["row"]
    col = data["col"]
    color = data["color"]  # 前端发送了颜色信息

    if not game.is_move_valid(row, col):
        return
    
    game.update_chessboard(row, col, color)
    game.update_history(row, col)
    sendDataToBackend(game)

    if game.is_game_over():
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

@socketio.on("redo_move")
def handle_redo_move():
    game.redo()
    sendDataToBackend(game)

@socketio.on("restart_game")
def handle_restart_game():
    game.restart()
    sendDataToBackend(game)
    set_auto_mode()


@socketio.on("minimax_move")
def handle_minimax_move():
    # MinimaxAI执棋
    color = game.get_color()
    move = minimax_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)

@socketio.on("mcts_move")
def handle_mcts_move():
    # MCTSAI执棋
    color = game.get_color()
    move = mcts_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)

@socketio.on("monky_move")
def handle_monky_move():
    # MonkyAI执棋
    color = game.get_color()
    move = monky_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)


@socketio.on("minimax_auto")
def handle_minimax_auto():
    # 与Minimax对弈
    set_auto_mode("minimax")
    handle_minimax_move()

@socketio.on("mcts_auto")
def handle_mcts_auto():
    # 与MCTS对弈
    set_auto_mode("mcts")
    handle_mcts_move()

@socketio.on("monky_auto")
def handle_monky_auto():
    # 与Monky对弈
    set_auto_mode("monky")
    handle_monky_move()

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
