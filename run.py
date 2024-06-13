from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO
from game.chess_game import ChessGame
from ai import MinimaxAI, MCTSAI

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app)

size = 11  # 棋盘大小
weight, height = (size, size)
power = 3  # 棋子力量

# 假设棋盘初始状态
board = [[[0, 0] for _ in range(height)] for _ in range(weight)]
game = ChessGame((weight, height), power)
minimax_ai = MinimaxAI(3)
mcts_ai = MCTSAI(1000)

def convert_inf_to_string(value):
    if value == float("inf"):
        return "inf"
    elif value == float("-inf"):
        return "-inf"
    else:
        return value

def prepare_board_for_json(board):
    return [[[convert_inf_to_string(cell[0]),cell[1]] for cell in row] for row in board]

def update_board(row, col, color):
    game.update_chessboard(row, col, color)
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    return {"board": prepared_board, "score": score, "step": game.step}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/init_state", methods=["GET"])
def init_state():
    # 获取初始棋局状态
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    return jsonify({"power": power, "weight": weight, "height": height,  
                    "board": prepared_board, "score": score, "step": game.step})


@socketio.on("play_move")
def handle_play_move(data):
    row = data["row"]
    col = data["col"]
    color = data["color"]  # 前端发送了颜色信息

    try:
        response = update_board(row, col, color)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("undo_move")
def handle_undo_move():
    game.undo()
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    try:
        socketio.emit("update_board", {"board": prepared_board, "score": score, "step": game.step})
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("redo_move")
def handle_redo_move():
    game.redo()
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    try:
        socketio.emit("update_board", {"board": prepared_board, "score": score, "step": game.step})
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("restart_game")
def handle_restart_game():
    game.restart()
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    try:
        socketio.emit("update_board", {"board": prepared_board, "score": score, "step": game.step})
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("minimax_move")
def handle_minimax_move():
    # AI执棋逻辑
    if game.is_game_over():
        return
    color = game.get_color()
    move = minimax_ai.find_best_move(game)
    try:
        response = update_board(*move, color)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400

@socketio.on("mcts_move")
def handle_mcts_move():
    # AI执棋逻辑
    if game.is_game_over():
        return
    color = game.get_color()
    move = mcts_ai.find_best_move(game)
    try:
        response = update_board(*move, color)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, debug=True)
