from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from game.chess_game import ChessGame
from ai import MinimaxAI, MCTSAI

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app)

size = 11  # 棋盘大小
weight, height = (size, size)
power = 3  # 棋子力量

board = [[[0, 0] for _ in range(height)] for _ in range(weight)]
game = ChessGame((weight, height), power)
game.init_cfunc()
game.init_history()

minimax_ai = MinimaxAI(5)
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

def get_update_board(game: ChessGame):
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    move = game.get_current_move()
    return {"board": prepared_board, "move": move, "score": score, "step": game.step}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/init_state", methods=["GET"])
def init_state():
    # 获取初始棋局状态
    prepared_board = prepare_board_for_json(game.chessboard)
    score = game.get_score()
    move = game.get_current_move()
    return jsonify({"power": power, "weight": weight, "height": height,  
                    "board": prepared_board, "score": score, "step": game.step})


@socketio.on("play_move")
def handle_play_move(data):
    row = data["row"]
    col = data["col"]
    color = data["color"]  # 前端发送了颜色信息

    try:
        game.update_chessboard(row, col, color)
        game.update_history(row, col)
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("undo_move")
def handle_undo_move():
    game.undo()
    
    try:
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("redo_move")
def handle_redo_move():
    game.redo()
    
    try:
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("restart_game")
def handle_restart_game():
    game.restart()
    
    try:
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


@socketio.on("minimax_move")
def handle_minimax_move():
    # MinimaxAI执棋
    if game.is_game_over():
        return
    color = game.get_color()
    move = minimax_ai.find_best_move(game)
    try:
        game.update_chessboard(*move, color)
        game.update_history(*move)
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400

@socketio.on("mcts_move")
def handle_mcts_move():
    # MCTSAI执棋
    if game.is_game_over():
        return
    color = game.get_color()
    move = mcts_ai.find_best_move(game)
    try:
        game.update_chessboard(*move, color)
        game.update_history(*move)
        response = get_update_board(game)
        socketio.emit("update_board", response)
    except AssertionError as e:
        socketio.emit("update_board", {"error": str(e)}), 400


if __name__ == "__main__":
    socketio.run(app, debug=True)
