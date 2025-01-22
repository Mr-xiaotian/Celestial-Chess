from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from game.chess_game import ChessGame
from ai import MinimaxAI, MCTSAI, MonkyAI

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
socketio = SocketIO(app)

chess_state = ((8, 10), 3)
(row_len, col_len), power = chess_state # 棋盘大小，power

game = ChessGame(*chess_state)
game.init_cfunc()
game.init_history()

minimax_ai = MinimaxAI(5, *chess_state)
mcts_ai = MCTSAI(10000)
monky_ai = MonkyAI()

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

    if game.is_move_valid(row, col):
        game.update_chessboard(row, col, color)
        game.update_history(row, col)
        sendDataToBackend(game)


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


@socketio.on("minimax_move")
def handle_minimax_move():
    # MinimaxAI执棋
    if game.is_game_over():
        return
    color = game.get_color()
    move = minimax_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)

@socketio.on("mcts_move")
def handle_mcts_move():
    # MCTSAI执棋
    if game.is_game_over():
        return
    color = game.get_color()
    move = mcts_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)

@socketio.on("monky_move")
def handle_monky_move():
    # MCTSAI执棋
    if game.is_game_over():
        return
    color = game.get_color()
    move = monky_ai.find_best_move(game)

    game.update_chessboard(*move, color)
    game.update_history(*move)
    sendDataToBackend(game)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5005)
