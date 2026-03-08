from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor

from celestialchess import ChessGame, MinimaxAI, MCTSAI, MonkyAI #, DeepLearningAI
import utils_backend


def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading")
    executor = ThreadPoolExecutor(max_workers=3)

    chess_state = ((11, 11), 3)

    game = ChessGame(*chess_state)
    game.init_cfunc()
    game.init_history()

    minimax_ai = MinimaxAI(2, False)
    minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
    mcts_ai = MCTSAI(1000, complate_mode=True)
    monky_ai = MonkyAI()

    ai_map = {
        "minimax": minimax_ai,
        "mcts": mcts_ai,
        "monky": monky_ai
    }

    session = utils_backend.GameSession(socketio, game, ai_map, executor)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/init_state", methods=["GET"])
    def init_state():
        return jsonify(session.get_init_state())

    @app.route("/configure", methods=["POST"])
    def configure():
        data = request.get_json(silent=True) or {}
        try:
            row_len = int(data.get("row_len"))
            col_len = int(data.get("col_len"))
            power = int(data.get("power"))
            if row_len <= 0 or col_len <= 0 or power <= 1:
                raise ValueError("invalid config")
        except Exception:
            return jsonify({"error": "invalid config"}), 400

        return jsonify(session.configure_game(row_len, col_len, power, "panel"))

    @socketio.on("play_move")
    def handle_play_move(data):
        session.handle_play_move(data)

    @socketio.on("undo_move")
    def handle_undo_move():
        session.handle_undo_move()

    @socketio.on("redo_move")
    def handle_redo_move():
        session.handle_redo_move()

    @socketio.on("restart_game")
    def handle_restart_game():
        session.handle_restart_game()

    def make_move_handler(ai_name):
        def handler():
            if not session.ensure_not_spectator():
                return
            session.handle_ai_move(session.ai_map[ai_name])
        return handler

    def make_auto_handler(ai_name):
        def handler():
            if not session.ensure_not_spectator():
                return
            session.set_auto_mode(ai_name)
            session.handle_ai_auto(session.ai_map[ai_name])
        return handler

    for name in ai_map:
        socketio.on(f"{name}_move")(make_move_handler(name))
        socketio.on(f"{name}_auto")(make_auto_handler(name))

    @socketio.on("cmd_input")
    def handle_cmd_input(data):
        session.handle_cmd_input(data)

    @socketio.on("start_spectator")
    def handle_start_spectator(data):
        data = data or {}
        session.start_spectator(data.get("blue"), data.get("red"), data.get("sleep"))

    @socketio.on("stop_spectator")
    def handle_stop_spectator():
        session.stop_spectator()

    return app, socketio


app, socketio = create_app()


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
