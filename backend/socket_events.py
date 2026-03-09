def register_socket_events(socketio, session):
    @socketio.on("play_move")
    def handle_play_move(data):
        """处理前端落子事件。"""
        session.handle_play_move(data)

    @socketio.on("undo_move")
    def handle_undo_move():
        """处理悔棋事件。"""
        session.handle_undo_move()

    @socketio.on("redo_move")
    def handle_redo_move():
        """处理重做事件。"""
        session.handle_redo_move()

    @socketio.on("restart_game")
    def handle_restart_game():
        """处理重开棋局事件。"""
        session.handle_restart_game()

    def make_move_handler(ai_name):
        """
        生成指定 AI 的单步执棋事件处理函数。

        参数:
            ai_name (str): AI 名称键（如 minimax / mcts / monky）。
        """

        def handler():
            """执行单步 AI 执棋，观战模式下拒绝执行。"""
            if not session.ensure_not_spectator():
                return
            session.handle_ai_move(session.ai_map[ai_name])

        return handler

    def make_auto_handler(ai_name):
        """
        生成指定 AI 的自动对弈事件处理函数。

        参数:
            ai_name (str): AI 名称键（如 minimax / mcts / monky）。
        """

        def handler():
            """切换自动模式并立即触发 AI 回合。"""
            if not session.ensure_not_spectator():
                return
            session.set_auto_mode(ai_name)
            session.handle_ai_auto(session.ai_map[ai_name])

        return handler

    for name in session.ai_map:
        socketio.on(f"{name}_move")(make_move_handler(name))
        socketio.on(f"{name}_auto")(make_auto_handler(name))

    @socketio.on("cmd_input")
    def handle_cmd_input(data):
        """处理 CMD 输入事件。"""
        session.handle_cmd_input(data)

    @socketio.on("start_spectator")
    def handle_start_spectator(data):
        """
        处理启动观战事件。

        参数:
            data (dict): 包含 blue/red AI 配置及 sleep 间隔。
        """
        data = data or {}
        session.start_spectator(data.get("blue"), data.get("red"), data.get("sleep"))

    @socketio.on("stop_spectator")
    def handle_stop_spectator():
        """处理停止观战事件。"""
        session.stop_spectator()

    @socketio.on("apply_roles")
    def handle_apply_roles(data):
        """
        应用红蓝方角色配置（PVP / PVE / AI 对战）。

        参数:
            data (dict): 包含 blue/red 角色配置与 sleep。
        """
        data = data or {}
        session.apply_role_config(data.get("blue"), data.get("red"), data.get("sleep"), "panel")
