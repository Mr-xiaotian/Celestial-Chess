import time
from typing import Dict, Optional

from celestialchess import ChessGame, BaseAI, MinimaxAI, MCTSAI, MonkyAI


class SessionGameplayMixin:
    def cmd_print(self, msg: str, msg_type: str = "system"):
        """
        向前端 CMD 面板发送日志。

        参数:
            msg: 文本内容。
            msg_type: 日志类型（用于前端样式分类）。
        """
        self.socketio.emit("cmd_log", {"msg": msg, "type": msg_type})

    def resolve_ai_msg_type(self, ai: BaseAI, color: int):
        """
        根据 AI 名称与执棋方生成前端日志类型。

        参数:
            ai: AI 实例。
            color: 执棋方（1=蓝方，-1=红方）。

        返回:
            str: 例如 ai-minimax-blue / ai-mcts-red。
        """
        side = "blue" if color == 1 else "red"
        ai_name = ai.name.lower()
        if "minimax" in ai_name:
            return f"ai-minimax-{side}"
        if "mcts" in ai_name:
            return f"ai-mcts-{side}"
        if "monky" in ai_name:
            return f"ai-monky-{side}"
        if "deep" in ai_name or "learning" in ai_name:
            return f"ai-deeplearning-{side}"
        return f"ai-generic-{side}"

    def emit_config_changed(self, source: str):
        """
        广播“配置已变化”事件。

        参数:
            source: 触发来源标识（如 panel / cmd）。
        """
        payload = self.get_init_state()
        payload["source"] = source
        self.socketio.emit("config_changed", payload)

    def emit_spectator_status(self, status: str, blue_config=None, red_config=None, sleep=None):
        """
        广播观战模式状态。

        参数:
            status: 状态字符串（start / stop）。
            blue_config: 蓝方 AI 配置（可选）。
            red_config: 红方 AI 配置（可选）。
            sleep: 观战回合间隔（可选）。
        """
        payload = {"status": status}
        if blue_config is not None:
            payload["blue"] = blue_config
        if red_config is not None:
            payload["red"] = red_config
        if sleep is not None:
            payload["sleep"] = sleep
        self.socketio.emit("spectator_status", payload)

    def sendDataToBackend(self):
        """
        向前端广播棋盘更新。

        如果构造更新数据时发生断言异常，会回传错误消息。
        """
        try:
            response = self.get_update_board()
            self.socketio.emit("update_board", response)
        except AssertionError as e:
            self.socketio.emit("update_board", {"error": str(e)})

    def ensure_not_spectator(self):
        """
        校验当前是否处于观战模式。

        返回:
            bool:
                - False: 观战中，不允许执行手动操作；
                - True: 非观战状态，可继续执行。
        """
        if self.spectator_mode:
            self.cmd_print("观战中，无法执行该操作。")
            return False
        return True

    def build_ai_map(self, chess_state, minimax_depth=2, mcts_iter=1000):
        """
        构建标准 AI 映射表。

        参数:
            chess_state: 棋局状态定义，供部分 AI 初始化使用。
            minimax_depth: Minimax 搜索深度。
            mcts_iter: MCTS 迭代次数。
        """
        minimax_ai = MinimaxAI(minimax_depth, False)
        minimax_ai.set_transposition_mode(chess_state, r"transposition_table/")
        mcts_ai = MCTSAI(mcts_iter)
        monky_ai = MonkyAI()
        return {
            "minimax": minimax_ai,
            "mcts": mcts_ai,
            "monky": monky_ai
        }

    def configure_game(self, row_len: int, col_len: int, power: int, source: str):
        """
        重新配置并重建棋局。

        行为:
            1. 停止观战；
            2. 新建游戏对象并初始化历史；
            3. 重建 AI 映射；
            4. 广播配置变更与棋盘状态。

        返回:
            dict: 新棋局初始化状态。
        """
        with self.lock:
            self.game = ChessGame((row_len, col_len), power)
            self.game.init_cfunc()
            self.game.init_history()

            chess_state = ((row_len, col_len), power)
            self.ai_map = self.build_ai_map(chess_state)
            self.auto_mode = None
            self.spectator_mode = self.detect_mode() == "spectator"
            self.refresh_side_ai_map()

        self.emit_config_changed(source)
        self.emit_role_status()
        self.sendDataToBackend()
        self.cmd_print(f"配置已更新: 棋盘={row_len}x{col_len}, power={power}, 来源={source}")
        self.schedule_role_ai_turn_if_needed()
        return self.get_init_state()

    def normalize_ai_config(self, config: Optional[Dict]):
        """
        归一化 AI 配置，补齐缺省值并转换类型。

        参数:
            config: 原始配置字典（可为空）。

        返回:
            dict: 标准化后的配置。
        """
        ai_type = "mcts"
        minimax_depth = 2
        mcts_iter = 1000
        if isinstance(config, dict):
            if config.get("type"):
                ai_type = str(config.get("type")).lower()
            if config.get("minimax_depth") is not None:
                minimax_depth = int(config.get("minimax_depth"))
            if config.get("mcts_iter") is not None:
                mcts_iter = int(config.get("mcts_iter"))
        return {
            "type": ai_type,
            "minimax_depth": minimax_depth,
            "mcts_iter": mcts_iter
        }

    def create_ai_from_config(self, config: Dict):
        """
        根据标准化配置创建 AI 实例。

        参数:
            config: AI 配置字典。

        返回:
            BaseAI: 对应的 AI 对象。
        """
        row_len, col_len = self.game.board_range
        chess_state = ((row_len, col_len), self.game.power)
        if config["type"] == "minimax":
            ai = MinimaxAI(config["minimax_depth"], False)
            ai.set_transposition_mode(chess_state, r"transposition_table/")
            return ai
        if config["type"] == "mcts":
            return MCTSAI(config["mcts_iter"])
        if config["type"] == "monky":
            return MonkyAI()
        return MCTSAI(config["mcts_iter"])

    def run_spectator_loop(self, blue_config: Dict, red_config: Dict):
        """
        观战循环主逻辑。

        根据当前执棋方轮流调用蓝方或红方 AI，直到游戏结束
        或观战模式被外部停止。
        """
        ai_blue = self.create_ai_from_config(blue_config)
        ai_red = self.create_ai_from_config(red_config)

        while self.spectator_mode and not self.game.is_game_over():
            ai = ai_blue if self.game.get_color() == 1 else ai_red
            self.handle_ai_move(ai)

        if self.spectator_mode:
            self.spectator_mode = False
            self.emit_spectator_status("stop")

    def start_spectator(self, blue_config: Optional[Dict], red_config: Optional[Dict], sleep: Optional[float] = None):
        """
        启动观战模式。

        参数:
            blue_config: 蓝方 AI 配置。
            red_config: 红方 AI 配置。
            sleep: 每步最小间隔秒数（可选）。
        """
        self.set_auto_mode()
        blue_ai = self.normalize_ai_config(blue_config)
        red_ai = self.normalize_ai_config(red_config)
        blue_role = {"role": blue_ai["type"], "ai": blue_ai}
        red_role = {"role": red_ai["type"], "ai": red_ai}
        self.apply_role_config(blue_role, red_role, sleep, "spectator")
        self.emit_spectator_status("start", blue_role["ai"], red_role["ai"], self.spectator_sleep)

    def stop_spectator(self):
        """停止观战模式并广播状态。"""
        if self.detect_mode() == "spectator":
            self.apply_role_config({"role": "human"}, {"role": "human"}, 0.0, "spectator")
            self.cmd_print("观战已停止")
            self.emit_spectator_status("stop")

    def format_ai_config_summary(self, config: Dict):
        """
        生成人类可读的 AI 配置摘要。

        参数:
            config: AI 配置字典。
        """
        if config["type"] == "minimax":
            return f"minimax(depth={config['minimax_depth']})"
        if config["type"] == "mcts":
            return f"mcts(iter={config['mcts_iter']})"
        if config["type"] == "monky":
            return "monky"
        return config["type"]

    def apply_move_and_update(self, row: int, col: int, color: int):
        """
        在游戏对象中应用落子并推送更新。

        参数:
            row: 落子行索引。
            col: 落子列索引。
            color: 执棋颜色（1/-1）。

        返回:
            bool: 是否在该步后进入终局。
        """
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
        """
        执行一次 AI 回合。

        该方法负责：
            - 打印 AI 思考日志；
            - 调用 AI 搜索落子；
            - 根据观战延时策略补齐等待；
            - 应用落子并处理终局收尾。
        """
        if self.game.is_game_over():
            return

        self.ai_thinking = True
        self.socketio.emit("ai_thinking", {"status": "start"})
        try:
            color = self.game.get_color()
            ai_msg_type = self.resolve_ai_msg_type(ai, color)
            self.cmd_print(f"({ai.name} thinking...)", ai_msg_type)
            start_time = time.perf_counter()
            move = ai.find_best_move(self.game)
            self.cmd_print(f"{ai.name}: {ai.deepseek_msg}", ai_msg_type)

            if self.spectator_mode and self.spectator_sleep > 0:
                elapsed = time.perf_counter() - start_time
                remaining = self.spectator_sleep - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                if not self.spectator_mode:
                    return

            with self.lock:
                finished = self.apply_move_and_update(move[0], move[1], color)

            if finished:
                ai.end_model()
        finally:
            self.ai_thinking = False
            self.socketio.emit("ai_thinking", {"status": "stop"})
            self.schedule_role_ai_turn_if_needed()

    def handle_ai_auto(self, ai: BaseAI):
        """
        进入与指定 AI 的自动对弈流程。

        参数:
            ai: 对手 AI 实例。
        """
        self.cmd_print(f"已开启与 {ai.name} 对弈模式。")
        self.handle_ai_move(ai)

    def handle_play_move(self, data):
        """
        处理前端手动落子请求。

        参数:
            data: 包含 row/col/color 的字典。
        """
        if not self.ensure_not_spectator():
            return
        if self.ai_thinking:
            self.cmd_print("AI 思考中，棋盘已锁定，不能下子。")
            return

        row = data["row"]
        col = data["col"]
        color = data["color"]
        side = "Blue" if color == 1 else "Red"
        if self.side_role_config[side]["role"] != "human":
            self.cmd_print(f"{side} 方由 AI 执棋，当前不允许手动落子。")
            return

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
            return
        self.schedule_role_ai_turn_if_needed()

    def handle_undo_move(self):
        """处理悔棋请求并广播更新。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.undo()
        self.sendDataToBackend()
        self.cmd_print("已悔棋。")
        self.schedule_role_ai_turn_if_needed()

    def handle_redo_move(self):
        """处理重做请求并广播更新。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.redo()
        self.sendDataToBackend()
        self.cmd_print("已重悔。")
        self.schedule_role_ai_turn_if_needed()

    def handle_restart_game(self):
        """处理重开棋局请求并清空自动模式。"""
        if not self.ensure_not_spectator():
            return
        with self.lock:
            self.game.restart()
        self.sendDataToBackend()
        self.set_auto_mode()
        self.cmd_print("已重开棋局。")
        self.schedule_role_ai_turn_if_needed()
