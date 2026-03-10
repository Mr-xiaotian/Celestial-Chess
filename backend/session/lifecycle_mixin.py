from typing import Dict, Optional

from celestialchess import ChessGame, MCTSAI

from ..session_utils import prepare_board_for_json


class SessionLifecycleMixin:
    def normalize_role_side_config(self, config: Optional[Dict]):
        """
        归一化单侧角色配置。

        参数:
            config: 角色配置，可为 human 或具体 AI 类型配置。

        返回:
            dict: 标准结构 {"role": str, "ai": dict}。
        """
        role = "human"
        ai_config = self.normalize_ai_config(None)
        ai_types = {"minimax", "mcts", "monky"}
        if isinstance(config, dict):
            raw_role = str(config.get("role") or "human").lower()
            if raw_role in ai_types:
                role = raw_role
            elif raw_role == "ai":
                role = "mcts"
            elif raw_role == "human":
                role = "human"
            raw_ai_config = config.get("ai") if isinstance(config.get("ai"), dict) else config
            ai_config = self.normalize_ai_config(raw_ai_config)
            if role != "human":
                ai_config["type"] = role
        return {"role": role, "ai": ai_config}

    def detect_mode(self):
        """
        根据红蓝两侧角色推断当前对局模式。

        返回:
            str: pvp / pve / spectator。
        """
        blue_ai = self.side_role_config["Blue"]["role"] != "human"
        red_ai = self.side_role_config["Red"]["role"] != "human"
        if blue_ai and red_ai:
            return "spectator"
        if blue_ai or red_ai:
            return "pve"
        return "pvp"

    def get_role_status_payload(self):
        """
        构造角色状态广播载荷。

        返回:
            dict: 包含 mode、blue、red、sleep。
        """
        return {
            "mode": self.detect_mode(),
            "blue": self.side_role_config["Blue"],
            "red": self.side_role_config["Red"],
            "sleep": self.spectator_sleep,
        }

    def get_analysis_config_payload(self):
        """
        构造分析配置广播载荷。

        返回:
            dict: 包含 iter。
        """
        return {"iter": self.analysis_iter}

    def emit_analysis_config(self):
        """向前端广播分析配置。"""
        self.socketio.emit("analysis_config", self.get_analysis_config_payload())

    def set_analysis_iter(self, iter_count):
        """
        设置分析迭代次数。

        参数:
            iter_count: 目标迭代次数，需为非负整数。
        """
        try:
            normalized_iter = int(iter_count)
        except (TypeError, ValueError):
            normalized_iter = 800
        self.analysis_iter = max(normalized_iter, 10)
        self.emit_analysis_config()
        self.schedule_analysis_if_needed()

    def emit_role_status(self):
        """向前端广播角色状态。"""
        self.socketio.emit("role_status", self.get_role_status_payload())

    def refresh_side_ai_map(self):
        """根据角色配置重建红蓝方 AI 实例缓存。"""
        self.side_ai_map = {"Blue": None, "Red": None}
        for side in ("Blue", "Red"):
            side_config = self.side_role_config[side]
            if side_config["role"] != "human":
                self.side_ai_map[side] = self.create_ai_from_config(side_config["ai"])

    def schedule_role_ai_turn_if_needed(self):
        """
        在当前回合调度“AI 执棋或人类分析”。

        规则:
            - AI 回合：提交 AI 执棋任务；
            - 所有回合：提交局面分析任务。
        """
        if self.ai_thinking or self.game.is_game_over():
            return
        self.schedule_analysis_if_needed()
        side = "Blue" if self.game.get_color() == 1 else "Red"
        ai = self.side_ai_map.get(side)
        if self.side_role_config[side]["role"] == "human":
            return
        if ai is None:
            ai = self.create_ai_from_config(self.side_role_config[side]["ai"])
            self.side_ai_map[side] = ai
        self.executor.submit(self.handle_ai_move, ai)

    def schedule_analysis_if_needed(self):
        """
        在当前回合异步触发 MCTS 分析。

        会先发送 pending 状态，再在后台线程完成分析并推送结果。
        """
        if self.ai_thinking or self.game.is_game_over():
            self.socketio.emit("analysis_update", {"status": "idle", "step": self.game.step})
            return
        side = "Blue" if self.game.get_color() == 1 else "Red"
        with self.lock:
            game_snapshot = self.game.copy()
            step = self.game.step
        self.analysis_seq += 1
        seq = self.analysis_seq
        self.socketio.emit("analysis_update", {"status": "pending", "turn": side, "step": step})
        self.executor.submit(self.run_human_analysis, seq, game_snapshot, side, step)

    def run_human_analysis(self, seq: int, game_snapshot: ChessGame, side: str, step: int):
        """
        执行单次人类回合分析任务并推送 analysis_update。

        参数:
            seq: 分析序号，用于丢弃过期结果。
            game_snapshot: 分析用棋局快照。
            side: 当前轮到的执棋方（Blue/Red）。
            step: 触发分析时的局面步数。
        """
        payload = {"status": "idle", "step": step}
        try:
            if game_snapshot.is_game_over():
                payload = {"status": "idle", "step": step}
            else:
                analyzer = MCTSAI(self.analysis_iter, complate_mode=False)
                analysis = analyzer.analyze_position(game_snapshot)
                payload = {
                    "status": "ready",
                    "turn": side,
                    "step": step,
                    "current_win_rate": analysis.get("current_win_rate"),
                    "moves": analysis.get("moves", []),
                }
        except Exception as e:
            payload = {"status": "error", "message": str(e), "step": step}
        if seq != self.analysis_seq:
            return
        self.socketio.emit("analysis_update", payload)

    def apply_role_config(self, blue_config: Optional[Dict], red_config: Optional[Dict], sleep: Optional[float], source: str):
        """
        应用红蓝双方角色配置并刷新模式状态。

        参数:
            blue_config: 蓝方角色配置。
            red_config: 红方角色配置。
            sleep: AI 对战回合间隔秒数。
            source: 触发来源标识。
        """
        normalized_blue = self.normalize_role_side_config(blue_config)
        normalized_red = self.normalize_role_side_config(red_config)
        try:
            normalized_sleep = float(sleep) if sleep is not None else 0.0
        except (TypeError, ValueError):
            normalized_sleep = 0.0
        self.side_role_config["Blue"] = normalized_blue
        self.side_role_config["Red"] = normalized_red
        self.spectator_sleep = max(normalized_sleep, 0.0)
        self.spectator_mode = self.detect_mode() == "spectator"
        self.refresh_side_ai_map()
        if self.spectator_mode:
            with self.lock:
                self.game.restart()
            self.sendDataToBackend()
        self.emit_role_status()
        blue_summary = "human" if normalized_blue["role"] == "human" else self.format_ai_config_summary(normalized_blue["ai"])
        red_summary = "human" if normalized_red["role"] == "human" else self.format_ai_config_summary(normalized_red["ai"])
        self.cmd_print(f"角色已更新: 蓝方={blue_summary}, 红方={red_summary}, 模式={self.detect_mode()}, 等待={self.spectator_sleep}s")
        self.schedule_role_ai_turn_if_needed()

    def get_init_state(self):
        """
        获取前端初始化所需的完整状态快照。

        返回:
            dict: 包含棋盘尺寸、棋力、棋盘内容、最近一步、分数、步数。
        """
        row_len, col_len = self.game.board_range
        power = self.game.power
        prepared_board = prepare_board_for_json(self.game.chessboard)
        move = self.game.get_current_move()
        score = self.game.get_score()
        step = self.game.step
        return {
            "power": power,
            "row_len": row_len,
            "col_len": col_len,
            "board": prepared_board,
            "move": move,
            "score": score,
            "step": step,
            "mode": self.detect_mode(),
            "blue": self.side_role_config["Blue"],
            "red": self.side_role_config["Red"],
            "sleep": self.spectator_sleep,
            "analysis_iter": self.analysis_iter,
        }

    def get_update_board(self):
        """
        获取棋盘增量广播数据。

        返回:
            dict: 包含棋盘、分数、当前步、游戏结束与胜者信息。
        """
        prepared_board = prepare_board_for_json(self.game.chessboard)
        score = self.game.get_score()
        move = self.game.get_current_move()

        game_over = self.game.is_game_over()
        winner = self.game.who_is_winner() if game_over else None

        return {
            "board": prepared_board,
            "move": move,
            "score": score,
            "step": self.game.step,
            "game_over": game_over,
            "winner": winner,
        }
