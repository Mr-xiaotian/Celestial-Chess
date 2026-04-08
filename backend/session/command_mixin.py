import shlex

from ..session_utils import parse_options


class SessionCommandMixin:
    def handle_cmd_input(self, data):
        """
        处理 CMD 文本输入并分派子命令。

        支持命令:
            play / undo / redo / restart / ai / auto / config / help
        """
        text = data.get("text", "").strip()

        if not text:
            self.cmd_print("空命令。")
            return

        try:
            args = shlex.split(text)
        except ValueError as e:
            self.socketio.emit("cmd_output", {
                "success": False,
                "message": f"命令解析错误: {e}"
            })
            return

        if args and args[0] == "celestial":
            args = args[1:]

        if not args:
            self.cmd_print("未指定子命令。")
            return

        cmd = args[0]
        rest = args[1:]

        if cmd == "play":
            opts = parse_options(rest)

            try:
                if "row" not in opts or "col" not in opts:
                    raise ValueError("缺少 --row 或 --col")

                row = int(opts["row"])
                col = int(opts["col"])
                color = int(opts.get("color", self.game.get_color()))
            except Exception as e:
                self.cmd_print(f"play 参数错误: {e}")
                return

            self.handle_play_move({"row": row, "col": col, "color": color})
            return

        elif cmd == "undo":
            self.handle_undo_move()
            return

        elif cmd == "redo":
            self.handle_redo_move()
            return

        elif cmd == "restart":
            self.handle_restart_game()
            return

        elif cmd == "ai":
            if not rest:
                self.cmd_print("ai 后需要指定类型: minimax / mcts / monky")
                return

            ai_type = rest[0]

            if self.game.is_game_over():
                self.cmd_print("游戏已结束，无法继续 AI 执棋。")
                return

            if ai_type == "minimax":
                self.executor.submit(self.handle_ai_move, self.ai_map["minimax"])
            elif ai_type == "mcts":
                self.executor.submit(self.handle_ai_move, self.ai_map["mcts"])
            elif ai_type == "monky":
                self.executor.submit(self.handle_ai_move, self.ai_map["monky"])
            else:
                self.cmd_print(f"未知 AI 类型: {ai_type}")
                return

            return

        elif cmd == "auto":
            if not rest:
                self.cmd_print("auto 后需要指定类型: minimax / mcts / monky")
                return

            ai_type = rest[0]

            if ai_type == "minimax":
                self.set_auto_mode("minimax")
                self.handle_ai_auto(self.ai_map["minimax"])
            elif ai_type == "mcts":
                self.set_auto_mode("mcts")
                self.handle_ai_auto(self.ai_map["mcts"])
            elif ai_type == "monky":
                self.set_auto_mode("monky")
                self.handle_ai_auto(self.ai_map["monky"])
            else:
                self.cmd_print(f"未知 auto 类型: {ai_type}")

            return

        elif cmd == "config":
            if not rest:
                self.cmd_print("config 用法: config board --row N --col N --power N | config spectator --start/--stop")
                return

            mode = rest[0] if rest[0] and not rest[0].startswith("--") else "board"
            opts = parse_options(rest[1:] if mode != "board" else rest)

            if mode == "board":
                try:
                    row = int(opts.get("row"))
                    col = int(opts.get("col"))
                    power = int(opts.get("power"))
                    if row <= 0 or col <= 0 or power <= 1:
                        raise ValueError("invalid config")
                except Exception:
                    self.cmd_print("配置参数错误: 需要 --row/--col/--power")
                    return

                self.configure_game(row, col, power, "cmd")
                return

            if mode == "spectator":
                if "stop" in opts or "stop" in rest:
                    self.stop_spectator()
                    return

                blue = {
                    "type": opts.get("blue") or opts.get("blue-type") or "mcts",
                    "minimax_depth": opts.get("blue-depth") or opts.get("blue_depth"),
                    "mcts_iter": opts.get("blue-iter") or opts.get("blue_iter"),
                }
                red = {
                    "type": opts.get("red") or opts.get("red-type") or "mcts",
                    "minimax_depth": opts.get("red-depth") or opts.get("red_depth"),
                    "mcts_iter": opts.get("red-iter") or opts.get("red_iter"),
                }
                sleep = opts.get("sleep") or opts.get("delay")
                self.start_spectator(blue, red, sleep)
                return

            self.cmd_print("config 用法: config board --row N --col N --power N | config spectator --start/--stop")
            return

        elif cmd == 'help':
            self.cmd_print(
"""可用命令:
play --row N --col N --color C  播放移动 (N: 行/列索引, C: 颜色 0/1)
undo  撤销上一步
redo  重做上一步
restart  重新开始游戏
ai minimax/mcts/monky  执行指定 AI 类型的移动
auto minimax/mcts/monky  自动执行指定 AI 类型的移动
config board --row N --col N --power N  配置游戏板 (N: 行/列数, P: 棋力)
config spectator --start/--stop  启动/停止观众模式
help  显示帮助信息"""
            )
            return

        else:
            self.cmd_print(f"未知命令: {cmd}, 请输入 'help' 查看可用命令")
            return
