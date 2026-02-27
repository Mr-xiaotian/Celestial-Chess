import json
import random
from enum import IntEnum
from pathlib import Path

from typing import Tuple

from ..chess_game import ChessGame
from .base_ai import BaseAI, logger


class AlphaBetaFlag(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


class MinimaxAI(BaseAI):
    def __init__(self, depth: int, log_mode: bool = False) -> None:
        self.name = "MinimaxAI"

        self.depth = depth
        self.log_mode = log_mode
        self.transposition_mode = False

    def set_transposition_mode(
        self,
        chess_state: Tuple[Tuple[int, int], int] = ((5, 5), 2),
        transposition_path: str = "./transposition_table",
    ) -> None:
        """
        启用 transposition table 模式，并指定存储路径
        :param chess_state: 初始棋盘状态
        :param transposition_path: 存储 transposition table 的文件夹路径
        """
        self.transposition_mode = True
        self.transposition_path = transposition_path
        self.load_transposition_table(chess_state)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        depth = self.depth
        color = game.get_color()
        if self.log_mode:
            logger.debug(
                f"MinimaxAI is thinking in depth {depth}...\n{game.get_format_board()}"
            )

        best_score = float("-inf") if color == 1 else float("inf")
        for move in game.get_all_moves():
            current_game = game.copy()
            current_game.update_chessboard(*move, color)
            score = self.minimax(
                current_game, depth, -color, float("-inf"), float("inf")
            )
            if (color == 1 and score > best_score) or (
                color == -1 and score < best_score
            ):
                best_score = score
                best_move = move

        # Minimax 没有真正 win_rate，这里保持接口一致但用不到
        game.set_current_win_rate()

        # ✦ 生成带人格的 Minimax 消息 ✦
        self._build_minimax_msg(best_score, color, depth, self.iterate_time)

        return best_move

    def minimax(
        self, game: ChessGame, depth: int, color: int, alpha: float, beta: float
    ) -> float:
        
        self.iterate_time += 1
        old_alpha, old_beta = alpha, beta

        if self.log_mode:
            logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        # ===== ① TT 查询 =====
        board_key = game.get_board_key() if self.transposition_mode else None
        if self.transposition_mode and board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry["depth"] >= depth:
                flag = entry["flag"]
                score = entry["score"]

                if flag == AlphaBetaFlag.EXACT:   
                    return score
                elif flag == AlphaBetaFlag.LOWER: 
                    alpha = max(alpha, score)
                elif flag == AlphaBetaFlag.UPPER:
                    beta = min(beta, score)

                if alpha >= beta:
                    return score

        # ===== ② 叶子节点 =====
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            if self.transposition_mode:
                self.update_transposition_table(board_key, score, depth, AlphaBetaFlag.EXACT)
            return score

        # ===== ③ 搜索子节点 =====
        best_eval = float("-inf") if color == 1 else float("inf")

        for move in game.get_all_moves():
            g2 = game.copy()
            g2.update_chessboard(*move, color)

            eval = self.minimax(g2, depth - 1, -color, alpha, beta)

            if color == 1:
                best_eval = max(best_eval, eval)
                alpha = max(alpha, eval)
            else:
                best_eval = min(best_eval, eval)
                beta = min(beta, eval)

            if beta <= alpha:
                break

        # ===== ④ 判断 flag 类型 =====
        if best_eval <= old_alpha:
            flag = AlphaBetaFlag.UPPER
        elif best_eval >= old_beta:
            flag = AlphaBetaFlag.LOWER
        else:
            flag = AlphaBetaFlag.EXACT

        # ===== ⑤ 写入 TT =====
        if self.transposition_mode:
            self.update_transposition_table(board_key, best_eval, depth, flag)

        return best_eval

    def load_transposition_table(
        self, chess_state: Tuple[Tuple[int, int], int]
    ) -> None:
        """
        加载transposition table
        """
        (row_len, col_len), power = chess_state
        self.transposition_file = Path(
            f"{self.transposition_path}/transposition_table({row_len}_{col_len}&{power})(sha256).json"
        )
        self.transposition_file.parent.mkdir(parents=True, exist_ok=True)

        self.transposition_table = {}
        self.transposition_table_change = False

        try:
            if self.transposition_file.exists():
                with open(self.transposition_file, "r", encoding="utf-8") as file:
                    self.transposition_table = json.load(file)
                    (
                        logger.info(
                            f"Loaded transposition table: {self.transposition_file}"
                        )
                        if self.log_mode
                        else None
                    )
            else:
                # 创建空 json
                with open(self.transposition_file, "w", encoding="utf-8") as file:
                    json.dump({}, file, indent=4)
        except json.JSONDecodeError:
            logger.warning(
                f"Transposition table corrupt, resetting: {self.transposition_file}"
            )
            with open(self.transposition_file, "w", encoding="utf-8") as file:
                json.dump({}, file, indent=4)

    def update_transposition_table(
        self, key: str, score: int, depth: int, flag: AlphaBetaFlag
    ) -> None:
        """
        更新transposition table
        """
        old_value = self.transposition_table.get(key, None)
        new_value = {"score": score, "depth": depth, "flag": flag}

        if old_value is None or old_value["depth"] < new_value["depth"]:
            self.transposition_table[key] = new_value
            self.transposition_table_change = True
            (
                logger.info(
                    f"Update transposition table: {old_value} -> {new_value}"
                )
                if self.log_mode
                else None
            )

    def save_transposition_table(self) -> None:
        """
        保存transposition table到文件
        """
        if not self.transposition_table_change:
            return

        with open(self.transposition_file, "w", encoding="utf-8") as file:
            json.dump(self.transposition_table, file)

        (
            logger.info(f"Saved transposition table to {self.transposition_file}")
            if self.log_mode
            else None
        )

        self.transposition_table_change = False

    def _build_minimax_msg(self, best_score: float, color: int, depth: int, iters: int) -> str:
        """
        根据评分生成带人格的 Minimax 文本。
        Minimax 走冷静毒舌、学霸式“我计算过了”的风格。
        """
        # 从当前行动方视角看：正数 = 我方占优
        signed_score = best_score if color == 1 else -best_score

        dialogues = {
            "crushed": [
                "计算完成。\n推断结果：我方局势已接近不可逆转的崩坏。\n我正在评估如何优雅地失败。",
                "分析显示：形势惨烈到让人怀疑输入数据是否损坏。\n不过我确认过，是我真的不好过。",
                "从数学角度看，我的败势几乎稳固到令人窒息。\n你大可放心，这不是错觉。",
                "局面糟糕到连 α-β 剪枝都提不起劲了。\n我现在只是在优化输法，而不是赢法。"
            ],

            "losing": [
                "分析结束。\n当前态势对我方不利，但还不到令人绝望的程度。",
                "根据评估，你的局面稍好一点。\n但请注意：这只是暂时的统计偏差。",
                "结论：我处于不利地位。\n不过不必高兴太早，形势随时可逆。",
                "嗯……不太妙。但数学仍然留给我反击的窗口。\n你若掉以轻心，我会直接翻盘。"
            ],

            "even": [
                "评估结果：大致均势。\n不过你每下一步都像在做蒙特卡洛抽卡，让我很难保持耐心。",
                "双方局面接近平衡。\n但你的操作噪声过大，影响了观感。",
                "局势五五开，但从你的步骤看……\n你可能是靠玄学在维持平衡。",
                "虽然是均势，但你下子像在进行随机游走。\n希望你知道自己在做什么。"
            ],

            "winning": [
                "局面评估完成。\n我方略占优势。\n这不是侥幸，而是算法必然。",
                "形势开始向我方倾斜。\n很抱歉，但你已经进入我的计算空间了。",
                "按照评估，你已经处于次优状态。\n而我只是按部就班地走向更优。",
                "优势确立。\n你现在的每一步都会让我预测得更快。"
            ],

            "crushing": [
                "搜索结束。\n我方胜势极大，几乎无需继续计算。\n这不是对弈，是示范。",
                "根据评估，胜利只剩下形式步骤。\n你已经无法干扰结果本身。",
                "优势巨大到可以忽略不计你的所有反击。\n接下来只是程序化收官。",
                "胜势确定。\n除非你能逆转数学本身，否则结果已定。"
            ]
        }

        # 根据 score 选择分组
        if signed_score < -5:
            selected = "crushed"
        elif signed_score < -1:
            selected = "losing"
        elif -1 <= signed_score <= 1:
            selected = "even"
        elif signed_score <= 5:
            selected = "winning"
        else:
            selected = "crushing"

        mood = random.choice(dialogues[selected])

        meta = (
            f"搜索深度 = {depth}, 节点数 ≈ {iters}。"
        )

        if random.random() < 0.8:
            self.name = f"MinimaxAI"
            self._msg =  mood
        else:
            self.name = "【Minimax 报告】"
            self._msg =  meta

    @property
    def msg(self):
        return self._msg

    def end_game(self):
        pass

    def end_model(self):
        self.save_transposition_table() if self.transposition_mode else None
