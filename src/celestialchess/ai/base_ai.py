import os
import json
import requests
from typing import Tuple
from loguru import logger
from time import strftime, localtime
from dotenv import load_dotenv

from ..chess_game import ChessGame
from .identity_prompts import get_identity_prompt


# Configure logging
logger.remove()  # remove the default handler
now_time = strftime("%Y-%m-%d", localtime())
os.makedirs("logs", exist_ok=True)
logger.add(
    f"logs/chess_log({now_time}).log",
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
    level="INFO",
)

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class BaseAI:
    _name = "BaseAI"
    _msg = ""
    _deepseek_api_key = None
    _deepseek_config_loaded = False

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def end_game(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def end_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @property
    def name(self):
        return self._name

    @property
    def msg(self):
        return self._msg
    
    def _get_position_context(self) -> str:
        pieces = [f"ai={self.name}", f"msg={self.msg}"]

        if hasattr(self, "best_score") and hasattr(self, "color"):
            signed_score = self.best_score if self.color == 1 else -self.best_score
            pieces.append(f"signed_score={signed_score:.4f}")
        if hasattr(self, "iterate_time"):
            pieces.append(f"iterate_time={self.iterate_time}")
        if hasattr(self, "depth"):
            pieces.append(f"depth={self.depth}")
        if hasattr(self, "best_win_rate"):
            pieces.append(f"best_win_rate={float(self.best_win_rate):.4f}")
        if hasattr(self, "itermax"):
            pieces.append(f"itermax={self.itermax}")
        if hasattr(self, "masked_outputs"):
            values = self.masked_outputs.flatten().tolist()
            finite_values = [v for v in values if v != float("-inf")]
            if finite_values:
                pieces.append(f"max_activation={max(finite_values):.4f}")
                pieces.append(f"min_activation={min(finite_values):.4f}")
                pieces.append(f"legal_points={len(finite_values)}")

        return "; ".join(str(item) for item in pieces if item is not None)

    def _request_deepseek_reply(self, context) -> str:
        fallback_msg = str(self.msg)
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            return fallback_msg

        identity_prompt = get_identity_prompt(self.name)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": identity_prompt},
                {
                    "role": "user",
                    "content": f"请基于以下局面摘要给出回应：{context}",
                },
            ],
            "stream": False,
            "temperature": 0.7,
        }

        req = requests.post(
            url="https://api.deepseek.com/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        
        return req.json()["choices"][0]["message"]["content"]

    @property
    def deepseek_msg(self):
        # return self._request_deepseek_reply()
        return ""
