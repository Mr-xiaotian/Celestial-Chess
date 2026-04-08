from flask import jsonify, render_template, request


def register_http_routes(app, session):
    @app.route("/")
    def index():
        """渲染主页面。"""
        return render_template("index.html")

    @app.route("/init_state", methods=["GET"])
    def init_state():
        """返回前端初始化所需的完整棋局状态。"""
        session.schedule_analysis_if_needed()
        return jsonify(session.get_init_state())

    @app.route("/configure", methods=["POST"])
    def configure():
        """
        更新棋盘配置。

        请求体:
            {
                "row_len": int,
                "col_len": int,
                "power": int
            }

        返回:
            - 成功: 更新后的初始化状态
            - 失败: {"error": "invalid config"}, 400
        """
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
