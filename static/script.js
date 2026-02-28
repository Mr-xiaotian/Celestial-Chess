// ==========================================================
// 全局状态与 Socket
// ==========================================================
let currentColor; 
let power;
let spectatorMode = false;
let cellRefs = [];
let valueRefs = [];
let diamondRefs = [];

let socket = io.connect(
    window.location.protocol + '//' + window.location.host,
    { secure: window.location.protocol === 'https:' }
);
const chessboardEl = document.getElementById('chessboard');
const celestialText = document.getElementById("celestial");
const scoreEl = document.getElementById("Score");


// ==========================================================
// DOMContentLoaded: 初始化入口
// ==========================================================
document.addEventListener('DOMContentLoaded', function () {

    bindUIButtons();
    bindConfigControls();
    bindChessboardClick();
    initStateFromBackend();
    bindSocketEvents();

});


// ==========================================================
// 初始化 & UI 按钮绑定
// ==========================================================
function bindUIButtons() {
    const clickBind = (id, eventName) =>
        document.getElementById(id).addEventListener("click", () => {
            if (uiLocked) return;
            if (spectatorMode) return;
            socket.emit(eventName);
        });

    clickBind("undoButton", 'undo_move');
    clickBind("redoButton", 'redo_move');
    clickBind("restartButton", 'restart_game');

    clickBind("minimaxButton", 'minimax_move');
    clickBind("mctsButton", 'mcts_move');
    clickBind("monkyButton", 'monky_move');
    // clickBind("dlButton", 'dl_move');

    clickBind("minimaxAuto", 'minimax_auto');
    clickBind("mctsAuto", 'mcts_auto');
    clickBind("monkyAuto", 'monky_auto');
    // clickBind("dlAuto", 'dl_auto');

    // CMD 开关
    document.getElementById("toggleCmd").addEventListener("click", toggleCmdPanel);
}

function bindChessboardClick() {
    chessboardEl.addEventListener('click', (event) => {
        const cell = event.target.closest('td');
        if (!cell || !chessboardEl.contains(cell)) return;
        const row = Number(cell.dataset.row);
        const col = Number(cell.dataset.col);
        if (!Number.isInteger(row) || !Number.isInteger(col)) return;
        onCellClick(row, col);
    });
}

function bindConfigControls() {
    document.getElementById("openConfig").addEventListener("click", () => setConfigOverlay(true));
    document.getElementById("closeConfig").addEventListener("click", () => setConfigOverlay(false));
    document.getElementById("configOverlay").addEventListener("click", (e) => {
        if (e.target.id === "configOverlay") {
            setConfigOverlay(false);
        }
    });

    document.getElementById("applyConfig").addEventListener("click", applyConfig);
    document.getElementById("startSpectator").addEventListener("click", startSpectator);
    document.getElementById("stopSpectator").addEventListener("click", stopSpectator);

    ["Blue", "Red"].forEach(side => {
        const typeSelect = document.getElementById(`spectator${side}Type`);
        typeSelect.addEventListener("change", () => updateAiConfigVisibility(side));
        updateAiConfigVisibility(side);
    });
}


// ==========================================================
// 后端初始状态
// ==========================================================
function initStateFromBackend() {
    fetch('/init_state')
        .then(response => response.json())
        .then(data => {
            power = data.power;

            renderChessboard(data.row_len, data.col_len);
            updateChessboard(data.board, data.move);
            updateTotalScore(0);
            toggleColor(0);
            setConfigInputs(data.row_len, data.col_len, data.power);
        })
        .catch(error => console.error('InitError:', error));
}


// ==========================================================
// Socket 事件绑定（主要：update_board + cmd_log）
// ==========================================================
function bindSocketEvents() {

    socket.on('update_board', function (data) {

        if (data.error) {
            console.error('Error:', data.error);
            return;
        }

        updateChessboard(data.board, data.move);
        updateTotalScore(data.score);
        toggleColor(data.step);

        // 游戏结束提示
        if (data.game_over) {
            let msg = (data.winner === 1) ? "蓝方胜利！"
                    : (data.winner === -1) ? "红方胜利！"
                    : "平局！";

            setTimeout(() => {
                alert("游戏结束\n" + msg);
            }, 100);
        }
    });

    // 后端主动发送的 CMD 文本
    socket.on("cmd_log", data => {
        cmdPrint(data.msg);
    });

    // AI 思考中：锁定 / 解锁 UI
    socket.on("ai_thinking", data => {
        setUILocked(data && data.status === "start");
    });

    socket.on("spectator_status", data => {
        const status = data && data.status ? data.status : "stop";
        spectatorMode = status === "start";
        updateSpectatorStatusText(status, data);
    });

    socket.on("config_changed", data => {
        if (!data || data.source !== "cmd") {
            return;
        }
        power = data.power;
        renderChessboard(data.row_len, data.col_len);
        updateChessboard(data.board, data.move);
        updateTotalScore(data.score ?? 0);
        toggleColor(data.step ?? 0);
        setConfigInputs(data.row_len, data.col_len, data.power);
    });
}


// ==========================================================
// 棋盘渲染相关函数
// ==========================================================
function renderChessboard(row_len, col_len) {
    chessboardEl.innerHTML = '';
    cellRefs = Array.from({ length: row_len }, () => new Array(col_len));
    valueRefs = Array.from({ length: row_len }, () => new Array(col_len));
    diamondRefs = Array.from({ length: row_len }, () => new Array(col_len));

    const fragment = document.createDocumentFragment();
    for (let i = 0; i < row_len; i++) {
        const row = document.createElement('tr');
        for (let j = 0; j < col_len; j++) {
            const cell = document.createElement('td');
            cell.dataset.row = String(i);
            cell.dataset.col = String(j);

            const numberSpan = document.createElement('span');
            numberSpan.className = 'cell-value';

            const diamond = document.createElement('div');
            diamond.className = 'cell-diamond';

            cell.appendChild(numberSpan);
            cell.appendChild(diamond);
            row.appendChild(cell);
            cellRefs[i][j] = cell;
            valueRefs[i][j] = numberSpan;
            diamondRefs[i][j] = diamond;
        }
        fragment.appendChild(row);
    }
    chessboardEl.appendChild(fragment);
}

function updateChessboard(board, move) {
    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {

            const cell = cellRefs[i]?.[j];
            const numberSpan = valueRefs[i]?.[j];
            const diamond = diamondRefs[i]?.[j];
            if (!cell || !numberSpan || !diamond) continue;

            const value = board[i][j][0];
            const load = board[i][j][1];

            cell.style.backgroundColor = getColor(value);
            numberSpan.textContent = (value === "inf") ? '∞' : load;
            diamond.style.display = move && move[0] === i && move[1] === j ? 'block' : 'none';
        }
    }
}


// ==========================================================
// 落子颜色、背景计算
// ==========================================================
function getColor(value) {
    if (value === "inf") {
        return 'lightgrey';
    }

    let alpha = Math.min(Math.abs(value) / 5, 1);

    return value > 0 ? `rgba(173, 216, 230, ${alpha})`
         : value < 0 ? `rgba(240, 128, 128, ${alpha})`
         : '';
}

// ==========================================================
// UI 状态：颜色、分数
// ==========================================================
function toggleColor(step) {
    currentColor = (step % 2 === 0) ? 1 : -1;
    updatePlayerIndicator();
}

function updatePlayerIndicator() {
    celestialText.style.color = (currentColor === 1) ? "#6495ED" : "lightcoral";
}

function updateTotalScore(score) {
    scoreEl.textContent = score;
    scoreEl.style.color =
        score > 0 ? "lightblue" :
        score < 0 ? "lightcoral" : "black";
}

function onCellClick(row, col) {
    if (uiLocked || spectatorMode) return;
    socket.emit('play_move', { row, col, color: currentColor });
}


// ==========================================================
// CMD Terminal 功能块：开关 / 打印 / 输入 
// ==========================================================
const cmdPanel  = document.getElementById("cmd-panel");
const cmdOutput = document.getElementById("cmd-output");
const cmdInput  = document.getElementById("cmd-input");
const cmdHeader = document.getElementById("cmd-header");
const cmdCloseBtn = document.getElementById("cmd-close");
const cmdMinBtn = document.getElementById("cmd-minimize");
const cmdMaxBtn = document.getElementById("cmd-maximize");
const cmdHistory = [];
let cmdHistoryIndex = -1;
let cmdDraft = "";

function toggleCmdPanel() {
    cmdPanel.style.display =
        (cmdPanel.style.display === "none") ? "flex" : "none";
}

function cmdPrint(msg, options = {}) {
    const line = document.createElement("div");
    const type = options.type || detectCmdType(msg);
    line.classList.add("cmd-output-line", type);
    line.textContent = msg;
    cmdOutput.appendChild(line);
    cmdOutput.scrollTop = cmdOutput.scrollHeight;
}

function setCmdInputValue(value) {
    cmdInput.value = value;
    cmdInput.setSelectionRange(value.length, value.length);
}

function resetCmdHistoryNav() {
    cmdHistoryIndex = -1;
    cmdDraft = "";
}

// UI 锁定管理
let uiLocked = false;
function setUILocked(locked) {
    uiLocked = locked;
    document.body.style.cursor = locked ? "wait" : "default";
}

function detectCmdType(msg) {
    if (msg.startsWith("User: ")) {
        return "user";
    }
    if (msg.startsWith("(") || msg.includes("thinking") || msg.includes("配置已更新") || msg.includes("观战已开始") || msg.includes("观战已停止")) {
        return "system";
    }
    const aiMatch = msg.match(/^([A-Za-z]+AI|MCTSAI|MinimaxAI|MonkyAI|MCTS|Minimax|Monky)\s*:/);
    if (aiMatch) {
        const name = aiMatch[1].toLowerCase();
        if (name.includes("minimax")) return "ai-minimax";
        if (name.includes("mcts")) return "ai-mcts";
        if (name.includes("monky")) return "ai-monky";
        return "ai-generic";
    }
    return "system";
}

function setCmdCollapsed(collapsed) {
    if (collapsed) {
        cmdPanel.classList.add("minimized");
    } else {
        cmdPanel.classList.remove("minimized");
    }
}

function toggleCmdMaximize() {
    cmdPanel.classList.toggle("maximized");
}

function setConfigInputs(rowLen, colLen, powerValue) {
    document.getElementById("configRowLen").value = rowLen;
    document.getElementById("configColLen").value = colLen;
    document.getElementById("configPower").value = powerValue;
}

function setSpectatorSleepInput(value) {
    const sleepInput = document.getElementById("spectatorSleep");
    if (sleepInput) {
        sleepInput.value = value;
    }
}

function setConfigOverlay(open) {
    const overlay = document.getElementById("configOverlay");
    if (open) {
        overlay.classList.add("active");
    } else {
        overlay.classList.remove("active");
    }
}

function applyConfig() {
    if (uiLocked) return;
    const rowLen = parseInt(document.getElementById("configRowLen").value, 10);
    const colLen = parseInt(document.getElementById("configColLen").value, 10);
    const powerValue = parseInt(document.getElementById("configPower").value, 10);

    if (!Number.isInteger(rowLen) || !Number.isInteger(colLen) || !Number.isInteger(powerValue)) {
        alert("配置参数无效");
        return;
    }

    fetch('/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row_len: rowLen, col_len: colLen, power: powerValue })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("配置失败");
                return;
            }
            power = data.power;
            renderChessboard(data.row_len, data.col_len);
            updateChessboard(data.board, data.move);
            updateTotalScore(0);
            toggleColor(0);
            setConfigInputs(data.row_len, data.col_len, data.power);
        })
        .catch(() => alert("配置失败"));
}

function getSpectatorConfig(side) {
    const type = document.getElementById(`spectator${side}Type`).value;
    const minimaxDepth = parseInt(document.getElementById(`spectator${side}Depth`).value, 10);
    const mctsIter = parseInt(document.getElementById(`spectator${side}Mcts`).value, 10);
    return {
        type,
        minimax_depth: Number.isInteger(minimaxDepth) ? minimaxDepth : 2,
        mcts_iter: Number.isInteger(mctsIter) ? mctsIter : 1000
    };
}

function getSpectatorSleep() {
    const sleepValue = parseFloat(document.getElementById("spectatorSleep").value);
    return Number.isFinite(sleepValue) && sleepValue >= 0 ? sleepValue : 0;
}

function startSpectator() {
    if (uiLocked) return;
    const blue = getSpectatorConfig("Blue");
    const red = getSpectatorConfig("Red");
    const sleep = getSpectatorSleep();
    socket.emit("start_spectator", { blue, red, sleep });
}

function stopSpectator() {
    socket.emit("stop_spectator");
}

function updateSpectatorStatusText(status, data) {
    const statusEl = document.getElementById("spectatorStatus");
    statusEl.textContent = status === "start" ? "观战模式" : "对弈模式";
    if (status === "stop") {
        spectatorMode = false;
    }
    if (data && data.sleep !== undefined && data.sleep !== null) {
        setSpectatorSleepInput(data.sleep);
    }
    if (data && data.blue && data.red) {
        setSpectatorConfigInputs(data.blue, data.red);
    }
}

function setSpectatorConfigInputs(blue, red) {
    const blueType = document.getElementById("spectatorBlueType");
    const redType = document.getElementById("spectatorRedType");
    if (blueType && blue && blue.type) {
        blueType.value = blue.type;
    }
    if (redType && red && red.type) {
        redType.value = red.type;
    }
    if (blue && blue.minimax_depth !== undefined) {
        document.getElementById("spectatorBlueDepth").value = blue.minimax_depth;
    }
    if (red && red.minimax_depth !== undefined) {
        document.getElementById("spectatorRedDepth").value = red.minimax_depth;
    }
    if (blue && blue.mcts_iter !== undefined) {
        document.getElementById("spectatorBlueMcts").value = blue.mcts_iter;
    }
    if (red && red.mcts_iter !== undefined) {
        document.getElementById("spectatorRedMcts").value = red.mcts_iter;
    }
    updateAiConfigVisibility("Blue");
    updateAiConfigVisibility("Red");
}

function updateAiConfigVisibility(side) {
    const type = document.getElementById(`spectator${side}Type`).value;
    const depthInput = document.getElementById(`spectator${side}Depth`);
    const mctsInput = document.getElementById(`spectator${side}Mcts`);
    depthInput.style.display = type === "minimax" ? "inline-block" : "none";
    mctsInput.style.display = type === "mcts" ? "inline-block" : "none";
}

cmdInput.addEventListener("keydown", function(e) {
    if (e.key === "ArrowUp") {
        if (!cmdHistory.length) return;
        e.preventDefault();
        if (cmdHistoryIndex === -1) {
            cmdDraft = cmdInput.value;
            cmdHistoryIndex = cmdHistory.length - 1;
        } else if (cmdHistoryIndex > 0) {
            cmdHistoryIndex -= 1;
        }
        setCmdInputValue(cmdHistory[cmdHistoryIndex]);
        return;
    }

    if (e.key === "ArrowDown") {
        if (cmdHistoryIndex === -1) return;
        e.preventDefault();
        if (cmdHistoryIndex < cmdHistory.length - 1) {
            cmdHistoryIndex += 1;
            setCmdInputValue(cmdHistory[cmdHistoryIndex]);
        } else {
            resetCmdHistoryNav();
            setCmdInputValue(cmdDraft);
        }
        return;
    }

    if (e.key.toLowerCase() === "l" && e.ctrlKey) {
        e.preventDefault();
        cmdOutput.innerHTML = "";
        return;
    }

    if (e.key === "Escape") {
        cmdInput.value = "";
        resetCmdHistoryNav();
        return;
    }

    if (e.key === "Enter") {
        let text = cmdInput.value.trim();
        if (text.length > 0) {
            // 本地显示
            cmdPrint("User: " + text, { type: "user" });

            // 发送给后端
            socket.emit("cmd_input", { text });
            cmdHistory.push(text);
        }
        cmdInput.value = "";
        resetCmdHistoryNav();
    }
});


// ==========================================================
// CMD 可拖拽
// ==========================================================
(function enableCmdDrag() {
    const panel = cmdPanel;
    let offsetX = 0, offsetY = 0;
    let dragging = false;

    cmdHeader.addEventListener("mousedown", function(e) {
        if (e.target.classList.contains("cmd-dot")) return;
        dragging = true;
        offsetX = e.clientX - panel.offsetLeft;
        offsetY = e.clientY - panel.offsetTop;
        cmdHeader.style.cursor = "grabbing";
    });

    document.addEventListener("mousemove", function(e) {
        if (dragging) {
            panel.style.left = (e.clientX - offsetX) + "px";
            panel.style.top  = (e.clientY - offsetY) + "px";
        }
    });

    document.addEventListener("mouseup", function() {
        dragging = false;
        cmdHeader.style.cursor = "grab";
    });
})();

cmdHeader.addEventListener("dblclick", function() {
    toggleCmdMaximize();
});

cmdCloseBtn.addEventListener("click", function() {
    cmdPanel.style.display = "none";
});

cmdMinBtn.addEventListener("click", function() {
    setCmdCollapsed(!cmdPanel.classList.contains("minimized"));
});

cmdMaxBtn.addEventListener("click", function() {
    toggleCmdMaximize();
});
