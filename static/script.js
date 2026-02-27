// ==========================================================
// 全局状态与 Socket
// ==========================================================
let currentColor; 
let power;
let spectatorMode = false;

let socket = io.connect(
    window.location.protocol + '//' + window.location.host,
    { secure: window.location.protocol === 'https:' }
);


// ==========================================================
// DOMContentLoaded: 初始化入口
// ==========================================================
document.addEventListener('DOMContentLoaded', function () {

    bindUIButtons();
    bindConfigControls();
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
    const chessboard = document.getElementById('chessboard');
    chessboard.innerHTML = '';

    for (let i = 0; i < row_len; i++) {
        let row = chessboard.insertRow();
        for (let j = 0; j < col_len; j++) {
            let cell = row.insertCell();
            cell.addEventListener('click', () => onCellClick(i, j));
        }
    }
}

function updateChessboard(board, move) {
    const chessboard = document.getElementById('chessboard');

    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {

            let cell = chessboard.rows[i].cells[j];
            let value = board[i][j][0];
            let load = board[i][j][1];

            cell.innerHTML = '';
            cell.style.backgroundColor = getColor(value);

            let numberSpan = document.createElement('span');
            numberSpan.style.position = 'absolute';
            numberSpan.style.top = '50%';
            numberSpan.style.left = '50%';
            numberSpan.style.transform = 'translate(-50%, -50%)';
            numberSpan.style.zIndex = '1';

            numberSpan.textContent = (value === "inf") ? '∞'
                                 : load;

            cell.appendChild(numberSpan);

            if (move && arraysEqual(move, [i, j])) {
                drawDiamond(cell);
            }
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

function arraysEqual(a, b) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
}


// ==========================================================
// 特效：落子菱形标记
// ==========================================================
function drawDiamond(cell) {
    let diamond = document.createElement('div');
    diamond.style.position = 'absolute';
    diamond.style.width = '70%';
    diamond.style.height = '70%';
    diamond.style.border = '2px solid #FAFAD2';
    diamond.style.boxSizing = 'border-box';
    diamond.style.transform = 'translate(-50%, -50%) rotate(45deg)';
    diamond.style.top = '50%';
    diamond.style.left = '50%';

    cell.appendChild(diamond);
}


// ==========================================================
// UI 状态：颜色、分数
// ==========================================================
function toggleColor(step) {
    currentColor = (step % 2 === 0) ? 1 : -1;
    updatePlayerIndicator();
}

function updatePlayerIndicator() {
    const celestialText = document.getElementById("celestial");
    celestialText.style.color = (currentColor === 1) ? "#6495ED" : "lightcoral";
}

function updateTotalScore(score) {
    const scoreEl = document.getElementById("Score");
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

function toggleCmdPanel() {
    cmdPanel.style.display =
        (cmdPanel.style.display === "none") ? "flex" : "none";
}

function cmdPrint(msg) {
    cmdOutput.textContent += msg + "\n";
    cmdOutput.scrollTop = cmdOutput.scrollHeight;
}

// UI 锁定管理
let uiLocked = false;
function setUILocked(locked) {
    uiLocked = locked;
    document.body.style.cursor = locked ? "wait" : "default";
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
    if (e.key === "Enter") {
        let text = cmdInput.value.trim();
        if (text.length > 0) {
            // 本地显示
            cmdPrint("I: " + text);

            // 发送给后端
            socket.emit("cmd_input", { text });
        }
        cmdInput.value = "";
    }
});


// ==========================================================
// CMD 可拖拽
// ==========================================================
(function enableCmdDrag() {

    const panel = cmdPanel;
    let offsetX = 0, offsetY = 0;
    let dragging = false;

    panel.addEventListener("mousedown", function(e) {
        dragging = true;
        offsetX = e.clientX - panel.offsetLeft;
        offsetY = e.clientY - panel.offsetTop;
        panel.style.cursor = "grabbing";
    });

    document.addEventListener("mousemove", function(e) {
        if (dragging) {
            panel.style.left = (e.clientX - offsetX) + "px";
            panel.style.top  = (e.clientY - offsetY) + "px";
        }
    });

    document.addEventListener("mouseup", function() {
        dragging = false;
        panel.style.cursor = "move";
    });

})();
