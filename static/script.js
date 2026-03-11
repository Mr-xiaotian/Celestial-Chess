// ==========================================================
// 全局状态与 Socket
// ==========================================================
let currentColor; 
let power;
let spectatorMode = false;
let cellRefs = [];
let valueRefs = [];
let diamondRefs = [];
let currentDisplayMode = "no-zero"; // 默认显示模式：归一
let lastBoardData = null;
let bluePastelMaxLevel = 50;
let redPastelMaxLevel = 50;
let playerMode = "pvp";
let currentColorScheme = "pastel";
let analysisRecommendationEnabled = false;
let latestAnalysisData = null;
let analysisRefs = [];
let currentStep = 0;

const COLOR_RANGE_MAX = 100;
const WHITE_RGB = [255, 255, 255];
const COLOR_SCHEMES = {
    pastel: {
        blueMin: [235, 245, 255], // #ebf5ff
        blueMax: [129, 212, 250], // #81d4fa
        redMin: [255, 238, 238],  // #ffeeee
        redMax: [229, 115, 115]   // #e57373
    },
    morandi: {
        blueMin: [236, 240, 241], // #ecf0f1
        blueMax: [130, 150, 161], // #8296a1
        redMin: [244, 236, 234],  // #f4ecea
        redMax: [177, 134, 130]   // #b18682
    },
    neon: {
        blueMin: [229, 247, 255], // #e5f7ff
        blueMax: [0, 191, 255],   // #00bfff
        redMin: [255, 233, 244],  // #ffe9f4
        redMax: [255, 64, 129]    // #ff4081
    }
};

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
    initCustomSelects();
    bindChessboardClick();
    initStateFromBackend();
    bindSocketEvents();

});


// ==========================================================
// 初始化 & UI 按钮绑定
// ==========================================================

/**
 * 绑定 UI 按钮的点击事件。
 * 包括悔棋、重悔、重开游戏、AI 执棋和 AI 对弈等功能按钮。
 * 同时绑定 CMD 面板的开关按钮。
 */
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

/**
 * 绑定棋盘点击事件（使用事件委托）。
 * 监听棋盘元素的点击，根据点击的目标单元格触发落子逻辑。
 */
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

/**
 * 绑定配置面板相关的控件事件。
 * 包括打开/关闭配置面板、应用配置、观战模式控制以及 AI 配置可见性切换。
 */
function bindConfigControls() {
    document.getElementById("openConfig").addEventListener("click", () => setConfigOverlay(true));
    document.getElementById("closeConfig").addEventListener("click", () => setConfigOverlay(false));
    document.getElementById("configOverlay").addEventListener("click", (e) => {
        if (e.target.id === "configOverlay") {
            setConfigOverlay(false);
        }
    });

    document.getElementById("applyConfig").addEventListener("click", applyConfig);
    document.getElementById("applyRoleConfig").addEventListener("click", applyRoleConfig);
    document.getElementById("configCmdVisible").addEventListener("change", (e) => {
        setCmdPanelVisible(e.target.value === "show");
    });
    setCmdPanelVisible(document.getElementById("configCmdVisible").value === "show");
    document.getElementById("toggleAnalysisRecommendations").addEventListener("click", () => {
        setAnalysisRecommendationEnabled(!analysisRecommendationEnabled);
    });
    document.getElementById("analysisIter").addEventListener("change", (e) => {
        applyAnalysisIter(parseInt(e.target.value, 10));
    });
    setAnalysisRecommendationEnabled(false);

    document.getElementById("displayMode").addEventListener("change", (e) => {
        currentDisplayMode = e.target.value;
        if (lastBoardData) {
            updateChessboard(lastBoardData.board, lastBoardData.move);
        }
    });

    const blueRange = document.getElementById("bluePastelMax");
    const redRange = document.getElementById("redPastelMax");
    if (blueRange) {
        blueRange.addEventListener("input", (e) => setPastelMaxLevel("blue", e.target.value));
    }
    if (redRange) {
        redRange.addEventListener("input", (e) => setPastelMaxLevel("red", e.target.value));
    }
    setPastelMaxLevel("blue", blueRange ? blueRange.value : bluePastelMaxLevel);
    setPastelMaxLevel("red", redRange ? redRange.value : redPastelMaxLevel);
    const colorSchemeSelect = document.getElementById("colorScheme");
    if (colorSchemeSelect) {
        colorSchemeSelect.addEventListener("change", (e) => applyColorScheme(e.target.value));
        applyColorScheme(colorSchemeSelect.value);
    }

    ["Blue", "Red"].forEach(side => {
        const roleSelect = document.getElementById(`spectator${side}Role`);
        roleSelect.addEventListener("change", () => updateAiConfigVisibility(side));
        updateAiConfigVisibility(side);
    });
}

 


// ==========================================================
// 后端初始状态
// ==========================================================

/**
 * 从后端获取初始游戏状态并初始化前端界面。
 * 发送 GET 请求到 /init_state，获取棋盘大小、棋子布局、分数等信息。
 */
function initStateFromBackend() {
    fetch('/init_state')
        .then(response => response.json())
        .then(data => {
            applyBackendState(data);
        })
        .catch(error => console.error('InitError:', error));
}


// ==========================================================
// Socket 事件绑定（主要：update_board + cmd_log）
// ==========================================================

/**
 * 绑定 Socket.IO 事件监听器。
 * 处理后端推送的棋盘更新、CMD 日志、AI 思考状态、观战状态及配置变更。
 */
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
        cmdPrint(data.msg, { type: data && data.type ? data.type : "system" });
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

    socket.on("role_status", data => {
        applyRoleStatus(data || {});
    });

    socket.on("config_changed", data => {
        if (!data || data.source !== "cmd") {
            return;
        }
        applyBackendState(data);
    });

    socket.on("analysis_update", data => {
        applyAnalysisUpdate(data || {});
    });

    socket.on("analysis_config", data => {
        applyAnalysisConfig(data || {});
    });
}

/**
 * 统一应用后端返回的棋盘与配置状态。
 * 
 * @param {object} data - 后端状态数据
 */
function applyBackendState(data) {
    power = data.power;
    renderChessboard(data.row_len, data.col_len);
    updateChessboard(data.board, data.move);
    updateTotalScore(data.score ?? 0);
    toggleColor(data.step ?? 0);
    setConfigInputs(data.row_len, data.col_len, data.power);
    if (data && Number.isInteger(data.analysis_iter)) {
        setAnalysisIterInput(data.analysis_iter);
    }
    if (data && data.mode && data.blue && data.red) {
        applyRoleStatus({
            mode: data.mode,
            sleep: data.sleep,
            blue: data.blue,
            red: data.red
        });
    }
}


// ==========================================================
// 棋盘渲染相关函数
// ==========================================================

/**
 * 渲染棋盘网格结构。
 * 根据指定的行数和列数创建表格行和单元格，并初始化 DOM 引用缓存。
 * 
 * @param {number} row_len - 棋盘行数
 * @param {number} col_len - 棋盘列数
 */
function renderChessboard(row_len, col_len) {
    chessboardEl.innerHTML = '';
    cellRefs = Array.from({ length: row_len }, () => new Array(col_len));
    valueRefs = Array.from({ length: row_len }, () => new Array(col_len));
    diamondRefs = Array.from({ length: row_len }, () => new Array(col_len));
    analysisRefs = Array.from({ length: row_len }, () => new Array(col_len));

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

            const analysisSpan = document.createElement('div');
            analysisSpan.className = 'cell-analysis';

            cell.appendChild(numberSpan);
            cell.appendChild(diamond);
            cell.appendChild(analysisSpan);
            row.appendChild(cell);
            cellRefs[i][j] = cell;
            valueRefs[i][j] = numberSpan;
            diamondRefs[i][j] = diamond;
            analysisRefs[i][j] = analysisSpan;
        }
        fragment.appendChild(row);
    }
    chessboardEl.appendChild(fragment);
}

/**
 * 更新棋盘状态。
 * 根据传入的棋盘数据更新每个单元格的数值、背景颜色和高亮标记。
 * 
 * @param {Array<Array<Array<number|string>>>} board - 棋盘数据，每个元素包含 [value, load]
 * @param {Array<number>|null} move - 当前落子位置 [row, col]，用于显示高亮标记
 */
function updateChessboard(board, move) {
    lastBoardData = { board, move };
    const analysisMap = buildAnalysisMap();

    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {

            const cell = cellRefs[i]?.[j];
            const numberSpan = valueRefs[i]?.[j];
            const diamond = diamondRefs[i]?.[j];
            const analysisSpan = analysisRefs[i]?.[j];
            if (!cell || !numberSpan || !diamond || !analysisSpan) continue;

            const value = board[i][j][0];
            const load = board[i][j][1];

            cell.style.backgroundColor = getColor(value);
            
            let displayText = load;
            
            if (currentDisplayMode === "all") {
                // 万象：显示所有数字
                if (value === "inf") {
                    displayText = "∞";
                } 
            } else if (currentDisplayMode === "no-zero") {
                // 归一：隐藏零值
                if (value === "inf" || load === 0) {
                    displayText = "";
                } 
            } else if (currentDisplayMode === "none") {
                // 无为：不显示任何数字
                displayText = "";
            }
            
            numberSpan.textContent = displayText;
            
            diamond.style.display = move && move[0] === i && move[1] === j ? 'block' : 'none';
            const analysisCell = analysisMap[`${i}-${j}`];
            const isPlayableCell = value !== "inf" && Number(value) === 0;
            if (analysisRecommendationEnabled && isPlayableCell && analysisCell) {
                analysisSpan.style.display = "block";
                analysisSpan.textContent = `#${analysisCell.rank} ${(analysisCell.win_rate * 100).toFixed(1)}%`;
                analysisSpan.classList.toggle("cell-analysis-top3", analysisCell.rank <= 3);
                analysisSpan.style.color = analysisCell.rank <= 3 ? getCelestialColor() : "#0f172a";
            } else {
                analysisSpan.style.display = "none";
                analysisSpan.textContent = "";
                analysisSpan.classList.remove("cell-analysis-top3");
                analysisSpan.style.color = "#0f172a";
            }
        }
    }
}


// ==========================================================
// 落子颜色、背景计算
// ==========================================================

/**
 * 根据棋盘格子的值计算背景颜色。
 * 
 * @param {number|string} value - 格子的值
 * @returns {string} CSS 颜色字符串
 */
function getColor(value) {
    if (value === "inf") {
        return 'lightgrey';
    }
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue) || numericValue === 0) {
        return '';
    }
    const cellColorMaxValue = Math.max((Number.isFinite(power) ? power : 0) * 2, 1);
    const factor = Math.min(Math.abs(numericValue) / cellColorMaxValue, 1);
    const targetColor = getTargetPastelColor(numericValue > 0 ? "blue" : "red");
    const finalColor = interpolateColor(WHITE_RGB, targetColor, factor);
    return `rgb(${finalColor.join(",")})`;
}

// ==========================================================
// UI 状态：颜色、分数
// ==========================================================

/**
 * 切换当前执棋方颜色指示。
 * 
 * @param {number} step - 当前步数，偶数为蓝方，奇数为红方
 */
function toggleColor(step) {
    currentStep = Number.isInteger(step) ? step : 0;
    currentColor = (step % 2 === 0) ? 1 : -1;
    updatePlayerIndicator();
}

/**
 * 更新界面上的执棋方文字颜色。
 */
function updatePlayerIndicator() {
    celestialText.style.color = (currentColor === 1) ? "#6495ED" : "lightcoral";
}

/**
 * 更新总分显示及动态背景。
 * 
 * @param {number} score - 当前总分
 */
function updateTotalScore(score) {
    scoreEl.textContent = score;
    scoreEl.style.color =
        score > 0 ? "lightblue" :
        score < 0 ? "lightcoral" : "black";

    updateDynamicBackground(score);
}

/**
 * 更新动态背景颜色
 * 
 * @param {number} score - 当前总分
 */
function updateDynamicBackground(score) {
    const bgElement = document.getElementById("dynamic-background");
    if (!bgElement) return;

    if (score === 0) {
        bgElement.style.backgroundColor = "#FFFFFF";
        return;
    }
    const factor = Math.min(Math.abs(score) / COLOR_RANGE_MAX, 1);
    const targetColor = getTargetPastelColor(score > 0 ? "blue" : "red");
    const finalColor = interpolateColor(WHITE_RGB, targetColor, factor);
    bgElement.style.backgroundColor = `rgb(${finalColor.join(",")})`;
}

/**
 * 颜色插值辅助函数
 */
function interpolateColor(color1, color2, factor) {
    const r = Math.round(color1[0] + (color2[0] - color1[0]) * factor);
    const g = Math.round(color1[1] + (color2[1] - color1[1]) * factor);
    const b = Math.round(color1[2] + (color2[2] - color1[2]) * factor);
    return [r, g, b];
}

/**
 * 处理单元格点击逻辑。
 * 如果 UI 未锁定且非观战模式，则发送落子指令。
 * 
 * @param {number} row - 行索引
 * @param {number} col - 列索引
 */
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

/**
 * 切换 CMD 面板的显示/隐藏状态。
 */
function toggleCmdPanel() {
    setCmdPanelVisible(cmdPanel.style.display === "none");
}

function setCmdPanelVisible(visible) {
    cmdPanel.style.display = visible ? "flex" : "none";
    const select = document.getElementById("configCmdVisible");
    if (select) {
        select.value = visible ? "show" : "hide";
        syncCustomSelect(select);
    }
}

/**
 * 在 CMD 面板中打印消息。
 * 
 * @param {string} msg - 消息内容
 * @param {object} options - 配置项，如消息类型
 */
function cmdPrint(msg, options = {}) {
    const line = document.createElement("div");
    const type = options.type || "system";
    line.classList.add("cmd-output-line", type);
    line.textContent = msg;
    cmdOutput.appendChild(line);
    cmdOutput.scrollTop = cmdOutput.scrollHeight;
}

/**
 * 设置 CMD 输入框的值并移动光标到末尾。
 * 
 * @param {string} value - 输入框的新值
 */
function setCmdInputValue(value) {
    cmdInput.value = value;
    cmdInput.setSelectionRange(value.length, value.length);
}

/**
 * 重置 CMD 历史导航状态。
 */
function resetCmdHistoryNav() {
    cmdHistoryIndex = -1;
    cmdDraft = "";
}

// UI 锁定管理
let uiLocked = false;

/**
 * 设置 UI 锁定状态。
 * 锁定期间鼠标变为等待样式。
 * 
 * @param {boolean} locked - 是否锁定
 */
function setUILocked(locked) {
    uiLocked = locked;
    document.body.style.cursor = locked ? "wait" : "default";
}

/**
 * 设置 CMD 面板的折叠状态。
 * 
 * @param {boolean} collapsed - 是否折叠
 */
function setCmdCollapsed(collapsed) {
    if (collapsed) {
        cmdPanel.classList.add("minimized");
    } else {
        cmdPanel.classList.remove("minimized");
    }
}

/**
 * 切换 CMD 面板的最大化/还原状态。
 */
function toggleCmdMaximize() {
    cmdPanel.classList.toggle("maximized");
}

/**
 * 设置配置面板中的输入框值。
 * 
 * @param {number} rowLen - 棋盘行数
 * @param {number} colLen - 棋盘列数
 * @param {number} powerValue - 棋力值
 */
function setConfigInputs(rowLen, colLen, powerValue) {
    document.getElementById("configRowLen").value = rowLen;
    document.getElementById("configColLen").value = colLen;
    document.getElementById("configPower").value = powerValue;
}

/**
 * 获取指定方的目标 pastel 颜色。
 * 
 * @param {string} side - 方别，"blue" 或 "red"
 * @returns {Array} 目标 pastel 颜色的 RGB 数组
 */
function getTargetPastelColor(side) {
    const scheme = COLOR_SCHEMES[currentColorScheme] || COLOR_SCHEMES.pastel;
    const level = side === "blue" ? bluePastelMaxLevel : redPastelMaxLevel;
    const from = side === "blue" ? scheme.blueMin : scheme.redMin;
    const to = side === "blue" ? scheme.blueMax : scheme.redMax;
    return interpolateColor(from, to, Math.min(Math.max(level, 0), 100) / 100);
}

/**
 * 应用指定的颜色方案。
 * 更新 UI 元素的颜色样式。
 * 
 * @param {string} schemeKey - 颜色方案键
 */
function applyColorScheme(schemeKey) {
    currentColorScheme = COLOR_SCHEMES[schemeKey] ? schemeKey : "pastel";
    const scheme = COLOR_SCHEMES[currentColorScheme];
    const blueRange = document.getElementById("bluePastelMax");
    const redRange = document.getElementById("redPastelMax");
    const toRgb = (arr) => `rgb(${arr[0]}, ${arr[1]}, ${arr[2]})`;
    if (blueRange) {
        blueRange.style.setProperty("--range-start", toRgb(scheme.blueMin));
        blueRange.style.setProperty("--range-end", toRgb(scheme.blueMax));
    }
    if (redRange) {
        redRange.style.setProperty("--range-start", toRgb(scheme.redMin));
        redRange.style.setProperty("--range-end", toRgb(scheme.redMax));
    }
    const select = document.getElementById("colorScheme");
    if (select) {
        select.value = currentColorScheme;
        syncCustomSelect(select);
    }
    if (lastBoardData) {
        updateChessboard(lastBoardData.board, lastBoardData.move);
    }
    const score = Number(scoreEl.textContent);
    updateDynamicBackground(Number.isFinite(score) ? score : 0);
}

/**
 * 设置指定方的 pastel 颜色最大等级。
 * 更新 UI 元素的显示值。
 * 
 * @param {string} side - 方别，"blue" 或 "red"
 * @param {number} value - 最大等级，0-100 之间的整数
 */
function setPastelMaxLevel(side, value) {
    const parsed = Number(value);
    const normalized = Number.isFinite(parsed) ? Math.min(Math.max(parsed, 0), 100) : 100;
    if (side === "blue") {
        bluePastelMaxLevel = normalized;
    } else {
        redPastelMaxLevel = normalized;
    }
    const valueEl = document.getElementById(side === "blue" ? "bluePastelMaxValue" : "redPastelMaxValue");
    if (valueEl) {
        valueEl.textContent = `${normalized}%`;
    }
    if (lastBoardData) {
        updateChessboard(lastBoardData.board, lastBoardData.move);
    }
    const score = Number(scoreEl.textContent);
    updateDynamicBackground(Number.isFinite(score) ? score : 0);
}

/**
 * 设置观战模式的休眠时间输入框值。
 * 
 * @param {number} value - 休眠时间（秒）
 */
function setSpectatorSleepInput(value) {
    const sleepInput = document.getElementById("spectatorSleep");
    if (sleepInput) {
        sleepInput.value = value;
    }
}

/**
 * 设置配置面板的显示/隐藏状态。
 * 
 * @param {boolean} open - 是否显示
 */
function setConfigOverlay(open) {
    const overlay = document.getElementById("configOverlay");
    if (open) {
        overlay.classList.add("active");
    } else {
        overlay.classList.remove("active");
    }
}

/**
 * 应用新的游戏配置。
 * 获取用户输入的配置参数并发送到后端。
 */
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
            applyBackendState(data);
        })
        .catch(() => alert("配置失败"));
}

/**
 * 获取指定方的观战配置。
 * 
 * @param {string} side - "Blue" 或 "Red"
 * @returns {object} AI 配置对象
 */
function getSpectatorConfig(side) {
    const type = document.getElementById(`spectator${side}Role`).value;
    const minimaxDepth = parseInt(document.getElementById(`spectator${side}Depth`).value, 10);
    const mctsIter = parseInt(document.getElementById(`spectator${side}Mcts`).value, 10);
    return {
        type,
        minimax_depth: Number.isInteger(minimaxDepth) ? minimaxDepth : 2,
        mcts_iter: Number.isInteger(mctsIter) ? mctsIter : 1000
    };
}

function getRoleConfig(side) {
    const role = document.getElementById(`spectator${side}Role`).value;
    if (role === "human") {
        return { role: "human" };
    }
    return { role, ai: getSpectatorConfig(side) };
}

/**
 * 获取观战模式的休眠时间配置。
 * 
 * @returns {number} 休眠时间（秒）
 */
function getSpectatorSleep() {
    const sleepValue = parseFloat(document.getElementById("spectatorSleep").value);
    return Number.isFinite(sleepValue) && sleepValue >= 0 ? sleepValue : 0;
}

/**
 * 开始观战模式。
 * 收集配置并发送 start_spectator 事件。
 */
function applyRoleConfig() {
    if (uiLocked) return;
    const blue = getRoleConfig("Blue");
    const red = getRoleConfig("Red");
    const sleep = getSpectatorSleep();
    socket.emit("apply_roles", { blue, red, sleep });
}

/**
 * 更新观战状态文本和配置回显。
 * 
 * @param {string} status - 状态 ("start" 或 "stop")
 * @param {object} data - 配置数据
 */
function updateSpectatorStatusText(status, data) {
    const mode = status === "start" ? "spectator" : "pvp";
    applyRoleStatus({ mode, sleep: data ? data.sleep : 0, blue: data ? data.blue : null, red: data ? data.red : null });
}

/**
 * 设置观战配置输入框的值。
 * 
 * @param {object} blue - 蓝方配置
 * @param {object} red - 红方配置
 */
function setSpectatorConfigInputs(blue, red) {
    const blueRole = document.getElementById("spectatorBlueRole");
    const redRole = document.getElementById("spectatorRedRole");
    if (blueRole && blue && blue.role) {
        blueRole.value = blue.role;
    }
    if (redRole && red && red.role) {
        redRole.value = red.role;
    }
    if (blue && blue.ai && blue.ai.minimax_depth !== undefined) {
        document.getElementById("spectatorBlueDepth").value = blue.ai.minimax_depth;
    }
    if (red && red.ai && red.ai.minimax_depth !== undefined) {
        document.getElementById("spectatorRedDepth").value = red.ai.minimax_depth;
    }
    if (blue && blue.ai && blue.ai.mcts_iter !== undefined) {
        document.getElementById("spectatorBlueMcts").value = blue.ai.mcts_iter;
    }
    if (red && red.ai && red.ai.mcts_iter !== undefined) {
        document.getElementById("spectatorRedMcts").value = red.ai.mcts_iter;
    }
    updateAiConfigVisibility("Blue");
    updateAiConfigVisibility("Red");
    syncCustomSelect(blueRole);
    syncCustomSelect(redRole);
}

/**
 * 更新 AI 配置选项的可见性。
 * 根据选择的 AI 类型显示或隐藏特定的参数输入框。
 * 
 * @param {string} side - "Blue" 或 "Red"
 */
function updateAiConfigVisibility(side) {
    const role = document.getElementById(`spectator${side}Role`).value;
    const depthInput = document.getElementById(`spectator${side}Depth`);
    const mctsInput = document.getElementById(`spectator${side}Mcts`);
    const depthWrap = depthInput.parentElement;
    const mctsWrap = mctsInput.parentElement;
    depthWrap.style.display = role === "minimax" ? "inline-flex" : "none";
    mctsWrap.style.display = role === "mcts" ? "inline-flex" : "none";
}

/**
 * 更新观战状态文本和配置回显。
 * 
 * @param {string} status - 状态 ("start" 或 "stop")
 * @param {object} data - 配置数据
 */
function applyRoleStatus(data) {
    const mode = data && data.mode ? data.mode : "pvp";
    playerMode = mode;
    spectatorMode = mode === "spectator";
    const statusEl = document.getElementById("spectatorStatus");
    if (statusEl) {
        statusEl.textContent = mode === "spectator" ? "AI 对战模式" : mode === "pve" ? "PVE 模式" : "PVP 模式";
    }
    if (data && data.sleep !== undefined && data.sleep !== null) {
        setSpectatorSleepInput(data.sleep);
    }
    if (data && data.blue && data.red) {
        setSpectatorConfigInputs(data.blue, data.red);
    }
}

/**
 * 设置分析推荐功能的启用状态。
 * 
 * @param {boolean} enabled - 是否启用分析推荐
 */
function setAnalysisRecommendationEnabled(enabled) {
    analysisRecommendationEnabled = Boolean(enabled);
    const btn = document.getElementById("toggleAnalysisRecommendations");
    if (btn) {
        btn.textContent = analysisRecommendationEnabled ? "关闭" : "开启";
    }
    if (lastBoardData) {
        updateChessboard(lastBoardData.board, lastBoardData.move);
    }
}

/**
 * 应用分析更新。
 * 
 * @param {object} data - 分析数据
 */
function applyAnalysisUpdate(data) {
    if (!Number.isInteger(data.step) || data.step !== currentStep) {
        return;
    }
    if (data.status === "ready") {
        latestAnalysisData = data;
    } else if (data.status === "pending") {
        latestAnalysisData = null;
    } else {
        latestAnalysisData = null;
    }
    updateAnalysisWinRateDisplay(data);
    if (lastBoardData) {
        updateChessboard(lastBoardData.board, lastBoardData.move);
    }
}

/**
 * 应用分析配置。
 * 
 * @param {object} data - 分析配置数据
 */
function applyAnalysisConfig(data) {
    if (!data || !Number.isInteger(data.iter)) {
        return;
    }
    setAnalysisIterInput(data.iter);
}

/**
 * 设置分析迭代次数输入框的值。
 * 
 * @param {number} iter - 目标迭代次数
 */
function setAnalysisIterInput(iter) {
    const iterSelect = document.getElementById("analysisIter");
    if (!iterSelect) return;
    const normalized = Number.isInteger(iter) && iter > 0 ? iter : 1500;
    const value = String(normalized);
    let optionExists = false;
    for (const option of iterSelect.options) {
        if (option.value === value) {
            optionExists = true;
            break;
        }
    }
    if (!optionExists) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `自定义 (${value})`;
        iterSelect.appendChild(option);
    }
    iterSelect.value = value;
    syncCustomSelect(iterSelect);
}

/**
 * 应用分析迭代次数。
 * 
 * @param {number} iter - 目标迭代次数
 */
function applyAnalysisIter(iter) {
    if (uiLocked) return;
    const normalized = Number.isInteger(iter) && iter > 0 ? iter : 1500;
    socket.emit("set_analysis_iter", { iter: normalized });
}

/**
 * 获取 Celestial 文本的颜色。
 * 
 * @returns {string} Celestial 文本的颜色，或默认值 "#334155"
 */
function getCelestialColor() {
    const color = window.getComputedStyle(celestialText).color;
    return color && color !== "rgba(0, 0, 0, 0)" ? color : "#334155";
}

/**
 * 更新分析胜率显示。
 * 
 * @param {object} data - 分析数据
 */
function updateAnalysisWinRateDisplay(data) {
    const valueEl = document.getElementById("analysisWinRateValue");
    if (!valueEl) return;
    if (data.status === "ready" && Number.isFinite(data.current_win_rate)) {
        valueEl.textContent = `${(data.current_win_rate * 100).toFixed(2)}%`;
        return;
    }
    if (data.status === "pending") {
        valueEl.textContent = "分析中";
        return;
    }
    valueEl.textContent = "--";
}

/**
 * 构建分析移动映射。
 * 
 * @returns {object} 分析移动映射，键为 "row-col" 格式，值为移动数据
 */
function buildAnalysisMap() {
    if (!latestAnalysisData || latestAnalysisData.status !== "ready" || !Array.isArray(latestAnalysisData.moves)) {
        return {};
    }
    const map = {};
    for (const move of latestAnalysisData.moves) {
        if (!Number.isInteger(move.row) || !Number.isInteger(move.col)) continue;
        map[`${move.row}-${move.col}`] = move;
    }
    return map;
}

/**
 * 处理 CMD 输入框的按键事件。
 * 
 * @param {KeyboardEvent} e - 按键事件对象
 */
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

/**
 * 启用 CMD 面板的拖拽功能（IIFE）。
 */
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

/**
 * 为 CMD 标题添加双击事件处理函数，用于切换最大化状态。
 */
cmdHeader.addEventListener("dblclick", function() {
    toggleCmdMaximize();
});

/**
 * 为 CMD 关闭按钮添加点击事件处理函数。
 */
cmdCloseBtn.addEventListener("click", function() {
    setCmdPanelVisible(false);
});

/**
 * 为 CMD 最小化按钮添加点击事件处理函数。
 */
cmdMinBtn.addEventListener("click", function() {
    setCmdCollapsed(!cmdPanel.classList.contains("minimized"));
});

/**
 * 为 CMD 最大化按钮添加点击事件处理函数。
 */
cmdMaxBtn.addEventListener("click", function() {
    toggleCmdMaximize();
});
