// ==========================================================
// 全局状态与 Socket
// ==========================================================
let currentColor; 
let power;

let socket = io.connect(
    window.location.protocol + '//' + window.location.host,
    { secure: true }
);


// ==========================================================
// DOMContentLoaded: 初始化入口
// ==========================================================
document.addEventListener('DOMContentLoaded', function () {

    bindUIButtons();
    initStateFromBackend();
    bindSocketEvents();

});


// ==========================================================
// 初始化 & UI 按钮绑定
// ==========================================================
function bindUIButtons() {
    const clickBind = (id, eventName) =>
        document.getElementById(id).addEventListener("click", () => socket.emit(eventName));

    clickBind("undoButton", 'undo_move');
    clickBind("redoButton", 'redo_move');
    clickBind("restartButton", 'restart_game');

    clickBind("minimaxButton", 'minimax_move');
    clickBind("mctsButton", 'mcts_move');
    clickBind("monkyButton", 'monky_move');

    clickBind("minimaxAuto", 'minimax_auto');
    clickBind("mctsAuto", 'mcts_auto');
    clickBind("monkyAuto", 'monky_auto');

    // CMD 开关
    document.getElementById("toggleCmd").addEventListener("click", toggleCmdPanel);
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

        // CMD 输出
        if (cmdPanel.style.display !== "none") {
            cmdPrint(`Move = ${JSON.stringify(data.move)}, Score = ${data.score}`);
            if (data.game_over) {
                cmdPrint("Game Over.");
                cmdPrint("Winner = " + data.winner);
            }
        }

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
        if (cmdPanel.style.display !== "none") {
            cmdPrint(data.msg);
        }
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
                                 : (value === "-inf") ? '-∞'
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
    if (value === "inf" || value === "-inf") {
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

cmdInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") {
        let text = cmdInput.value.trim();
        if (text.length > 0) {
            cmdPrint("I:" + text);
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
