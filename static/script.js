let currentColor; // 设定颜色
let power; // 设定power
let socket = io.connect(window.location.protocol + '//' + window.location.host, { secure: true });

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById("undoButton").addEventListener("click", function () {
        socket.emit('undo_move')
    });
    document.getElementById("redoButton").addEventListener("click", function () {
        socket.emit('redo_move')
    });
    document.getElementById("restartButton").addEventListener("click", function () {
        socket.emit('restart_game')
    });
    document.getElementById("minimaxButton").addEventListener("click", function () {
        socket.emit('minimax_move')
    });
    document.getElementById("mctsButton").addEventListener("click", function () {
        socket.emit('mcts_move')
    });

    fetch('/init_state')
        .then(response => response.json())
        .then(data => {
            power = data.power; // 初始化power
            renderChessboard(data.weight, data.height); // 初始化棋盘size
            updateChessboard(data.board); // 初始化棋盘
            updateTotalScore(data.score); // 初始化分数
            toggleColor(data.step); // 根据步数切换颜色
        })
        .catch(error => console.error('Error:', error));

    // 添加 SocketIO 事件监听器
    socket.on('update_board', function (data) {
        if (data.error) {
            console.error('Error:', data.error);
        } else {
            updateChessboard(data.board);
            updateTotalScore(data.score);
            toggleColor(data.step);
        }
    });
});

function renderChessboard(weight, height) {
    const chessboard = document.getElementById('chessboard');
    chessboard.innerHTML = '';  // 清除现有棋盘

    for (let i = 0; i < weight; i++) {
        let row = chessboard.insertRow();
        for (let j = 0; j < height; j++) {
            let cell = row.insertCell();
            cell.addEventListener('click', function () {
                onCellClick(i, j);
            });
        }
    }
}

function updateChessboard(board) {
    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {
            let cell = document.getElementById('chessboard').rows[i].cells[j];
            let value = board[i][j][0]; // 数值
            let load = board[i][j][1]; // 负载

            // 先清除单元格内容
            cell.innerHTML = '';

            // 根据值设置背景色
            cell.style.backgroundColor = getColor(value);

            // 添加包含数字的 span 元素
            let numberSpan = document.createElement('span');
            numberSpan.style.position = 'absolute';
            numberSpan.style.top = '50%';
            numberSpan.style.left = '50%';
            numberSpan.style.transform = 'translate(-50%, -50%)';
            numberSpan.style.zIndex = '1'; // 确保数字位于最上层

            // 设置文本
            if (value === "inf") {
                numberSpan.textContent = '∞';
            } else if (value === "-inf") {
                numberSpan.textContent = '-∞';
            } else {
                numberSpan.textContent = load;
            }
            cell.appendChild(numberSpan);

            // 根据负载显示特殊标记
            if (load >= power && load < 2 * power) {
                // 显示菱形
                // drawDiamond(cell);
                
            } else if (load >= 2 * power) {
                // 显示圆形
                // drawCircle(cell);
            }
        }
    }
}

function getColor(value) {
    if (value === "inf" || value === "-inf") {
        return 'lightgrey'; // 黑洞点设为灰色
    }

    let alpha = Math.min(Math.abs(value) / 5, 1); // 假设5是最大值，超过5则饱和度为100%
    if (value > 0) {
        return `rgba(173, 216, 230, ${alpha})`; // 蓝色，透明度根据数值变化
    } else if (value < 0) {
        return `rgba(240, 128, 128, ${alpha})`; // 红色，透明度根据数值变化
    } else {
        return ''; // 数值为0时没有背景色
    }
}

function drawSquare(cell) {
    // clearShape(cell);

    let svgns = "http://www.w3.org/2000/svg";
    let svg = document.createElementNS(svgns, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");

    let diamond = document.createElementNS(svgns, "polygon");
    let cellWidth = cell.offsetWidth;
    let cellHeight = cell.offsetHeight;
    let points = [
        (cellWidth / 2) + ",0",        // 上中点
        "0," + (cellHeight / 2),       // 左中点
        (cellWidth / 2) + "," + cellHeight, // 下中点
        cellWidth + "," + (cellHeight / 2)  // 右中点
    ].join(" ");
    diamond.setAttribute("points", points);
    diamond.setAttribute("style", "fill:transparent;stroke:lightgrey;stroke-width:1");

    svg.appendChild(diamond);
    cell.appendChild(svg);
}

function drawCircle(cell) {
    // clearShape(cell);

    let circle = document.createElement('div');
    circle.style.width = '100%';
    circle.style.height = '100%';
    circle.style.borderRadius = '50%';
    circle.style.border = '2px solid lightgrey';
    circle.style.boxSizing = 'border-box'; // 包含边框在内的宽高

    cell.appendChild(circle);
}

function clearShape(cell) {
    cell.innerHTML = ''; // 移除单元格中的所有内容
}


function toggleColor(step) {
    currentColor = (step % 2 === 0) ? 1 : -1; // 切换颜色
    updatePlayerIndicator(); // 更新玩家指示器
}

function updatePlayerIndicator() {
    const celestialText = document.getElementById("celestial");

    if (currentColor === 1) {
        celestialText.style.color = "#6495ED"; // 设置为蓝色
    } else {
        celestialText.style.color = "lightcoral"; // 设置为珊瑚色
    }
}

function updateTotalScore(score) {
    document.getElementById("totalScore").textContent = score;
}

function onCellClick(row, col) {
    // fetch('/play', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({ row: row, col: col, color: currentColor })
    // })
    //     .then(response => response.json())
    //     .then(data => {
    //         updateChessboard(data.board);
    //         updateTotalScore(data.score); // 更新总分数
    //         toggleColor(data.step); // 切换颜色
    //     })
    //     .catch(error => console.error('Error:', error));
    socket.emit('play_move', { row: row, col: col, color: currentColor });
    // console.log('Clicked cell:', row, col);
}

