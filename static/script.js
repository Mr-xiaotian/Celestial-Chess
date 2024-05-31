let currentColor = 1; // 初始颜色设为 1
let board_size = 5; //设定棋盘大小
let power = 2; // 设定power

document.addEventListener('DOMContentLoaded', function () {
    renderChessboard(board_size, board_size); // 假设棋盘是 8x8 的
    document.getElementById("undoButton").addEventListener("click", undoMove);
    document.getElementById("redoButton").addEventListener("click", redoMove);
    document.getElementById("restartButton").addEventListener("click", restartGame);
    document.getElementById("aiButton").addEventListener("click", aiDo);

    fetch('/current_state')
        .then(response => response.json())
        .then(data => {
            updateChessboard(data.board); // 更新棋盘
            updateTotalScore(data.score); // 更新分数
            toggleColor(data.step); // 根据步数切换颜色
        })
        .catch(error => console.error('Error:', error));
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

function onCellClick(row, col) {
    fetch('/play', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ row: row, col: col, color: currentColor })
    })
        .then(response => response.json())
        .then(data => {
            updateChessboard(data.board);
            updateTotalScore(data.score); // 更新总分数
            toggleColor(data.step); // 切换颜色
        })
        .catch(error => console.error('Error:', error));
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

function undoMove() {
    fetch('/undo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
        // 不需要发送任何数据，因为悔棋操作通常不需要额外的信息
    })
        .then(response => response.json())
        .then(data => {
            updateChessboard(data.board); // 更新棋盘
            updateTotalScore(data.score); // 更新分数
            toggleColor(data.step); // 切换颜色
        })
        .catch(error => console.error('Error:', error));
}

function redoMove() {
    fetch('/redo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
        // 不需要发送任何数据，因为悔棋操作通常不需要额外的信息
    })
        .then(response => response.json())
        .then(data => {
            updateChessboard(data.board); // 更新棋盘
            updateTotalScore(data.score); // 更新分数
            toggleColor(data.step); // 切换颜色
        })
        .catch(error => console.error('Error:', error));
}

function restartGame() {
    // 发送 AJAX 请求到后端重启游戏
    // 根据后端响应重置界面
    fetch('/restart', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
        // 不需要发送任何数据，因为悔棋操作通常不需要额外的信息
    })
        .then(response => response.json())
        .then(data => {
            updateChessboard(data.board); // 更新棋盘
            updateTotalScore(data.score); // 更新分数
            toggleColor(data.step); // 切换颜色
        })
        .catch(error => console.error('Error:', error));
}

function aiDo() {
    // 发送 AJAX 请求到后端，让 AI 执棋
    fetch('/ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
        // 不需要发送任何数据，因为操作通常不需要额外的信息
    })
        .then(response => response.json())
        .then(data => {
            console.log('AI move received:', data); // 打印响应数据，用于调试
            updateChessboard(data.board); // 更新棋盘
            updateTotalScore(data.score); // 更新分数
            toggleColor(data.step); // 切换颜色
        })
        .catch(error => console.error('Error:', error));
}
