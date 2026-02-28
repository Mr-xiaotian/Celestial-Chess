/**
 * 初始化配置面板内的原生 select，替换为自定义下拉组件。
 */
function initCustomSelects() {
    const selects = document.querySelectorAll(".config-panel select");
    if (!selects.length) return;

    selects.forEach(select => {
        if (select.dataset.customized === "true") return;
        select.dataset.customized = "true";

        const wrapper = document.createElement("div");
        wrapper.className = "custom-select-wrapper";
        const trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "custom-select-trigger";
        const options = document.createElement("div");
        options.className = "custom-select-options";

        wrapper.appendChild(trigger);
        wrapper.appendChild(options);

        const parent = select.parentNode;
        parent.insertBefore(wrapper, select);
        wrapper.appendChild(select);
        select.classList.add("custom-select-native");

        rebuildCustomOptions(select, wrapper);
        syncCustomSelect(select);

        trigger.addEventListener("click", (e) => {
            e.stopPropagation();
            closeAllCustomSelects(wrapper);
            const shouldOpen = !wrapper.classList.contains("open");
            wrapper.classList.toggle("open", shouldOpen);
            if (shouldOpen) {
                setCustomSelectDirection(wrapper);
            } else {
                wrapper.classList.remove("open-up");
                clearCustomSelectPosition(wrapper);
            }
        });

        options.addEventListener("click", (e) => {
            const item = e.target.closest(".custom-select-option");
            if (!item) return;
            select.value = item.dataset.value;
            select.dispatchEvent(new Event("change", { bubbles: true }));
            syncCustomSelect(select);
            wrapper.classList.remove("open");
            wrapper.classList.remove("open-up");
            clearCustomSelectPosition(wrapper);
        });

        select.addEventListener("change", () => {
            syncCustomSelect(select);
        });
    });

    document.addEventListener("click", () => closeAllCustomSelects());
    window.addEventListener("resize", () => closeAllCustomSelects());
}

/**
 * 关闭所有已打开的自定义下拉。
 * 
 * @param {HTMLElement} [except] - 需要保留打开状态的容器
 */
function closeAllCustomSelects(except) {
    document.querySelectorAll(".custom-select-wrapper.open").forEach(wrapper => {
        if (except && wrapper === except) return;
        wrapper.classList.remove("open");
        wrapper.classList.remove("open-up");
        clearCustomSelectPosition(wrapper);
    });
}

/**
 * 根据视口空间决定下拉向上或向下展开，并设置浮层位置。
 * 
 * @param {HTMLElement} wrapper - 自定义下拉容器
 */
function setCustomSelectDirection(wrapper) {
    const trigger = wrapper.querySelector(".custom-select-trigger");
    const options = wrapper.querySelector(".custom-select-options");
    if (!trigger || !options) return;

    const rect = trigger.getBoundingClientRect();
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    const spaceBelow = viewportHeight - rect.bottom;
    const spaceAbove = rect.top;
    const optionsHeight = Math.min(options.scrollHeight, 240);
    const gap = 6;

    const openUp = spaceBelow < optionsHeight && spaceAbove > spaceBelow;
    wrapper.classList.toggle("open-up", openUp);
    options.style.minWidth = `${rect.width}px`;
    options.style.left = `${rect.left}px`;
    if (openUp) {
        const top = Math.max(gap, rect.top - optionsHeight - gap);
        options.style.top = `${top}px`;
    } else {
        const top = Math.min(viewportHeight - gap - optionsHeight, rect.bottom + gap);
        options.style.top = `${top}px`;
    }
}

/**
 * 清理自定义下拉浮层定位样式。
 * 
 * @param {HTMLElement} wrapper - 自定义下拉容器
 */
function clearCustomSelectPosition(wrapper) {
    const options = wrapper.querySelector(".custom-select-options");
    if (!options) return;
    options.style.minWidth = "";
    options.style.left = "";
    options.style.top = "";
}

/**
 * 根据原生 select 构建自定义选项列表。
 * 
 * @param {HTMLSelectElement} select - 原生 select
 * @param {HTMLElement} wrapper - 自定义下拉容器
 */
function rebuildCustomOptions(select, wrapper) {
    const options = wrapper.querySelector(".custom-select-options");
    options.innerHTML = "";
    Array.from(select.options).forEach(option => {
        const item = document.createElement("div");
        item.className = "custom-select-option";
        item.dataset.value = option.value;
        item.textContent = option.textContent;
        if (option.selected) {
            item.classList.add("selected");
        }
        options.appendChild(item);
    });
}

/**
 * 同步自定义下拉显示与原生 select 状态。
 * 
 * @param {HTMLSelectElement} select - 原生 select
 */
function syncCustomSelect(select) {
    const wrapper = select.closest(".custom-select-wrapper");
    if (!wrapper) return;
    const trigger = wrapper.querySelector(".custom-select-trigger");
    const options = wrapper.querySelectorAll(".custom-select-option");
    const selectedOption = select.options[select.selectedIndex];
    trigger.textContent = selectedOption ? selectedOption.textContent : "";
    options.forEach(option => {
        option.classList.toggle("selected", option.dataset.value === select.value);
    });
}

window.initCustomSelects = initCustomSelects;
window.syncCustomSelect = syncCustomSelect;
