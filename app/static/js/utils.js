/**
 * ExamAI — Utility Functions
 * Common helpers for UI interactions.
 */

// ─────────────────────────────────────────────────────────────────
// DOM Helpers
// ─────────────────────────────────────────────────────────────────

function $(selector, parent = document) {
    return parent.querySelector(selector);
}

function $$(selector, parent = document) {
    return Array.from(parent.querySelectorAll(selector));
}

// ─────────────────────────────────────────────────────────────────
// Format Helpers
// ─────────────────────────────────────────────────────────────────

function formatTime(seconds) {
    if (seconds < 0) seconds = 0;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function formatDate(isoString) {
    if (!isoString) return '-';
    const d = new Date(isoString);
    return d.toLocaleString();
}

function formatPercent(value, decimals = 1) {
    if (value === null || value === undefined) return '-';
    return (value * 100).toFixed(decimals) + '%';
}

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined) return '-';
    return Number(value).toFixed(decimals);
}

// ─────────────────────────────────────────────────────────────────
// Risk Level Helpers
// ─────────────────────────────────────────────────────────────────

function getRiskLevel(score) {
    if (score >= 0.7) return { level: 'HIGH', class: 'danger', label: 'High Risk' };
    if (score >= 0.5) return { level: 'MEDIUM', class: 'warning', label: 'Medium Risk' };
    if (score >= 0.3) return { level: 'LOW', class: 'info', label: 'Low Risk' };
    return { level: 'NORMAL', class: 'success', label: 'Normal' };
}

function getStateLevel(state) {
    const states = {
        'NORMAL': { class: 'success', label: 'Normal' },
        'SUSPICIOUS': { class: 'warning', label: 'Suspicious' },
        'HIGH_RISK': { class: 'danger', label: 'High Risk' },
        'REVIEW_REQUIRED': { class: 'danger', label: 'Review Required' },
    };
    return states[state] || states['NORMAL'];
}

function getProgressLevel(percent) {
    if (percent >= 100) return { class: 'success', label: 'Completed' };
    if (percent >= 50) return { class: 'info', label: 'In Progress' };
    return { class: 'neutral', label: 'Not Started' };
}

// ─────────────────────────────────────────────────────────────────
// UI Component Builders
// ─────────────────────────────────────────────────────────────────

function createBadge(text, type = 'neutral') {
    const badge = document.createElement('span');
    badge.className = `badge badge-${type}`;
    badge.textContent = text;
    return badge;
}

function createStatCard(value, label, subtitle = '') {
    const card = document.createElement('div');
    card.className = 'stat-card';
    card.innerHTML = `
        <div class="stat-value">${value}</div>
        <div class="stat-label">${label}</div>
        ${subtitle ? `<div class="text-muted" style="font-size: 0.75rem; margin-top: 4px;">${subtitle}</div>` : ''}
    `;
    return card;
}

function createProgressBar(value, max = 100, className = '') {
    const percent = Math.min(100, Math.max(0, (value / max) * 100));
    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    bar.innerHTML = `
        <div class="progress-fill ${className}" style="width: ${percent}%"></div>
    `;
    return bar;
}

function createRiskMeter(score, showValue = true) {
    const level = getRiskLevel(score);
    const percent = Math.round(score * 100);

    const container = document.createElement('div');
    container.className = 'risk-meter';
    container.innerHTML = `
        ${showValue ? `<div class="risk-value" style="color: var(--accent-${level.class})">${percent}</div>` : ''}
        <div class="risk-meter-bar">
            <div class="risk-meter-fill" style="width: ${percent}%; background: var(--accent-${level.class})"></div>
        </div>
        <div class="risk-meter-labels">
            <span>0</span>
            <span>50</span>
            <span>100</span>
        </div>
    `;
    return container;
}

// ─────────────────────────────────────────────────────────────────
// Toast Notifications
// ─────────────────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
    let container = $('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toast-in 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─────────────────────────────────────────────────────────────────
// Loading States
// ─────────────────────────────────────────────────────────────────

function showLoading(element) {
    element.innerHTML = `
        <div class="flex items-center justify-center" style="padding: 40px;">
            <div class="spinner"></div>
        </div>
    `;
}

function showEmpty(element, message = 'No data available') {
    element.innerHTML = `
        <div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"/>
            </svg>
            <h3>${message}</h3>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────
// Modal
// ─────────────────────────────────────────────────────────────────

function showModal(title, content, buttons = []) {
    let overlay = $('.modal-overlay');
    if (overlay) overlay.remove();

    overlay = document.createElement('div');
    overlay.className = 'modal-overlay';

    let buttonsHtml = '';
    buttons.forEach((btn, i) => {
        buttonsHtml += `<button class="btn btn-${btn.type || 'secondary'}" data-action="${i}">${btn.label}</button>`;
    });

    overlay.innerHTML = `
        <div class="modal">
            <div class="modal-header">
                <h3 class="modal-title">${title}</h3>
                <button class="modal-close" data-close>&times;</button>
            </div>
            <div class="modal-content">${content}</div>
            ${buttons.length ? `<div class="flex gap-md mt-lg">${buttonsHtml}</div>` : ''}
        </div>
    `;

    document.body.appendChild(overlay);

    // Handle close
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay || e.target.dataset.close) {
            overlay.classList.remove('active');
            setTimeout(() => overlay.remove(), 300);
        }
    });

    // Handle button clicks
    overlay.addEventListener('click', (e) => {
        if (e.target.dataset.action !== undefined) {
            const idx = parseInt(e.target.dataset.action);
            if (buttons[idx].onClick) buttons[idx].onClick();
        }
    });

    // Animate in
    requestAnimationFrame(() => overlay.classList.add('active'));

    return overlay;
}

// ─────────────────────────────────────────────────────────────────
// Confirm Dialog
// ─────────────────────────────────────────────────────────────────

async function confirm(message, title = 'Confirm') {
    return new Promise((resolve) => {
        showModal(title, `<p>${message}</p>`, [
            { label: 'Cancel', type: 'secondary', onClick: () => resolve(false) },
            { label: 'Confirm', type: 'primary', onClick: () => resolve(true) },
        ]);
    });
}

// ─────────────────────────────────────────────────────────────────
// Signal Activity Bar
// ─────────────────────────────────────────────────────────────────

function createSignalBar(label, percent, color = 'primary') {
    const bar = document.createElement('div');
    bar.className = 'signal-bar';
    bar.innerHTML = `
        <span class="signal-label">${label}</span>
        <div class="signal-track">
            <div class="signal-fill" style="width: ${Math.min(100, percent * 100)}%; background: var(--accent-${color})"></div>
        </div>
        <span class="signal-value">${Math.round(percent * 100)}%</span>
    `;
    return bar;
}

// ─────────────────────────────────────────────────────────────────
// Timeline Item
// ─────────────────────────────────────────────────────────────────

function createTimelineItem(time, title, description, severity = 'info') {
    const item = document.createElement('div');
    item.className = 'timeline-item';
    item.innerHTML = `
        <div class="timeline-time">${time}</div>
        <div class="timeline-title">${title}</div>
        ${description ? `<div class="timeline-description">${description}</div>` : ''}
    `;
    return item;
}

// ─────────────────────────────────────────────────────────────────
// Safe HTML (XSS Prevention)
// ─────────────────────────────────────────────────────────────────

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// ─────────────────────────────────────────────────────────────────
// Debounce / Throttle
// ─────────────────────────────────────────────────────────────────

function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

function throttle(fn, limit) {
    let inThrottle;
    return (...args) => {
        if (!inThrottle) {
            fn(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ─────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────

window.ExamAI = {
    $,
    $$,
    formatTime,
    formatDate,
    formatPercent,
    formatNumber,
    getRiskLevel,
    getStateLevel,
    getProgressLevel,
    createBadge,
    createStatCard,
    createProgressBar,
    createRiskMeter,
    showToast,
    showLoading,
    showEmpty,
    showModal,
    confirm,
    createSignalBar,
    createTimelineItem,
    escapeHtml,
    debounce,
    throttle,
};
