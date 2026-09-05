/**
 * Autonomous Exam Proctoring & Grading Agent — Shared App JavaScript
 */

const API_BASE = window.location.origin;

const App = {
    currentUser: null,
    authToken: localStorage.getItem('token'),
    currentPage: 'dashboard',
    pollingInterval: null,

    init() {
        this.authToken = localStorage.getItem('token');
        this.currentUser = this.getUserFromStorage();

        // Check if on a protected page without token
        const publicPages = ['/', '/login', '/register'];
        const path = window.location.pathname;
        if (!this.authToken && !publicPages.some(p => path.startsWith(p))) {
            window.location.href = '/login';
            return;
        }

        if (this.authToken) {
            this.startPolling();
        }

        // Listen for tab visibility for session safety
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.currentPage === 'exam') {
                this.logBrowserEvent('tab_switch');
            }
        });

        // Fullscreen change
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement && this.currentPage === 'exam') {
                this.logBrowserEvent('fullscreen_exit');
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.currentPage === 'exam') {
                this.logBrowserEvent('fullscreen_exit');
            }
        });
    },

    getUserFromStorage() {
        try {
            return JSON.parse(localStorage.getItem('user') || '{}');
        } catch { return {}; }
    },

    isAuthenticated() { return !!this.authToken; },

    getHeaders() {
        return {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.authToken}`,
        };
    },

    async api(url, options = {}) {
        const res = await fetch(`${API_BASE}${url}`, {
            ...options,
            headers: { ...this.getHeaders(), ...options.headers },
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Request failed' }));
            throw new Error(err.error || err.detail || `HTTP ${res.status}`);
        }
        return res.json();
    },

    async apiPost(url, data) {
        return this.api(url, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },

    async apiPut(url, data) {
        return this.api(url, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },

    async apiGet(url) {
        return this.api(url, { method: 'GET' });
    },

    async apiDelete(url) {
        return this.api(url, { method: 'DELETE' });
    },

    async login(username, password) {
        const res = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Login failed' }));
            throw new Error(err.error);
        }
        const data = await res.json();
        this.setAuth(data.token, data.user);
        return data;
    },

    async register(userData) {
        const res = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Registration failed' }));
            throw new Error(err.error);
        }
        const data = await res.json();
        this.setAuth(data.token, data.user);
        return data;
    },

    setAuth(token, user) {
        this.authToken = token;
        this.currentUser = user;
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));
    },

    logout() {
        this.authToken = null;
        this.currentUser = null;
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        this.stopPolling();
        window.location.href = '/login';
    },

    getUserRole() {
        return this.currentUser?.role || 'student';
    },

    // ─── Navigation ──────────────────────────────────────────────────────────

    navigate(page) {
        this.currentPage = page;
        document.querySelectorAll('.nav-item').forEach(el => {
            el.classList.toggle('active', el.dataset.page === page);
        });
    },

    // ─── Polling ─────────────────────────────────────────────────────────────

    startPolling() {
        this.stopPolling();
        this.pollingInterval = setInterval(() => this.pollUpdates(), 5000);
    },

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    },

    async pollUpdates() {
        // Silent polling for dashboard data
    },

    // ─── Browser Events ──────────────────────────────────────────────────────

    browserEvents: [],
    sessionStartTime: Date.now(),

    logBrowserEvent(type) {
        this.browserEvents.push({ type, timestamp: Date.now() });
    },

    getBrowserStats() {
        const elapsed = (Date.now() - this.sessionStartTime) / 1000;
        return {
            tab_switches: this.browserEvents.filter(e => e.type === 'tab_switch').length,
            window_blurs: this.browserEvents.filter(e => e.type === 'window_blur').length,
            fullscreen_exits: this.browserEvents.filter(e => e.type === 'fullscreen_exit').length,
            visibility_changes: this.browserEvents.filter(e => e.type === 'visibility_change').length,
            session_elapsed: elapsed,
        };
    },

    // ─── Exam Preflight ──────────────────────────────────────────────────────

    async runPreflight(checks) {
        try {
            const result = await this.apiPost('/api/preflight', checks);
            return result;
        } catch (e) {
            return { checks, can_proceed: false, error: e.message };
        }
    },

    // ─── Exam Session ────────────────────────────────────────────────────────

    startExamSession(studentId, examId) {
        this.examSession = {
            studentId,
            examId,
            startTime: Date.now(),
            answers: {},
            browserEvents: [],
        };
    },

    submitAnswer(questionId, answer) {
        if (this.examSession) {
            this.examSession.answers[questionId] = answer;
        }
    },

    getExamElapsed() {
        if (!this.examSession) return 0;
        return Math.floor((Date.now() - this.examSession.startTime) / 1000);
    },
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => App.init());
