/**
 * Autonomous Exam Proctoring & Grading Agent - Shared App JavaScript
 */

const API_BASE = window.location.origin;

const App = {
    currentUser: null,
    authToken: localStorage.getItem('token'),
    currentPage: 'dashboard',
    pollingInterval: null,

    _keystroke: {
        keyDownTimes: {},
        holdDurations: [],
        latencies: [],
        lastKeyDownTime: null,
        eventCount: 0,
        sessionStart: null,
        idleSeconds: 0,
        _idleTimer: null,
    },

    initKeystrokeTracking() {
        const k = this._keystroke;
        k.sessionStart = Date.now();
        document.addEventListener('keydown', (e) => {
            const now = Date.now();
            if (k.lastKeyDownTime !== null) {
                const lat = now - k.lastKeyDownTime;
                if (lat < 5000) k.latencies.push(lat);
            }
            k.keyDownTimes[e.code] = now;
            k.lastKeyDownTime = now;
            k.eventCount++;
            clearTimeout(k._idleTimer);
            k._idleTimer = setTimeout(() => { k.idleSeconds += 5; }, 5000);
        });
        document.addEventListener('keyup', (e) => {
            if (k.keyDownTimes[e.code] !== undefined) {
                const hold = Date.now() - k.keyDownTimes[e.code];
                if (hold >= 0 && hold < 2000) k.holdDurations.push(hold);
                delete k.keyDownTimes[e.code];
            }
        });
    },

    getKeyboardStats() {
        const k = this._keystroke;
        const elapsedMinutes = Math.max((Date.now() - (k.sessionStart || Date.now())) / 60000, 0.001);
        const holds = k.holdDurations;
        const lats  = k.latencies;
        const mean = (arr) => arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
        const std  = (arr, m) => arr.length > 1 ? Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length) : 0;
        const meanHold = mean(holds);
        const meanLat  = mean(lats);
        const stdHold  = std(holds, meanHold);
        const stdLat   = std(lats, meanLat);
        const deviation = Math.min((stdHold / Math.max(meanHold, 1) + stdLat / Math.max(meanLat, 1)) / 2, 1.0);
        return {
            events_per_minute: Math.round(k.eventCount / elapsedMinutes),
            mean_key_hold:     Math.round(meanHold),
            mean_latency:      Math.round(meanLat),
            std_key_hold:      Math.round(stdHold),
            std_latency:       Math.round(stdLat),
            deviation:         Math.round(deviation * 10000) / 10000,
            idle_seconds:      k.idleSeconds,
            total_events:      k.eventCount,
        };
    },

    _mouse: {
        clicks: 0,
        rightClicks: 0,
        totalDistance: 0,
        speeds: [],
        lastPos: null,
        lastMoveTime: null,
        idleSeconds: 0,
        _idleTimer: null,
        sessionStart: null,
    },

    initMouseTracking() {
        const m = this._mouse;
        m.sessionStart = Date.now();
        document.addEventListener('mousemove', (e) => {
            const now = Date.now();
            const x = e.clientX, y = e.clientY;
            if (m.lastPos && m.lastMoveTime) {
                const dt = (now - m.lastMoveTime) / 1000;
                if (dt > 0 && dt < 2) {
                    const dx = x - m.lastPos.x, dy = y - m.lastPos.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    m.totalDistance += dist;
                    const speed = dist / dt;
                    if (speed < 5000) m.speeds.push(speed);
                }
            }
            m.lastPos = { x, y };
            m.lastMoveTime = now;
            clearTimeout(m._idleTimer);
            m._idleTimer = setTimeout(() => { m.idleSeconds += 5; }, 5000);
        });
        document.addEventListener('click', () => { m.clicks++; });
        document.addEventListener('contextmenu', () => { m.rightClicks++; });
    },

    getMouseStats() {
        const m = this._mouse;
        const elapsedSeconds = Math.max((Date.now() - (m.sessionStart || Date.now())) / 1000, 0.001);
        const speeds = m.speeds;
        const mean = (arr) => arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
        const std  = (arr, mu) => arr.length > 1 ? Math.sqrt(arr.reduce((s, v) => s + (v - mu) ** 2, 0) / arr.length) : 0;
        const avgSpeed = mean(speeds);
        const stdSpeed = std(speeds, avgSpeed);
        const deviation = Math.min(stdSpeed / Math.max(avgSpeed, 1), 1.0);
        return {
            avg_speed:      Math.round(avgSpeed),
            std_speed:      Math.round(stdSpeed),
            total_distance: Math.round(m.totalDistance),
            click_rate:     Math.round((m.clicks / elapsedSeconds) * 100) / 100,
            right_clicks:   m.rightClicks,
            deviation:      Math.round(deviation * 10000) / 10000,
            idle_seconds:   m.idleSeconds,
        };
    },

    init() {
        this.authToken = localStorage.getItem('token');
        this.currentUser = this.getUserFromStorage();
        const publicPages = ['/', '/login', '/register'];
        const path = window.location.pathname;
        if (!this.authToken && !publicPages.some(p => path.startsWith(p))) {
            window.location.href = '/login';
            return;
        }
        if (this.authToken) { this.startPolling(); }
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.currentPage === 'exam') { this.logBrowserEvent('tab_switch'); }
        });
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement && this.currentPage === 'exam') { this.logBrowserEvent('fullscreen_exit'); }
        });
        window.addEventListener('blur', () => {
            if (this.currentPage === 'exam') { this.logBrowserEvent('window_blur'); }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.currentPage === 'exam') { this.logBrowserEvent('fullscreen_exit'); }
        });
    },

    getUserFromStorage() {
        try { return JSON.parse(localStorage.getItem('user') || '{}'); } catch { return {}; }
    },

    isAuthenticated() { return !!this.authToken; },

    getHeaders() {
        return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.authToken}` };
    },

    async api(url, options = {}) {
        const res = await fetch(`${API_BASE}${url}`, { ...options, headers: { ...this.getHeaders(), ...options.headers } });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Request failed' }));
            throw new Error(err.error || err.detail || `HTTP ${res.status}`);
        }
        return res.json();
    },

    async apiPost(url, data) { return this.api(url, { method: 'POST', body: JSON.stringify(data) }); },
    async apiPut(url, data)  { return this.api(url, { method: 'PUT',  body: JSON.stringify(data) }); },
    async apiGet(url)         { return this.api(url, { method: 'GET'  }); },
    async apiDelete(url)      { return this.api(url, { method: 'DELETE' }); },

    async login(username, password) {
        const res = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) { const err = await res.json().catch(() => ({ error: 'Login failed' })); throw new Error(err.error); }
        const data = await res.json();
        this.setAuth(data.token, data.user);
        return data;
    },

    async register(userData) {
        const res = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData),
        });
        if (!res.ok) { const err = await res.json().catch(() => ({ error: 'Registration failed' })); throw new Error(err.error); }
        const data = await res.json();
        this.setAuth(data.token, data.user);
        return data;
    },

    setAuth(token, user) {
        this.authToken = token; this.currentUser = user;
        localStorage.setItem('token', token); localStorage.setItem('user', JSON.stringify(user));
    },

    logout() {
        this.authToken = null; this.currentUser = null;
        localStorage.removeItem('token'); localStorage.removeItem('user');
        this.stopPolling();
        window.location.href = '/login';
    },

    getUserRole() { return this.currentUser?.role || 'student'; },

    navigate(page) {
        this.currentPage = page;
        document.querySelectorAll('.nav-item').forEach(el => { el.classList.toggle('active', el.dataset.page === page); });
    },

    startPolling() { this.stopPolling(); this.pollingInterval = setInterval(() => this.pollUpdates(), 5000); },
    stopPolling()  { if (this.pollingInterval) { clearInterval(this.pollingInterval); this.pollingInterval = null; } },
    async pollUpdates() { /* Silent polling for dashboard data */ },

    browserEvents: [],
    sessionStartTime: Date.now(),

    logBrowserEvent(type) { this.browserEvents.push({ type, timestamp: Date.now() }); },

    getBrowserStats() {
        const t0 = (this.examSession && this.examSession.startTime) ? this.examSession.startTime : this.sessionStartTime;
        const elapsed = (Date.now() - t0) / 1000;
        return {
            tab_switches:      this.browserEvents.filter(e => e.type === 'tab_switch').length,
            window_blurs:      this.browserEvents.filter(e => e.type === 'window_blur').length,
            fullscreen_exits:  this.browserEvents.filter(e => e.type === 'fullscreen_exit').length,
            visibility_changes:this.browserEvents.filter(e => e.type === 'visibility_change').length,
            session_elapsed:   elapsed,
        };
    },

    async runPreflight(checks) {
        try { return await this.apiPost('/api/preflight', checks); }
        catch (e) { return { checks, can_proceed: false, error: e.message }; }
    },

    startExamSession(studentId, examId) {
        this.examSession = { studentId, examId, startTime: Date.now(), answers: {}, browserEvents: [] };
    },

    submitAnswer(questionId, answer) { if (this.examSession) { this.examSession.answers[questionId] = answer; } },

    getExamElapsed() {
        if (!this.examSession) return 0;
        return Math.floor((Date.now() - this.examSession.startTime) / 1000);
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
