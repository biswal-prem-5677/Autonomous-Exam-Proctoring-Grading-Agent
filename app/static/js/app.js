/**
 * Autonomous Exam Proctoring & Grading Agent - Shared App JavaScript
 */

window.API_BASE = window.API_BASE || '/api';

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
        document.addEventListener('contextmenu', () => {
            m.rightClicks++;
            App.logBrowserEvent('contextmenu');
        });
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

    // ─── Audio Analysis via Web Audio ─────────────────────────────────────────

    _audio: {
        audioContext: null,
        analyser: null,
        stream: null,
        active: false,
    },

    async initAudioCapture(stream) {
        try {
            const audioStream = stream || await navigator.mediaDevices.getUserMedia({ audio: true });
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioContextClass();
            const source = ctx.createMediaStreamSource(audioStream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 512;
            source.connect(analyser);

            this._audio.audioContext = ctx;
            this._audio.analyser = analyser;
            this._audio.stream = audioStream;
            this._audio.active = true;
            return true;
        } catch (e) {
            console.warn('Audio capture not available:', e);
            this._audio.active = false;
            return false;
        }
    },

    getAudioStats() {
        if (!this._audio.active || !this._audio.analyser) {
            return {
                rms_energy: 0.0,
                volume_db: -100.0,
                zero_crossing_rate: 0.0,
                spectral_centroid: 0.0,
                speech_detected: false,
            };
        }
        const analyser = this._audio.analyser;
        const timeData = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(timeData);

        const freqData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(freqData);

        let sumSq = 0;
        let zcrCount = 0;
        for (let i = 0; i < timeData.length; i++) {
            const norm = (timeData[i] - 128) / 128;
            sumSq += norm * norm;
            if (i > 0) {
                const prevNorm = (timeData[i - 1] - 128) / 128;
                if ((norm >= 0 && prevNorm < 0) || (norm < 0 && prevNorm >= 0)) {
                    zcrCount++;
                }
            }
        }
        const rms = Math.sqrt(sumSq / timeData.length);
        const db = rms > 0.00001 ? 20 * Math.log10(rms) : -100.0;
        const zcr = zcrCount / timeData.length;

        let num = 0;
        let den = 0;
        for (let k = 0; k < freqData.length; k++) {
            num += k * freqData[k];
            den += freqData[k];
        }
        const sampleRate = analyser.context.sampleRate || 44100;
        const centroid = den > 0 ? (num / den) * (sampleRate / analyser.fftSize) : 0;

        return {
            rms_energy: Math.round(rms * 1000) / 1000,
            volume_db: Math.round(db * 10) / 10,
            zero_crossing_rate: Math.round(zcr * 1000) / 1000,
            spectral_centroid: Math.round(centroid * 10) / 10,
            speech_detected: db > -40.0 && rms > 0.04,
        };
    },

    // ─── Browser Events with Per-Interval Deltas ──────────────────────────────

    browserEvents: [],
    _deltas: {
        tab_switch: 0,
        fullscreen_exit: 0,
        paste: 0,
        copy: 0,
        cut: 0,
        contextmenu: 0,
        window_blur: 0,
    },
    sessionStartTime: Date.now(),

    logBrowserEvent(type) {
        this.browserEvents.push({ type, timestamp: Date.now() });
        if (this._deltas[type] !== undefined) {
            this._deltas[type]++;
        }
    },

    getBrowserStats(resetDelta = true) {
        const t0 = (this.examSession && this.examSession.startTime) ? this.examSession.startTime : this.sessionStartTime;
        const elapsed = (Date.now() - t0) / 1000;

        const stats = {
            tab_switches:       this.browserEvents.filter(e => e.type === 'tab_switch').length,
            window_blurs:       this.browserEvents.filter(e => e.type === 'window_blur').length,
            fullscreen_exits:   this.browserEvents.filter(e => e.type === 'fullscreen_exit').length,
            visibility_changes: this.browserEvents.filter(e => e.type === 'visibility_change').length,
            pastes:             this.browserEvents.filter(e => e.type === 'paste').length,
            copies:             this.browserEvents.filter(e => e.type === 'copy').length,
            cuts:               this.browserEvents.filter(e => e.type === 'cut').length,
            context_menus:      this.browserEvents.filter(e => e.type === 'contextmenu').length,

            // Per-interval deltas to prevent repetitive alert firing
            delta_tab_switches:     this._deltas.tab_switch,
            delta_fullscreen_exits: this._deltas.fullscreen_exit,
            delta_pastes:           this._deltas.paste,
            delta_copies:           this._deltas.copy,
            delta_cuts:             this._deltas.cut,
            delta_window_blurs:     this._deltas.window_blur,
            session_elapsed:        elapsed,
        };

        if (resetDelta) {
            this._deltas = {
                tab_switch: 0,
                fullscreen_exit: 0,
                paste: 0,
                copy: 0,
                cut: 0,
                contextmenu: 0,
                window_blur: 0,
            };
        }

        return stats;
    },

    // ─── Lifecycle & Auth ─────────────────────────────────────────────────────

    init() {
        this.authToken = localStorage.getItem('token');
        this.currentUser = this.getUserFromStorage();

        // Listeners for copy, cut, paste, tab switch, fullscreen
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                App.logBrowserEvent('tab_switch');
            }
        });
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement) {
                App.logBrowserEvent('fullscreen_exit');
            }
        });
        window.addEventListener('blur', () => {
            App.logBrowserEvent('window_blur');
        });
        document.addEventListener('copy', () => {
            App.logBrowserEvent('copy');
        });
        document.addEventListener('cut', () => {
            App.logBrowserEvent('cut');
        });
        document.addEventListener('paste', () => {
            App.logBrowserEvent('paste');
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                App.logBrowserEvent('fullscreen_exit');
            }
        });

        this.renderNavbar();
    },

    renderNavbar() {
        const navEl = document.getElementById('main-nav-links');
        const userEl = document.getElementById('nav-user-info');
        if (!navEl) return;

        const role = this.currentUser?.role || 'guest';
        let links = [];

        if (!this.authToken) {
            links = [
                { href: '/', label: 'Home' },
                { href: '/login', label: 'Login' },
                { href: '/register', label: 'Register' },
            ];
        } else if (role === 'student') {
            links = [
                { href: '/dashboard', label: 'Dashboard' },
                { href: '/student/exam', label: 'Take Exam' },
                { href: '/student/results', label: 'Results' },
            ];
        } else if (role === 'examiner') {
            links = [
                { href: '/examiner/dashboard', label: 'Dashboard' },
                { href: '/examiner/create-exam', label: 'Create Exam' },
                { href: '/examiner/grading', label: 'Grading' },
                { href: '/monitor', label: 'Live Monitor' },
                { href: '/analytics', label: 'Analytics' },
                { href: '/training', label: 'ML Models' },
            ];
        } else if (role === 'admin' || role === 'super_admin') {
            links = [
                { href: '/dashboard', label: 'Dashboard' },
                { href: '/examiner/dashboard', label: 'Examiner View' },
                { href: '/examiner/create-exam', label: 'Exam Builder' },
                { href: '/examiner/grading', label: 'Grading' },
                { href: '/monitor', label: 'Monitor' },
                { href: '/analytics', label: 'Analytics' },
                { href: '/training', label: 'ML Training' },
            ];
        }

        navEl.innerHTML = links.map(l =>
            `<a href="${l.href}" class="nav-link ${window.location.pathname === l.href ? 'active' : ''}">${l.label}</a>`
        ).join('');

        if (userEl) {
            if (this.authToken && this.currentUser) {
                userEl.innerHTML = `
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span class="user-badge role-${role}">${role.toUpperCase()}</span>
                        <span style="font-weight:600;font-size:13px;">${this.currentUser.full_name || this.currentUser.username || ''}</span>
                        <button onclick="App.logout()" class="btn btn-sm btn-secondary" style="margin-left:8px;">Logout</button>
                    </div>
                `;
            } else {
                userEl.innerHTML = `<a href="/login" class="btn btn-sm btn-primary">Sign In</a>`;
            }
        }
    },

    getUserFromStorage() {
        try { return JSON.parse(localStorage.getItem('user') || '{}'); } catch { return {}; }
    },

    isAuthenticated() { return !!this.authToken; },

    getHeaders() {
        const h = { 'Content-Type': 'application/json' };
        if (this.authToken) {
            h['Authorization'] = `Bearer ${this.authToken}`;
        }
        return h;
    },

    async api(url, options = {}) {
        const fullUrl = url.startsWith('http') ? url : (url.startsWith('/api') ? url : `/api${url}`);
        const res = await fetch(fullUrl, {
            ...options,
            headers: { ...this.getHeaders(), ...(options.headers || {}) },
        });
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
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Login failed' }));
            throw new Error(err.error || 'Invalid credentials');
        }
        const data = await res.json();
        this.setAuth(data.token, data.user);
        return data;
    },

    async register(userData) {
        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Registration failed' }));
            throw new Error(err.error || 'Registration failed');
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

    getUserRole() { return this.currentUser?.role || 'student'; },

    startPolling() { this.stopPolling(); this.pollingInterval = setInterval(() => this.pollUpdates(), 5000); },
    stopPolling()  { if (this.pollingInterval) { clearInterval(this.pollingInterval); this.pollingInterval = null; } },
    async pollUpdates() { /* Polling hook */ },

    async startExamSession(studentId, examId) {
        let serverSession = null;
        try {
            serverSession = await this.apiPost('/api/exam-sessions', {
                student_id: studentId,
                exam_id: examId,
            });
        } catch (e) {
            console.warn('Server session creation notice:', e.message);
        }

        const sessionKey = (serverSession && serverSession.session_key)
            ? serverSession.session_key
            : `${studentId}:${examId}`;

        this.examSession = {
            studentId,
            examId,
            sessionId: serverSession?.session_id || 'session_' + Math.random().toString(36).substring(2, 9),
            session_key: sessionKey,
            startTime: Date.now(),
            answers: {},
            browserEvents: [],
            serverData: serverSession,
        };
        return this.examSession;
    },

    submitAnswer(questionId, answer) {
        // Always cache locally first
        if (this.examSession) {
            this.examSession.answers[questionId] = answer;
        }
        // POST to server for real persistence
        const session = this.examSession;
        if (session && session.studentId && session.examId) {
            fetch(`${window.API_BASE || '/api'}/exam-sessions/${encodeURIComponent(session.studentId)}/${encodeURIComponent(session.examId)}/answers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question_id: questionId, answer }),
            }).catch(err => console.warn('[autosave] Server save failed:', err));
        }
    },

    getExamElapsed() {
        if (!this.examSession) return 0;
        return Math.floor((Date.now() - this.examSession.startTime) / 1000);
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
window.App = App;
