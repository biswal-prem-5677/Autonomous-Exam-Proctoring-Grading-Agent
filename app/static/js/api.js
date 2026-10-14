/**
 * ExamAI — API Client
 * Centralized fetch wrapper for all backend API calls.
 */

window.API_BASE = window.API_BASE || '/api';

class API {
    static getHeaders() {
        const token = localStorage.getItem('token') || (window.App && window.App.authToken);
        const headers = { 'Content-Type': 'application/json' };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    static async get(endpoint) {
        try {
            const base = window.API_BASE || '/api';
            const url = endpoint.startsWith('http') ? endpoint : `${base}${endpoint}`;
            const res = await fetch(url, {
                headers: this.getHeaders(),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: res.statusText }));
                throw new Error(err.error || err.message || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.error(`[API GET] ${endpoint}:`, err);
            throw err;
        }
    }

    static async post(endpoint, data = {}) {
        try {
            const base = window.API_BASE || '/api';
            const url = endpoint.startsWith('http') ? endpoint : `${base}${endpoint}`;
            const res = await fetch(url, {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify(data),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: res.statusText }));
                throw new Error(err.error || err.message || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.error(`[API POST] ${endpoint}:`, err);
            throw err;
        }
    }

    static async put(endpoint, data = {}) {
        try {
            const base = window.API_BASE || '/api';
            const url = endpoint.startsWith('http') ? endpoint : `${base}${endpoint}`;
            const res = await fetch(url, {
                method: 'PUT',
                headers: this.getHeaders(),
                body: JSON.stringify(data),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: res.statusText }));
                throw new Error(err.error || err.message || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.error(`[API PUT] ${endpoint}:`, err);
            throw err;
        }
    }

    static async delete(endpoint) {
        try {
            const base = window.API_BASE || '/api';
            const url = endpoint.startsWith('http') ? endpoint : `${base}${endpoint}`;
            const res = await fetch(url, {
                method: 'DELETE',
                headers: this.getHeaders(),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: res.statusText }));
                throw new Error(err.error || err.message || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.error(`[API DELETE] ${endpoint}:`, err);
            throw err;
        }
    }

    // ── Health ────────────────────────────────────────────────────
    static async health() {
        return this.get('/health');
    }

    // ── Exam Management ──────────────────────────────────────────
    static async listExams() {
        return this.get('/exams');
    }

    static async createExam(data) {
        return this.post('/exams', data);
    }

    static async getExam(examId) {
        return this.get(`/exams/${examId}`);
    }

    static async addQuestion(examId, question) {
        return this.post(`/exams/${examId}/questions`, question);
    }

    static async getExamCustomization(examId) {
        return this.get(`/exams/${examId}/customize`);
    }

    static async updateExamCustomization(examId, data) {
        return this.put(`/exams/${examId}/customize`, data);
    }

    // ── Roster & Enrollment ───────────────────────────────────────
    static async getRoster(examId) {
        return this.get(`/exams/${examId}/roster`);
    }

    static async enrollStudent(examId, data) {
        return this.post(`/exams/${examId}/roster`, data);
    }

    static async bulkEnrollStudents(examId, data) {
        return this.post(`/exams/${examId}/roster/bulk`, data);
    }

    static async removeStudentFromRoster(examId, studentId) {
        return this.delete(`/exams/${examId}/roster/${studentId}`);
    }

    // ── Sessions & Live Proctoring ───────────────────────────────
    static async startExamSession(studentId, examId) {
        return this.post('/exam-sessions', { student_id: studentId, exam_id: examId });
    }

    static async submitExamSession(studentId, examId, data = {}) {
        return this.post(`/exam-sessions/${studentId}/${examId}/submit`, data);
    }

    static async sendVisionFrame(data) {
        return this.post('/vision/analyze', data);
    }

    static async sendBehavioralData(data) {
        return this.post('/behavioral/analyze', data);
    }

    static async getProctorStatus(studentId, examId) {
        return this.get(`/proctor/status/${studentId}/${examId}`);
    }

    static async getProctorSummary(studentId, examId) {
        return this.get(`/proctor/summary/${studentId}/${examId}`);
    }

    static async getProctorTrace(studentId, examId) {
        return this.get(`/proctor/trace/${studentId}/${examId}`);
    }

    static async getEvidence(studentId, examId) {
        return this.get(`/proctor/evidence/${studentId}/${examId}`);
    }

    static async updateEvidence(evidenceId, data) {
        return this.put(`/evidence/${evidenceId}`, data);
    }

    // ── Examiner ─────────────────────────────────────────────────
    static async getExamOverview(examId) {
        return this.get(`/examiner/exam/${examId}/overview`);
    }

    static async getStudentDetail(studentId, examId) {
        return this.get(`/examiner/student/${studentId}/${examId}`);
    }

    static async sendExaminerAction(data) {
        return this.post('/examiner/action', data);
    }

    // ── Grading & Overrides ──────────────────────────────────────
    static async submitGrading(data) {
        return this.post('/grading/submit', data);
    }

    static async inspectGrading(studentId, examId) {
        return this.get(`/grading/inspect/${studentId}/${examId}`);
    }

    static async overrideGrade(data) {
        return this.post('/grading/override', data);
    }

    static async setExamRelease(examId, data) {
        return this.post(`/exams/${examId}/release`, data);
    }

    static async checkPlagiarism(data) {
        return this.post('/grading/plagiarism', data);
    }

    static async evaluateConsensus(data) {
        return this.post('/grading/consensus', data);
    }

    // ── Reports & Analytics ──────────────────────────────────────
    static async generateStudentReport(data) {
        return this.post('/reports/student', data);
    }

    static async generateExaminerReport(data) {
        return this.post('/reports/examiner', data);
    }

    static async getAnalytics() {
        return this.get('/analytics');
    }

    static async getKnowledgeState(studentId, examId) {
        return this.get(`/knowledge-state/${encodeURIComponent(studentId)}/${encodeURIComponent(examId)}`);
    }

    // ── Training & ML ────────────────────────────────────────────
    static async generateTrainingData(data) {
        return this.post('/train/generate-data', data);
    }

    static async trainModels(data) {
        return this.post('/training/train', data);
    }

    static async runEvaluation(data) {
        return this.post('/train/evaluate', data);
    }

    static async getTrainingStatus() {
        return this.get('/training/status');
    }

    static async runAdversarialBenchmark(data = {}) {
        return this.post('/ml/adversarial-benchmark', data);
    }

    static async getModelStatus() {
        return this.get('/models/status');
    }

    static async activateModels() {
        return this.post('/models/activate', {});
    }

    // ── Replay & Audit Chain ─────────────────────────────────────
    static async getReplay(sessionId, step = null) {
        const query = step !== null ? `?step=${step}` : '';
        return this.get(`/trace/replay/${sessionId}${query}`);
    }

    static async getAuditChain(sessionId) {
        return this.get(`/trace/audit-chain/${sessionId}`);
    }

    static async verifyAuditChain(sessionId) {
        return this.post(`/trace/audit-chain/${sessionId}/verify`, {});
    }

    // ── Prediction ───────────────────────────────────────────────
    static async predictPerformance(data) {
        return this.post('/predict/performance', data);
    }

    // ── Admin Demo ───────────────────────────────────────────────
    static async demoSetup(data) {
        return this.post('/demo/setup', data);
    }

    static async generateDemoAnswers(data) {
        return this.post('/demo/generate-answers', data);
    }
}

// Export for use in other scripts
window.API = API;
