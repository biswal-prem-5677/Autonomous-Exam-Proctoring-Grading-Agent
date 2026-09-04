/**
 * ExamAI — API Client
 * Centralized fetch wrapper for all backend API calls.
 */

const API_BASE = '/api';

class API {
    static async get(endpoint) {
        try {
            const res = await fetch(`${API_BASE}${endpoint}`);
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
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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

    // ── Exam Sessions ────────────────────────────────────────────
    static async startExam(data) {
        return this.post('/exam-sessions', data);
    }

    static async getSessionStatus(studentId, examId) {
        return this.get(`/exam-sessions/${studentId}/${examId}/status`);
    }

    static async submitAnswer(studentId, examId, answer) {
        return this.post(`/exam-sessions/${studentId}/${examId}/answers`, answer);
    }

    static async endExam(studentId, examId) {
        return this.post(`/exam-sessions/${studentId}/${examId}/submit`);
    }

    // ── Grading ──────────────────────────────────────────────────
    static async gradeAnswers(data) {
        return this.post('/grade', data);
    }

    static async gradeSingle(data) {
        return this.post('/grade/single', data);
    }

    // ── Proctoring ───────────────────────────────────────────────
    static async processSignals(sessionKey, signals) {
        return this.post('/proctor/process', { session_key: sessionKey, signals });
    }

    static async getProctorStatus(studentId, examId) {
        return this.get(`/proctor/status/${studentId}/${examId}`);
    }

    static async getProctorHistory(studentId, examId) {
        return this.get(`/proctor/history/${studentId}/${examId}`);
    }

    static async getEvidence(studentId, examId) {
        return this.get(`/proctor/evidence/${studentId}/${examId}`);
    }

    // ── Examiner ─────────────────────────────────────────────────
    static async getExamOverview(examId) {
        return this.get(`/examiner/exam/${examId}/overview`);
    }

    static async getStudentDetail(studentId, examId) {
        return this.get(`/examiner/student/${studentId}/${examId}`);
    }

    // ── Reports ──────────────────────────────────────────────────
    static async generateStudentReport(data) {
        return this.post('/reports/student', data);
    }

    static async generateExaminerReport(data) {
        return this.post('/reports/examiner', data);
    }

    // ── Training ─────────────────────────────────────────────────
    static async generateTrainingData(data) {
        return this.post('/train/generate-data', data);
    }

    static async trainModels(data) {
        return this.post('/train/models', data);
    }

    static async runEvaluation(data) {
        return this.post('/train/evaluate', data);
    }

    // ── Prediction ───────────────────────────────────────────────
    static async predictPerformance(data) {
        return this.post('/predict/performance', data);
    }

    // ── Demo ─────────────────────────────────────────────────────
    static async demoSetup(data) {
        return this.post('/demo/setup', data);
    }

    static async generateDemoAnswers(data) {
        return this.post('/demo/generate-answers', data);
    }
}

// Export for use in other scripts
window.API = API;
