# DELIVERABLES.md — Deliverables

## Software Deliverables

### 1. Core Application
- `main.py` — CLI & Flask web server entry point with `serve`, `create-user`, `demo`, `grade`, and `test` commands
- `app/api/server.py` — Unified REST API & route handlers (3,400+ lines, 76 endpoints)
- `app/templates/` — Jinja2 HTML templates for all role-based interfaces

### 2. Source Code Modules (12,000+ lines)

#### Exam Engine (`src/exam/`)
- `exam.py` — Exam configuration, rules, and lifecycle
- `questions.py` — Question data model (MCQ, numerical, short/long answer, math, code, true/false)
- `session.py` — Exam session management with timer and auto-submit

#### Sensors (`src/sensors/`)
- `webcam.py` — MediaPipe face detection with OpenCV Haar cascade fallback
- `keyboard.py` — Keystroke event tracking (hold time, inter-key latency, idle detection, clipboard)
- `mouse.py` — Mouse movement/click tracking (velocity, distance, idle detection)

#### Vision (`src/vision/`)
- `face.py` — Face detection and count
- `landmarks.py` — Facial landmark extraction
- `head_pose.py` — Head orientation estimation (yaw/pitch/roll)
- `movement.py` — Frame-to-frame displacement tracking

#### Audio (`src/audio/`)
- `processor.py` — RMS energy, FFT, ZCR, spectral centroid features

#### Feature Engineering (`src/features/`)
- `canonical.py` — **Canonical 22-dimensional feature schema** (authoritative single source of truth)
- `behavioral.py` — Behavioral feature extraction using canonical schema
- `temporal.py` — Rolling window aggregation with trend detection
- `normalization.py` — Z-score normalization with inverse transform

#### ML Models (`src/ml/`)
- `logistic.py` — Logistic regression from scratch (gradient descent, L2 regularization)
- `anomaly.py` — Z-score + Mahalanobis anomaly detection
- `regression.py` — Linear and Random Forest regression
- `calibration.py` — Threshold calibration, model evaluation, ROC-AUC
- `pipeline.py` — Training data generation and model pipeline

#### Grading Engine (`src/grading/`)
- `mcq.py` — MCQ and true/false grading
- `numerical.py` — Tolerance-based numerical grading
- `tfidf.py` — Self-implemented TF-IDF vectorization
- `similarity.py` — Cosine similarity computation
- `scorer.py` — Unified grading orchestrator
- `plagiarism.py` — Cohort plagiarism detection
- `consensus.py` — Inter-grader consensus scoring

#### Autonomous Agent (`src/agent/`)
- `controller.py` — Full OBSERVE→PERCEIVE→FEATUREIZE→PREDICT→REASON→DECIDE→ACT loop (643 lines)
- `risk.py` — Multi-signal risk fusion engine
- `memory.py` — Temporal evidence memory with sliding window and exponential decay
- `state.py` — State machine with valid transitions (NORMAL→SUSPICIOUS→HIGH_RISK→REVIEW_REQUIRED)
- `counterfactual.py` — Counterfactual risk replay engine
- `uncertainty.py` — Uncertainty quantification engine
- `technical.py` — Technical event engine

#### Cross-Modal Fusion (`src/fusion/`)
- Cross-modal corroboration and contradiction detection

#### Prediction (`src/prediction/`)
- `performance.py` — Score prediction using trained regression models
- `difficulty.py` — Question difficulty analysis (Item Response Theory)
- `knowledge.py` — Student knowledge profiling (Bayesian knowledge tracing)

#### Reports (`src/reports/`)
- `evidence.py` — Evidence timeline log
- `student_report.py` — Individual student reports
- `examiner_report.py` — Class-level analytics

#### Trace & Audit (`src/trace/`)
- `audit_chain.py` — SHA-256 tamper-evident hash chain
- `replay.py` — Forensic replay engine
- Decision trace builder with explainability

#### Database Layer (`src/database/`)
- SQLite with WAL journaling, thread-local connections
- Auth/RBAC tables with JWT + bcrypt
- Exam, session, roster, override, and action persistence

#### Data Models (`src/models/`)
- `models.py` — Question, ExamSession, ProctoringEvent, AgentState dataclasses

#### Utilities (`src/utils/`)
- `config.py` — Configuration loading from `config/default.yaml`

### 3. Web Application (12,700+ lines)

#### API Server (`app/api/server.py`)
- 76 REST endpoints covering exam CRUD, session management, proctoring telemetry, grading, analytics, training, and admin

#### Frontend Templates (`app/templates/`)
- `base.html` — Unified navigation layout with role gating
- `index.html` — Public landing portal
- `dashboard.html` — Admin & session dashboard
- `analytics.html` — Real database performance analytics
- `training.html` — ML training & calibration studio
- `examiner_dashboard.html` — Examiner live monitor & interventions
- `auth/` — Login & registration views
- `examiner/` — Create exam, grading, explainability interfaces
- `monitor/` — Live telemetry & trace replay
- `student/` — Proctored exam runner & results views

### 4. Data
- `data/sample_data.py` — Sample questions, student answers, proctoring events

### 5. Configuration
- `config/default.yaml` — All configurable thresholds and settings
- `.env.example` — Environment variable template

### 6. Tests (5,500+ lines, 400+ tests)
- `tests/test_agent.py` — Core agent tests
- `tests/test_defect_regressions.py` — Regression test suite for all 13 defect fixes
- `tests/test_ui_e2e.py` — End-to-end browser & workflow tests
- `tests/test_sandbox.py` — Code execution sandbox security tests
- `tests/test_vision.py` — Face tracking & posture tests
- `tests/test_world_class_agent.py` — End-to-end system integration tests
- `tests/test_independent_certification.py` — Independent production certification suite

### 7. Documentation
- `README.md` — Project overview, architecture, quick start
- `docs/planning.md` — Architecture & design documentation
- `docs/flow.md` — Data and control flow diagrams
- `docs/deliverables.md` — This file
- `docs/how_to_use.md` — Usage guide for students and examiners
- `docs/security.md` — Security and privacy documentation
- `docs/context.md` — Project context graph
- `docs/execution_rules.md` — Development session rules
- `docs/FEATURE_IMPLEMENTATION_MATRIX.md` — Feature implementation status
- `docs/TEST_INTEGRITY_AUDIT.md` — Test integrity audit
- `docs/UI_TRACEABILITY.md` — Complete route & class traceability matrix
- `docs/VERIFICATION_MATRIX.md` — Verification matrix
- `docs/report/main.tex` — Comprehensive LaTeX project report (20+ pages)

---

## Mathematical Deliverables

### Implemented Algorithms

| Algorithm | File | Formula |
|-----------|------|---------|
| Logistic Regression | `src/ml/logistic.py` | σ(z) = 1/(1+e^(-z)), gradient descent with L2 |
| Stable Sigmoid | `src/ml/logistic.py` | Piecewise stable implementation avoiding overflow |
| Binary Cross-Entropy | `src/ml/logistic.py` | -[y·log(p) + (1-y)·log(1-p)] + λ/(2m)·Σw² |
| Z-score | `src/ml/anomaly.py` | z_i = (x_i - μ_i) / σ_i |
| Mahalanobis Distance | `src/ml/anomaly.py` | D_M = sqrt((x-μ)^T Σ^(-1) (x-μ)) |
| Risk Decay | `src/agent/memory.py` | R_t = α·R_{t-1} + (1-α)·E_t |
| Risk Fusion | `src/agent/risk.py` | R = 0.40·base + 0.35·severity + 0.25·convergence |
| ML Refinement | `src/agent/risk.py` | R = 0.60·base + 0.40·logistic |
| TF-IDF | `src/grading/tfidf.py` | TF·IDF = (count/Σ) × log(N/(1+DF)) |
| Cosine Similarity | `src/grading/tfidf.py` | (A·B)/(||A||·||B||) |
| Temporal Trend | `src/features/temporal.py` | np.polyfit linear slope |
| Audit Chain | `src/trace/audit_chain.py` | H_n = SHA-256(H_{n-1} || data_n || ts_n) |

---

## Test Deliverables

### Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Models (Question, AgentState) | 5 | ✅ Passing |
| MCQ Grader | 5 | ✅ Passing |
| Numerical Grader | 4 | ✅ Passing |
| TF-IDF Scorer | 4 | ✅ Passing |
| Keyword Coverage | 4 | ✅ Passing |
| Grading Orchestrator | 2 | ✅ Passing |
| Logistic Regression | 5 | ✅ Passing |
| Anomaly Detector | 4 | ✅ Passing |
| Model Calibration | 4 | ✅ Passing |
| Evidence Memory | 5 | ✅ Passing |
| Risk Engine | 5 | ✅ Passing |
| Agent State Machine | 7 | ✅ Passing |
| Proctoring Agent | 10 | ✅ Passing |
| Feature Extractor | 2 | ✅ Passing |
| Temporal Aggregator | 4 | ✅ Passing |
| Normalization | 4 | ✅ Passing |
| Evidence Log | 5 | ✅ Passing |
| Student Report | 3 | ✅ Passing |
| Examiner Report | 3 | ✅ Passing |
| Defect Regressions | 13 | ✅ Passing |
| Independent Certification | 12 | ✅ Passing |
| UI End-to-End | 15 | ✅ Passing |
| API Integration | 50+ | ✅ Passing |

**Total: 400+ tests**

---

## Evaluation Metrics

### Grading Metrics
- Accuracy (MCQ match rate)
- Numerical tolerance compliance
- Cosine similarity scores for short answers
- Keyword coverage percentages

### Proctoring Metrics
- Risk score ∈ [0, 1]
- Precision, Recall, F1 (when ground truth available)
- False Positive Rate
- State transitions (NORMAL → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED)

### Model Metrics
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression**: MAE, MSE, RMSE, R²
- **Anomaly Detection**: True Positive Rate, False Positive Rate

---

## Deployment Deliverables

### Prerequisites
```
Python 3.10+
pip install -r requirements.txt
```

### Run Commands
```bash
# Start the web application
python main.py serve

# Create user accounts
python main.py create-user --username admin --password admin123 --role admin
python main.py create-user --username examiner1 --password exam123 --role examiner
python main.py create-user --username student1 --password stu123 --role student

# Run tests
python -m pytest

# Open in browser
# http://localhost:5000
```

### No External Dependencies
- No API keys required
- No cloud services
- No internet required (after pip install)
- All ML implemented mathematically

---

## GitHub Repository

**URL:** https://github.com/biswal-prem-5677/Autonomous-Exam-Proctoring-Grading-Agent.git

**Description:** A fully local, mathematics-driven AI exam system for autonomous proctoring, explainable cheating-risk prediction, automated grading, and student performance analysis — with no external AI APIs.

**License:** MIT
