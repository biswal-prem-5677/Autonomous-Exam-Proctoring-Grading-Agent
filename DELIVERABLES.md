# DELIVERABLES.md — Deliverables

## Software Deliverables

### 1. Core Application
- `main.py` — CLI entry point with demo, grade, and test commands
- `app/student/exam.py` — Streamlit student exam interface
- `app/examiner/dashboard.py` — Streamlit examiner dashboard

### 2. Source Code Modules

#### Exam Engine (`src/exam/`)
- `exam.py` — Exam configuration and rules
- `questions.py` — Question data model
- `session.py` — Exam session management

#### Sensors (`src/sensors/`)
- `webcam.py` — MediaPipe face detection with OpenCV
- `keyboard.py` — Keystroke event tracking
- `mouse.py` — Mouse movement/click tracking
- `__init__.py`

#### Vision (`src/vision/`)
- `face.py` — Face detection and count
- `landmarks.py` — Facial landmark extraction
- `head_pose.py` — Head orientation estimation
- `movement.py` — Frame-to-frame movement analysis

#### Audio (`src/audio/`)
- `processor.py` — RMS, FFT, ZCR, spectral features

#### Feature Engineering (`src/features/`)
- `behavioral.py` — 22-dimensional behavioral feature extraction
- `temporal.py` — Rolling window aggregation with trend detection
- `normalization.py` — Z-score normalization with inverse transform

#### ML Models (`src/ml/`)
- `logistic.py` — Logistic regression from scratch (gradient descent)
- `anomaly.py` — Z-score + Mahalanobis anomaly detection
- `regression.py` — Linear and Random Forest regression
- `calibration.py` — Threshold calibration, model evaluation

#### Grading Engine (`src/grading/`)
- `mcq.py` — MCQ and true/false grading
- `numerical.py` — Tolerance-based numerical grading
- `tfidf.py` — Self-implemented TF-IDF vectorization
- `similarity.py` — Cosine similarity computation
- `scorer.py` — Unified grading orchestrator

#### Autonomous Agent (`src/agent/`)
- `state.py` — State machine with valid transitions
- `risk.py` — Multi-signal risk fusion engine
- `memory.py` — Temporal evidence memory with sliding window
- `controller.py` — Full agent loop (OBSERVE→PERCEIVE→FEATUREIZE→PREDICT→REASON→DECIDE→ACT)

#### Prediction (`src/prediction/`)
- `performance.py` — Score prediction
- `difficulty.py` — Question difficulty analysis
- `knowledge.py` — Student knowledge profile

#### Reports (`src/reports/`)
- `evidence.py` — Evidence timeline log
- `student_report.py` — Per-student report generator
- `examiner_report.py` — Class-level examiner report

#### Data Models (`src/models/`)
- `models.py` — Question, ExamSession, ProctoringEvent dataclasses

#### Utilities (`src/utils/`)
- `config.py` — Configuration loading

### 3. Data
- `data/sample_data.py` — Sample questions, student answers, proctoring events
- `data/__init__.py`

### 4. Configuration
- `config/default.yaml` — All configurable thresholds and settings

### 5. Tests
- `tests/test_agent.py` — Comprehensive test suite (50+ tests)
- `tests/__init__.py`

### 6. Documentation
- `README.md` — Project overview, architecture, quick start
- `PLANNING.md` — Detailed project plan and architecture decisions
- `FEATURES.md` — Complete feature list with priorities
- `TASKS.md` — Task tracking
- `FLOW.md` — Data and control flow diagrams
- `HOW_TO_USE.md` — Usage guide for students and examiners
- `SECURITY.md` — Security and privacy documentation
- `CONTEXT.md` — Project context graph for session continuity
- `EXECUTION_RULES.md` — Rules for all development sessions
- `CHANGELOG.md` — Version history

---

## Mathematical Deliverables

### Implemented Algorithms

| Algorithm | File | Formula |
|-----------|------|---------|
| Logistic Regression | `src/ml/logistic.py` | σ(z) = 1/(1+e^(-z)), gradient descent |
| Sigmoid | `src/ml/logistic.py` | Stable implementation avoiding overflow |
| Binary Cross-Entropy | `src/ml/logistic.py` | -[y·log(p) + (1-y)·log(1-p)] + L2 reg |
| Z-score | `src/ml/anomaly.py` | z_i = (x_i - μ_i) / σ_i |
| Mahalanobis Distance | `src/ml/anomaly.py` | D_M = sqrt((x-μ)^T Σ^(-1) (x-μ)) |
| Risk Decay | `src/agent/memory.py` | R_t = α·R_{t-1} + (1-α)·E_t |
| Risk Fusion | `src/agent/risk.py` | R = 0.40·base + 0.35·severity + 0.25·convergence |
| TF-IDF | `src/grading/tfidf.py` | TF·IDF = (count/Σ) × log(N/(1+DF)) |
| Cosine Similarity | `src/grading/tfidf.py` | (A·B)/(||A||·||B||) |
| Risk Decay Formula | `src/agent/risk.py` | R_t = α × R_{t-1} + (1-α) × E_t |
| Temporal Trend | `src/features/temporal.py` | np.polyfit linear slope |

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
| Sample Data | 6 | ✅ Passing |
| End-to-End Demo | 1 | ✅ Passing |

**Total: ~85 tests**

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
- State transitions (NORMAL → SUSPICIOUS → HIGH_RISK)

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
# Run full demo
python main.py demo

# Grade answers from JSON
python main.py grade answers.json

# Run tests
python main.py test

# Student exam UI
streamlit run app/student/exam.py

# Examiner dashboard
streamlit run app/examiner/dashboard.py
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
