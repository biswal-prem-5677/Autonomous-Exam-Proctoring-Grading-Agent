# PLANNING.md — Project Plan

## Vision

A fully local, mathematics-driven autonomous exam intelligence system that converts raw multimodal exam signals into proctoring evidence, cheating-risk prediction, answer evaluation, and student performance analytics — with zero external API dependencies.

## Research Question

> Can a lightweight, locally executable, multimodal behavioral model with temporal evidence fusion detect suspicious examination behavior more reliably and explainably than isolated rule-based signals?

---

## System Overview

```
                  ┌───────────────────────────┐
                  │       EXAM SESSION        │
                  └─────────────┬─────────────┘
                                │
          ┌──────────┬───────────┼───────────┬──────────┐
          │          │           │           │          │
          ▼          ▼           ▼           ▼          ▼
       Webcam    Keyboard     Mouse      Exam UI   Microphone
          │          │           │           │          │
          ▼          ▼           ▼           │          ▼
    Face/Movement  Typing    Mouse Stats   Answers   Audio DSP
          │          │           │           │          │
          └──────────┴───────────┴───────────┘
                              │
                       FEATURE ENGINE
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Risk Classifier  Anomaly Det.  Performance Reg.
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                       TEMPORAL FUSION
                              │
                              ▼
                    AUTONOMOUS AGENT
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Evidence Log    Risk Score     Agent State
                              │
              ┌───────────────┘
              ▼
       GRADING ENGINE
      (MCQ / Numerical / TF-IDF Short Answer)
              │
              ▼
     PERFORMANCE PREDICTION
              │
              ▼
      FINAL REPORT
```

---

## Four ML/Math Modules

| Module | Purpose | Mathematics |
|--------|---------|-------------|
| **A. Behavioral Risk Classifier** | Cheating probability | Logistic Regression (gradient descent) |
| **B. Anomaly Detector** | Statistical unusualness | Z-score aggregation → Mahalanobis distance |
| **C. Automated Grading** | Answer evaluation | TF-IDF, cosine similarity, keyword coverage |
| **D. Performance Predictor** | Future score regression | Linear Regression, Random Forest Regression |

---

## Phased Build Plan

| Phase | Focus | Deliverable | Status |
|-------|-------|-------------|--------|
| **1. Foundation** | Exam engine, question bank, DB, timer | Working exam platform | ✅ Done |
| **2. Grading** | MCQ, numerical, TF-IDF, similarity | Auto-grading pipeline | ✅ Done |
| **3. Proctoring** | Face, presence, gaze, audio, keyboard/mouse | Signal acquisition | ✅ Done |
| **4. ML** | Feature engineering, logistic regression, anomaly detection, regression | Trained models | ✅ Done |
| **5. Agent** | Temporal state, risk fusion, risk decay, evidence memory | Autonomous agent | ✅ Done |
| **6. Evaluation** | Precision, recall, F1, ROC-AUC, MAE, RMSE, ablation | Evaluation results | 🔄 In progress |
| **7. Final Product** | Full UI, dashboards, reports | Complete system | 🔄 In progress |

---

## Architecture Decisions

### Why no external APIs?
- Privacy: No biometric data leaves the machine
- Offline: Works without internet
- Mathematics: Core algorithms are self-implemented
- Explainability: Every decision is traceable

### Why Logistic Regression for risk?
- Mathematically explainable (sigmoid function, gradient descent)
- Weight interpretability: each feature's contribution is visible
- Fast inference (no GPU required)
- Well-suited for binary classification with limited data

### Why temporal risk decay?
- One suspicious event shouldn't permanently affect the score
- Normal behavior after an event should reduce risk
- Formula: R_t = α × R_{t-1} + (1-α) × E_t

### Why multi-signal confirmation?
- Single signal false positives are common (e.g., looking away to think)
- Multiple independent signals converging = much stronger evidence
- Required: at least 2 independent signal categories before escalation

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.10+ | ML ecosystem, rapid prototyping |
| Math/ML | NumPy, SciPy | Core computation |
| Vision | OpenCV, MediaPipe | Local face detection (no API) |
| NLP | TF-IDF, cosine (self) | Self-implemented, no LLM |
| UI | Streamlit | Rapid prototyping, web UI |
| Storage | SQLite | Lightweight, local-first |
| Viz | Matplotlib, Plotly | Charts and reports |
| Tests | pytest | Test coverage |

---

## Scope Boundaries

**IN SCOPE:**
- Local webcam face detection
- Keyboard/mouse behavior tracking
- Temporal feature aggregation
- Logistic regression for risk scoring
- Anomaly detection (statistical)
- TF-IDF + cosine similarity grading
- Performance regression
- Evidence timeline
- Explainable reports

**OUT OF SCOPE:**
- Custom CNN face detector
- Speech recognition
- LLM-based grading
- Facial emotion recognition
- 360° room surveillance
- Cloud deployment
- Mobile app
- Blockchain records

---

## Success Criteria

1. ✅ All core ML modules implemented mathematically (no API calls)
2. ✅ Risk score with evidence breakdown (explainable)
3. ✅ Grading pipeline for MCQ, numerical, and short-answer
4. ✅ Agent state machine with proper transitions
5. ✅ Risk decay working (temporal behavior)
6. ✅ Test suite passing
7. 🔄 Premium UI (avoiding blue defaults)
8. 🔄 Complete documentation
9. ⏳ GitHub repository live
10. ⏳ Final report generation
