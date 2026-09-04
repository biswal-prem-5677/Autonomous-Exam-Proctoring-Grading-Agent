# Autonomous Exam Proctoring & Grading Intelligence Agent

**Sub-theme:** ML & Predictive Analysis
**Constraint:** No external AI/LLM/vision APIs · Mathematics-first · Self-developed

---

## What This Project Is

An autonomous, fully-local examination intelligence system that converts raw multimodal exam signals into proctoring evidence, cheating-risk prediction, answer evaluation, and student performance analytics — with zero external API dependencies.

## Core Philosophy

The system does **not** simply "watch via webcam." It is a mathematics-driven decision engine that:

1. **Observes** — captures webcam, keyboard, mouse, and exam-interface signals locally
2. **Perceives** — extracts mathematically grounded features from each signal stream
3. **Predicts** — applies supervised ML (logistic regression), unsupervised anomaly detection, and regression models
4. **Reasons** — fuses evidence temporally using weighted evidence aggregation and risk decay
5. **Decides** — operates a state-machine agent that records evidence and escalates for human review
6. **Reports** — produces explainable grading reports with cheating-risk indicators, evidence timelines, and performance predictions

**Risk score ≠ proof of cheating.** The system generates evidence and risk confidence for examiner review, never automatic accusation.

## System Architecture (High-Level)

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
                     (Observe → Perceive → Predict
                      → Reason → Decide → Act)
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

## Four ML/Math Modules (Tier 1+2)

| Module | Purpose | Mathematics |
|--------|---------|-------------|
| **A. Behavioral Risk Classifier** | Cheating probability | Logistic Regression (gradient descent) |
| **B. Anomaly Detector** | Statistical unusualness | Z-score aggregation → Mahalanobis distance |
| **C. Automated Grading** | Answer evaluation | TF-IDF, cosine similarity, keyword coverage |
| **D. Performance Predictor** | Future score regression | Linear Regression, Random Forest Regression |

## Autonomous Agent Loop

```
OBSERVE → PERCEIVE → FEATUREIZE → PREDICT → REASON → DECIDE → ACT → REPEAT
```

**Agent States:** `NORMAL` → `SUSPICIOUS` → `HIGH_RISK` → `REVIEW_REQUIRED`

**Risk Decay:** `R_t = α R_{t-1} + (1-α) E_t` (α = memory factor, prevents permanent scoring from one event)

**Multi-Signal Confirmation:** Independent evidence from vision + audio + keyboard + mouse + temporal must converge before escalation.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Math/ML | NumPy, SciPy, scikit-learn |
| Vision | OpenCV, MediaPipe (local inference) |
| NLP | TF-IDF, cosine similarity (self-implemented) |
| UI | Streamlit |
| Storage | SQLite |
| Viz | Matplotlib, Plotly |

**No external APIs. No cloud services. No LLM calls. Everything runs locally.**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch student exam interface
streamlit run app/student/main.py

# Launch examiner dashboard
streamlit run app/examiner/main.py
```

## Repository Structure

```
autonomous-exam-agent/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── config/
│   ├── exam_config.yaml
│   ├── model_config.yaml
│   └── thresholds.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── models/
│   ├── risk_model.pkl
│   └── performance_model.pkl
├── src/
│   ├── exam/
│   │   ├── exam.py
│   │   ├── questions.py
│   │   └── session.py
│   ├── sensors/
│   │   ├── webcam.py
│   │   ├── keyboard.py
│   │   └── mouse.py
│   ├── vision/
│   │   ├── face.py
│   │   ├── landmarks.py
│   │   ├── head_pose.py
│   │   └── movement.py
│   ├── features/
│   │   ├── behavioral.py
│   │   ├── temporal.py
│   │   └── normalization.py
│   ├── ml/
│   │   ├── logistic.py
│   │   ├── anomaly.py
│   │   ├── regression.py
│   │   └── calibration.py
│   ├── grading/
│   │   ├── mcq.py
│   │   ├── numerical.py
│   │   ├── tfidf.py
│   │   ├── similarity.py
│   │   └── scorer.py
│   ├── agent/
│   │   ├── state.py
│   │   ├── risk.py
│   │   ├── memory.py
│   │   └── controller.py
│   ├── prediction/
│   │   ├── performance.py
│   │   ├── difficulty.py
│   │   └── knowledge.py
│   └── reports/
│       ├── evidence.py
│       ├── student_report.py
│       └── examiner_report.py
├── tests/
├── experiments/
│   ├── risk_model.ipynb
│   ├── anomaly_model.ipynb
│   ├── grading_model.ipynb
│   └── performance_model.ipynb
└── app/
    ├── student/
    └── examiner/
```

## Phased Build Plan

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **1. Foundation** | Exam engine, question bank, DB, timer | Working exam platform |
| **2. Grading** | MCQ, numerical, TF-IDF, similarity | Auto-grading pipeline |
| **3. Proctoring** | Face, presence, gaze, audio, keyboard/mouse | Signal acquisition |
| **4. ML** | Feature engineering, logistic regression, anomaly detection, regression | Trained models |
| **5. Agent** | Temporal state, risk fusion, risk decay, evidence memory | Autonomous agent |
| **6. Evaluation** | Precision, recall, F1, ROC-AUC, MAE, RMSE, ablation | Evaluation results |
| **7. Final Product** | Full UI, dashboards, reports | Complete system |

## Research Question

> **Can a lightweight, locally executable, multimodal behavioral model with temporal evidence fusion detect suspicious examination behavior more reliably and explainably than isolated rule-based signals?**

## Central Mathematical Formula

```
Raw Signals → Mathematical Features → ML Prediction → Temporal Reasoning → Risk → Evidence → Decision
```

## License

MIT
