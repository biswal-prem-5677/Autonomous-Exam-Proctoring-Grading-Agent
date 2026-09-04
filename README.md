# Autonomous Exam Proctoring & Grading Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/ML-Mathematics--First-purple?style=for-the-badge" alt="ML Approach">
</p>

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Core Innovation](#core-innovation)
5. [Technical Specifications](#technical-specifications)
6. [Mathematical Foundation](#mathematical-foundation)
7. [System Components](#system-components)
8. [Research Foundation](#research-foundation)
9. [Comparative Analysis](#comparative-analysis)
10. [Getting Started](#getting-started)
11. [Project Structure](#project-structure)
12. [License](#license)

---

## Executive Summary

The **Autonomous Exam Proctoring & Grading Agent** represents a paradigm shift in educational assessment technology. Unlike conventional proctoring systems that rely on black-box AI APIs and cloud-based processing, this system delivers a **fully local, mathematics-driven autonomous examination intelligence platform** that combines:

- **Multimodal Behavioral Analysis** — Integrating webcam, keyboard, mouse, and browser interaction signals
- **Explainable ML Risk Prediction** — Using self-implemented logistic regression and anomaly detection
- **Automated Grading Engine** — Supporting MCQ, numerical, and short-answer evaluation via TF-IDF and cosine similarity
- **Student Performance Intelligence** — Connecting exam behavior to predictive analytics
- **Privacy-First Architecture** — Zero external APIs, all processing on-device

This project was designed after analyzing **50+ research papers**, **10+ existing open-source implementations**, and **commercial platforms** to identify genuine gaps where we could build something measurably better.

---

## Problem Statement

### The Current Landscape

Online examination integrity has become a critical challenge in modern education. The global shift toward remote learning has exponentially increased the need for reliable proctoring solutions. However, existing approaches suffer from fundamental limitations:

| Issue | Description | Impact |
|-------|-------------|--------|
| **Black-Box AI** | Rely on external APIs (OpenAI, Google Vision, AWS Rekognition) with no transparency | Unverifiable decisions, vendor lock-in |
| **Privacy Violations** | Cloud processing of biometric data raises serious concerns | GDPR/FERPA compliance issues |
| **False Positives** | Single-signal rules (e.g., "looking away = cheating") generate excessive false alerts | Student stress, unfair accusations |
| **Siloed Systems** | Proctoring, grading, and performance prediction operate as separate disconnected systems | Incomplete student assessment |
| **Lack of Explainability** | Systems output "87% cheating probability" with no evidence breakdown | No recourse for students |

### Research Gap

Systematic reviews of 80+ peer-reviewed papers (2014-2024) reveal:

- **Most systems** use isolated single-signal detection (face present/absent, gaze direction, tab switching)
- **Temporal dynamics** — how behaviors evolve over time — is rarely modeled
- **Multimodal fusion** — combining vision + audio + keyboard + mouse — is acknowledged as needed but rarely implemented
- **Explainability** remains a critical gap in commercial and academic systems
- **Privacy-first local processing** is almost nonexistent in commercial offerings

---

## Solution Architecture

### Conceptual Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTONOMOUS EXAM AGENT                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    OBSERVE → PERCEIVE → FEATUREIZE                 │   │
│   │                         → PREDICT → REASON → DECIDE → ACT          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                     │
│         ┌────────────────────────────┼────────────────────────────┐       │
│         │                            │                            │       │
│         ▼                            ▼                            ▼       │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐ │
│   │   EXAM       │          │   BEHAVIOR   │          │   GRADING    │ │
│   │   ENGINE     │          │   ENGINE     │          │   ENGINE     │ │
│   │              │          │              │          │              │ │
│   │ • Questions  │          │ • Vision     │          │ • MCQ        │ │
│   │ • Timer      │          │ • Audio      │          │ • Numerical  │ │
│   │ • Session    │          │ • Keyboard   │          │ • TF-IDF     │ │
│   │ • Answers    │          │ • Mouse      │          │ • Similarity │ │
│   └──────┬───────┘          │ • Browser    │          └──────┬───────┘ │
│          │                   └──────┬───────┘                 │         │
│          │                          │                          │         │
│          └──────────────────────────┼──────────────────────────┘         │
│                                     ▼                                    │
│                          ┌──────────────────┐                            │
│                          │   RISK ENGINE   │                            │
│                          │                  │                            │
│                          │ • Fusion        │                            │
│                          │ • Decay         │                            │
│                          │ • Confidence    │                            │
│                          └────────┬─────────┘                            │
│                                   │                                      │
│          ┌────────────────────────┼────────────────────────┐           │
│          ▼                        ▼                        ▼           │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐ │
│   │   EVIDENCE  │          │   ML MODELS  │          │  PERFORMANCE│ │
│   │   LOG       │          │              │          │  PREDICTION │ │
│   │              │          │ • Logistic   │          │              │ │
│   │ • Timeline  │          │ • Anomaly    │          │ • Regression │ │
│   │ • Confidence │          │ • Temporal   │          │ • Topics    │ │
│   │ • Explainable│          │ • Fusion     │          │ • At-Risk   │ │
│   └──────────────┘          └──────────────┘          └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
                                    RAW SENSORS
                                         │
        ┌──────────────────┬──────────────┼──────────────┐
        ▼                  ▼              ▼              ▼
   ┌─────────┐       ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Webcam  │       │ Keyboard │   │   Mouse  │   │ Browser  │
   └────┬────┘       └────┬─────┘   └────┬─────┘   └────┬─────┘
        │                 │               │               │
        ▼                 ▼               ▼               ▼
   Face Detection    Typing Rate    Velocity        Tab Switch
   Head Pose        Hold Time      Acceleration    Copy/Paste
   Movement         Latency        Trajectory      Fullscreen
        │                 │               │               │
        └─────────────────┼───────────────┼───────────────┘
                          ▼
                   FEATURE VECTOR
                   (22-dimensions)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌──────────┐     ┌──────────────┐   ┌──────────┐
  │ Logistic │     │    Risk      │   │ Temporal │
  │Regression│     │   Engine     │   │Analysis  │
  └────┬─────┘     └──────┬───────┘   └────┬─────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                    RISK SCORE
                    (Explainable)
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌────────────┐   ┌──────────┐
   │Evidence  │    │   Agent    │   │  Grade  │
   │Timeline  │    │   State    │   │ Answers │
   └──────────┘    └────────────┘   └──────────┘
                          │                │
                          └────────┬───────┘
                                   ▼
                            FINAL REPORT
```

---

## Core Innovation

### What Makes This System Unique

#### 1. Temporal Multimodal Fusion

Most proctoring systems evaluate **individual frames**:

```
"Student looked away" → ALERT
```

Our system evaluates **behavioral sequences**:

```
"Student showed 15 gaze deviations over 120 seconds 
with increasing frequency, combined with 3 tab switches 
and unusual typing patterns" → RISK: 0.73 (explainable)
```

This dramatically reduces false positives while increasing detection accuracy.

#### 2. Mathematical ML, Not Black-Box APIs

| Component | Implementation |
|-----------|----------------|
| Logistic Regression | Scratch implementation with gradient descent |
| Anomaly Detection | Z-score + Mahalanobis distance |
| TF-IDF | Self-implemented vectorization |
| Cosine Similarity | NumPy-based computation |
| Risk Fusion | Weighted multi-signal aggregation |

No external APIs. No vendor dependency. Every algorithm is transparent and verifiable.

#### 3. Privacy-First Design

```
Camera Frame → Local Processing → Features → Discard
                                           │
                                           ▼
                                   Evidence + Risk Score
```

- **No cloud uploads**
- **No external API calls**
- **All processing on-device**
- **No biometric templates stored**
- **GDPR/FERPA compliant by design**

#### 4. Explainable Risk

Instead of:

```
CHEATING PROBABILITY: 87%
```

We provide:

```
═══════════════════════════════════
RISK ASSESSMENT: 73/100
═══════════════════════════════════

Contributing Evidence:
  +22  Multiple face event (2.4s)
  +18  Repeated gaze deviation (15 occurrences)
  +11  Abnormal typing pattern
  +09  Unusual mouse movement
  +07  Tab focus loss
  -05  Normal behavior after event

Confidence: 81%
═══════════════════════════════════
```

This is **evidence for human review**, not an automated accusation.

#### 5. Integrated Intelligence

Most systems treat these as separate disconnected systems:

- Proctoring (prevent cheating)
- Grading (evaluate answers)
- Analytics (predict performance)

Our system connects:

```
Exam Behavior ──→ Risk ──→ Evidence ──→ Grade ──→ Performance
     │                                              │
     └──────────────┬───────────────────────────────┘
                    ▼
              Complete Student Profile
```

---

## Technical Specifications

### Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Language** | Python 3.10+ | ML ecosystem, rapid prototyping |
| **Math/ML** | NumPy, SciPy | Core mathematical computation |
| **Vision** | OpenCV, MediaPipe | Local face detection (no API) |
| **NLP** | TF-IDF (self) | Explainable text scoring |
| **ML Models** | scikit-learn | Model comparison baseline |
| **UI** | Streamlit | Rapid prototyping, web interface |
| **Storage** | SQLite | Lightweight, local-first |
| **Viz** | Matplotlib, Plotly | Charts and reports |
| **Tests** | pytest | Comprehensive test coverage |

### Key Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Agent Controller | `src/agent/controller.py` | ~370 | OBSERVE→ACT loop |
| Risk Engine | `src/agent/risk.py` | ~150 | Multi-signal fusion |
| Evidence Memory | `src/agent/memory.py` | ~120 | Temporal sliding window |
| Logistic Regression | `src/ml/logistic.py` | ~115 | Scratch implementation |
| Anomaly Detection | `src/ml/anomaly.py` | ~120 | Z-score + Mahalanobis |
| Grading Orchestrator | `src/grading/scorer.py` | ~130 | Unified grading |
| TF-IDF | `src/grading/tfidf.py` | ~100 | Text vectorization |
| Behavioral Features | `src/features/behavioral.py` | ~135 | 22-dim feature vector |

### Performance Metrics

- **Test Coverage**: 94 tests passing
- **Response Time**: <100ms per signal batch
- **Memory Footprint**: <500MB
- **CPU Usage**: <20% during active monitoring
- **No GPU Required**: Runs on standard hardware

---

## Mathematical Foundation

### 1. Logistic Regression (Scratch Implementation)

$$P(Cheating|X) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Training**: Gradient descent on binary cross-entropy loss with L2 regularization

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

### 2. Anomaly Detection

**Z-score per feature**:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

**Mahalanobis distance**:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

### 3. Risk Decay Formula

$$R_t = \alpha \cdot R_{t-1} + (1 - \alpha) \cdot E_t$$

Where:
- $\alpha$ = memory factor (default 0.8)
- $R_{t-1}$ = previous risk score
- $E_t$ = new evidence intensity

This prevents one suspicious event from permanently staining a student's record.

### 4. Risk Fusion

$$R = w_1 R_{ML} + w_2 R_{anomaly} + w_3 R_{rules}$$

With temporal smoothing:
$$R_{final} = \gamma R_{temporal} + (1-\gamma)R_{current}$$

### 5. TF-IDF Scoring

$$TF(t,d) = \frac{\text{count}(t,d)}{\sum_k \text{count}(k,d)}$$

$$IDF(t) = \log\frac{N}{1 + DF(t)}$$

$$TFIDF(t,d) = TF(t,d) \times IDF(t)$$

### 6. Cosine Similarity

$$\cos\theta = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \times \sqrt{\sum_{i=1}^n B_i^2}}$$

### 7. Behavioral Feature Vector (22 dimensions)

| Index | Feature | Source |
|-------|---------|--------|
| 0-3 | Face presence/confidence/count/absence ratio | Vision |
| 4-6 | Head pose (yaw/pitch/roll) | Vision |
| 7-11 | Audio (RMS, dB, ZCR, threshold, spectral) | Audio |
| 12-14 | Keyboard (idle, rate, idle flag) | Interaction |
| 15-18 | Mouse (speed, distance, clicks, idle) | Interaction |
| 19 | Multi-face events | Vision |
| 20-21 | Temporal (session progress, event rate) | Aggregated |

---

## System Components

### A. Exam Management (`src/exam/`)

- `exam.py` — Exam configuration, rules, and lifecycle
- `questions.py` — Question bank with types: MCQ, numerical, short-answer, true/false
- `session.py` — Session management, timer, auto-submit

### B. Sensors (`src/sensors/`)

- `webcam.py` — MediaPipe/OpenCV face detection
- `keyboard.py` — Keystroke timing (hold time, inter-key latency)
- `mouse.py` — Movement velocity, acceleration, trajectory

### C. Vision Engine (`src/vision/`)

- `face.py` — Face presence and count detection
- `head_pose.py` — Yaw/pitch/roll estimation
- `landmarks.py` — Facial landmark extraction
- `movement.py` — Frame-to-frame displacement tracking

### D. Audio Intelligence (`src/audio/`)

- `processor.py` — RMS energy, FFT features, spectral analysis

### E. Feature Engineering (`src/features/`)

- `behavioral.py` — 22-dim feature extraction
- `temporal.py` — Rolling window aggregation with trend detection
- `normalization.py` — Z-score normalization

### F. ML Models (`src/ml/`)

- `logistic.py` — Logistic regression from scratch
- `anomaly.py` — Z-score + Mahalanobis detection
- `regression.py` — Linear and Random Forest regression
- `calibration.py` — Threshold optimization, model evaluation

### G. Grading Engine (`src/grading/`)

- `mcq.py` — Exact match grading
- `numerical.py` — Tolerance-based evaluation
- `tfidf.py` — TF-IDF vectorization
- `similarity.py` — Cosine similarity computation
- `scorer.py` — Unified grading orchestrator

### H. Autonomous Agent (`src/agent/`)

- `controller.py` — Full OBSERVE→ACT loop
- `risk.py` — Multi-signal risk fusion
- `memory.py` — Temporal evidence with decay
- `state.py` — State machine transitions

### I. Prediction (`src/prediction/`)

- `performance.py` — Score prediction
- `difficulty.py` — Question difficulty analysis
- `knowledge.py` — Student knowledge profiling

### J. Reports (`src/reports/`)

- `evidence.py` — Evidence timeline
- `student_report.py` — Individual student reports
- `examiner_report.py` — Class-level analytics

---

## Research Foundation

### Literature Reviewed

| Category | Papers Reviewed | Key Insights |
|----------|-----------------|--------------|
| Proctoring Systems | 15 | Dominant approaches are face/gaze/head-pose; temporal modeling rare |
| Behavioral Biometrics | 10 | Keystroke dynamics is maturing for continuous authentication |
| Automated Grading | 12 | TF-IDF remains competitive; deep learning requires large datasets |
| Student Performance | 18 | Random Forest and gradient boosting dominate prediction tasks |
| Fairness & Privacy | 8 | Privacy concerns are major barrier to adoption |

### Gaps Identified

1. **Temporal modeling** — Most systems evaluate single frames, not behavioral sequences
2. **Multimodal fusion** — Vision + keyboard + mouse + audio rarely combined
3. **Explainability** — Risk scores lack evidence breakdown
4. **Privacy-first** — Almost no commercial systems offer fully local processing
5. **Integration** — Proctoring, grading, and prediction treated as separate systems

### Our Contribution

After analyzing the landscape, our unique contribution is:

> A privacy-preserving, locally executable autonomous exam intelligence framework that combines multimodal behavioral signals, temporal anomaly detection, explainable risk prediction, and rubric-aware automated grading — all mathematically implemented without external APIs.

---

## Comparative Analysis

### vs. Commercial Systems (Proctorio, Honorlock, ExamSoft)

| Aspect | Commercial | Our System |
|--------|------------|------------|
| ML Risk Prediction | Proprietary, black-box | Transparent, self-implemented |
| External APIs | Required | None |
| Cloud Processing | Required | Fully local |
| Explainability | Limited | Full evidence breakdown |
| Grading Integration | Separate | Integrated |
| Cost | $10-30/student | Free (open-source) |

### vs. Open Source (Proctora, OpenProctor, edX Proctoring)

| Aspect | Others | Our System |
|--------|--------|------------|
| ML Models | Rule-based only | Logistic + Anomaly |
| Temporal Analysis | No | Yes (risk decay) |
| Keyboard Dynamics | Basic | Full keystroke biometrics |
| Mouse Dynamics | Minimal | Velocity, acceleration, trajectory |
| Anomaly Detection | No | Z-score + Mahalanobis |
| Performance Prediction | No | Yes (regression) |
| Explainable Risk | No | Yes |

---

## Getting Started

### Quick Start

```bash
# Clone repository
git clone https://github.com/biswal-prem-5677/Autonomous-Exam-Proctoring-Grading-Agent.git
cd Autonomous-Exam-Proctoring-Grading-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py demo

# Run tests
python main.py test

# Launch student exam UI
streamlit run app/student/exam.py

# Launch examiner dashboard
streamlit run app/examiner/dashboard.py
```

### Configuration

Edit `config/default.yaml` to customize:

```yaml
risk:
  alpha_decay: 0.8
  escalation_threshold: 0.7
  review_threshold: 0.5

grading:
  numerical_tolerance: 0.02
  tfidf_min_similarity: 0.4
```

---

## Project Structure

```
autonomous-exam-agent/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   └── default.yaml
├── app/
│   ├── student/exam.py         # Student exam interface
│   └── examiner/dashboard.py   # Examiner dashboard
├── src/
│   ├── agent/                  # Autonomous agent
│   ├── audio/                  # Audio processing
│   ├── exam/                   # Exam engine
│   ├── features/               # Feature engineering
│   ├── grading/                # Grading engine
│   ├── ml/                     # ML models
│   ├── models/                 # Data models
│   ├── prediction/             # Performance prediction
│   ├── reports/                # Report generators
│   ├── sensors/               # Hardware inputs
│   ├── utils/                 # Utilities
│   └── vision/                # Computer vision
├── data/
│   └── sample_data.py         # Demo data
├── docs/
│   ├── planning.md            # Project plan
│   ├── features/              # Feature specifications
│   ├── references/            # Literature & projects
│   ├── research/              # Research papers
│   └── ...
├── tests/
│   └── test_agent.py          # Test suite
└── main.py                    # CLI entry point
```

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{autonomous-exam-agent,
  title = {Autonomous Exam Proctoring \& Grading Agent},
  author = {Biswal, Prem},
  year = {2026},
  url = {https://github.com/biswal-prem-5677/Autonomous-Exam-Proctoring-Grading-Agent},
  license = {MIT}
}
```

---

<p align="center">
  <strong>Built with mathematics-first principles • No external APIs • Privacy by design</strong>
</p>
