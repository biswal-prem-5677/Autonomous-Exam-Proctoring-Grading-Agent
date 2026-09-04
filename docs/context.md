# CONTEXT.md — Project Context Graph

## Quick Reference

This file provides context for any new development session to understand the project without re-reading all files.

---

## Project Overview

**Name:** Autonomous Exam Proctoring & Grading Agent  
**Sub-theme:** ML & Predictive Analysis  
**Constraint:** No external APIs · Mathematics-first · Self-developed

**GitHub:** https://github.com/biswal-prem-5677/Autonomous-Exam-Proctoring-Grading-Agent.git

---

## Core Philosophy

1. **No external APIs** — all ML is mathematical and self-developed
2. **Risk ≠ proof** — scores are evidence for human review, not automatic accusations
3. **Explainability** — every decision is traceable
4. **Privacy-first** — all data stays local

---

## Key Components

### ML Modules (4 total)
| Module | File | Purpose |
|--------|------|---------|
| Risk Classifier | `src/ml/logistic.py` | Logistic regression (scratch impl) |
| Anomaly Detector | `src/ml/anomaly.py` | Z-score + Mahalanobis |
| Grading Engine | `src/grading/` | TF-IDF + cosine + keywords |
| Performance Predictor | `src/prediction/` | Linear/Random Forest regression |

### Agent System
| Component | File | Purpose |
|-----------|------|---------|
| Controller | `src/agent/controller.py` | Full OBSERVE→ACT loop |
| Risk Engine | `src/agent/risk.py` | Multi-signal fusion |
| Memory | `src/agent/memory.py` | Temporal evidence with decay |
| State Machine | `src/agent/state.py` | Valid transitions |

### Key Formulas
- **Risk Decay:** `R_t = α × R_{t-1} + (1-α) × E_t`
- **Risk Fusion:** `R = 0.40×base + 0.35×severity + 0.25×convergence`
- **TF-IDF:** `TF × log(N / (1+DF))`
- **Cosine Similarity:** `(A·B) / (||A|| × ||B||)`

---

## File Structure

```
src/
├── agent/          # Autonomous agent (core intelligence)
├── audio/         # Audio processing
├── exam/          # Exam engine
├── features/      # Feature extraction
├── grading/      # Answer grading
├── ml/            # ML models (logistic, anomaly, regression)
├── models/        # Data models
├── prediction/    # Performance prediction
├── reports/       # Report generators
├── sensors/       # Hardware inputs (webcam, keyboard, mouse)
├── utils/         # Config utilities
└── vision/       # Computer vision

app/
├── student/exam.py       # Student UI
└── examiner/dashboard.py # Examiner UI
```

---

## Run Commands

```bash
# Demo (CLI)
python main.py demo

# Student UI
streamlit run app/student/exam.py

# Examiner Dashboard
streamlit run app/examiner/dashboard.py

# Tests
python main.py test
```

---

## Current Status

- [x] Core ML models implemented
- [x] Agent system working
- [x] Grading pipeline complete
- [x] Test suite passing (~85 tests)
- [x] Documentation complete (README, PLANNING, FEATURES, DELIVERABLES, FLOW, HOW_TO_USE, SECURITY)
- [ ] Premium UI styling
- [ ] GitHub push

---

## Recent Changes

- Created comprehensive documentation (PLANNING.md, FEATURES.md, DELIVERABLES.md, FLOW.md, HOW_TO_USE.md, SECURITY.md, TASKS.md)
- Verified all tests passing
- Verified demo runs correctly

---

## Key Config

See `config/default.yaml` for thresholds:
- `risk.alpha_decay: 0.8`
- `risk.escalation_threshold: 0.7`
- `risk.review_threshold: 0.5`
- `ml.zscore_threshold: 2.5`

---

## Important Notes for Future Sessions

1. **DO NOT add external APIs** — keep it mathematics-first
2. **Risk is evidence, not proof** — always include this disclaimer in UIs
3. **Run tests before pushing** — `python main.py test`
4. **Update docs** when adding features
5. **Premium UI** — avoid blue tones, use sophisticated palette (deep purple, emerald, charcoal)
