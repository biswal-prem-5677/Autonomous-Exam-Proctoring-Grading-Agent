# Autonomous Exam Proctoring & Grading Agent — Tasks

## Phase 1: Foundation (Completed ✅)
- [x] Exam engine core (src/exam/)
- [x] Question bank and question types
- [x] Session management
- [x] Timer and auto-submit

## Phase 2: Grading Engine (Completed ✅)
- [x] MCQ grading with exact match
- [x] Numerical grading with tolerance
- [x] TF-IDF cosine similarity for short answers
- [x] Keyword coverage scoring
- [x] Unified grading orchestrator

## Phase 3: Proctoring Sensors (Completed ✅)
- [x] Webcam sensor with MediaPipe face detection
- [x] Keyboard event capture
- [x] Mouse movement tracking
- [x] Audio processor (RMS, FFT features)

## Phase 4: Vision Engine (Completed ✅)
- [x] Face presence detection
- [x] Face count detection (multiple faces)
- [x] Head pose estimation
- [x] Movement analysis

## Phase 5: ML Models (Completed ✅)
- [x] Logistic Regression (scratch implementation)
- [x] Anomaly detection (Z-score + Mahalanobis)
- [x] Linear Regression for performance prediction
- [x] Model calibration utilities

## Phase 6: Autonomous Agent (Completed ✅)
- [x] Agent state machine (NORMAL → SUSPICIOUS → HIGH_RISK)
- [x] Evidence memory with temporal sliding window
- [x] Risk engine with severity weights
- [x] Risk decay formula: R_t = α × R_{t-1} + (1-α) × E_t
- [x] Multi-signal convergence logic

## Phase 7: Reports (Completed ✅)
- [x] Student report generator
- [x] Examiner report generator
- [x] Evidence timeline log
- [x] Performance prediction

## Phase 8: UI/UX (In Progress 🔄)
- [x] Student exam interface
- [x] Examiner dashboard
- [ ] Premium UI styling (avoid blue, sophisticated palette)

## Phase 9: Documentation (In Progress 🔄)
- [x] README.md
- [ ] TASKS.md (this file)
- [ ] PLANNING.md
- [ ] FEATURES.md
- [ ] DELIVERABLES.md
- [ ] FLOW.md
- [ ] HOW_TO_USE.md
- [ ] SECURITY.md
- [ ] Context graph / project memory

## Phase 10: GitHub & Release (Pending ⏳)
- [ ] Create GitHub repository
- [ ] Push all code
- [ ] Create releases/tags

---

## Current Priorities

1. **Premium UI** — Replace blue tones with sophisticated palette (deep purple, emerald, charcoal)
2. **Context Graph** — Project memory file for session continuity
3. **Execution Rules** — Force completion guidelines
4. **Final Documentation** — All remaining docs
5. **GitHub Push** — Push everything

---

## Notes

- No external APIs used — all ML is mathematical and self-developed
- Focus on explainability over black-box AI
- Risk score = evidence indicator, NOT proof of cheating
