# Verification Matrix

Evidence-based feature status. Each feature is classified honestly.

Legend:
  [VERIFIED]              — End-to-end verified: real input → real processing → real output → real persistence → real UI
  [PARTIAL]               — Core implementation exists but one or more legs of the pipeline are unverified or incomplete
  [IMPLEMENTED UNVERIFIED]— Code/API/UI exists but end-to-end flow has NOT been confirmed
  [MOCKED]                — Uses hardcoded/demo/fake data instead of real computation
  [STUB]                  — Endpoint exists but returns placeholder/empty response
  [NOT IMPLEMENTED]       — No code exists

---

## 1. Grading Engine

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| MCQ grading | YES | YES | NO direct UI | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Numerical grading | YES | YES | NO direct UI | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Short answer (TF-IDF) | YES | YES | NO direct UI | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Long answer (multi-signal) | YES | YES | NO direct UI | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Grading orchestrator | YES | YES | NO direct UI | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Grading persistence | PARTIAL | YES | NO | NO | PARTIAL | YES | NO | NO | NO | NO | PARTIAL — submit returns result but doesn't persist to DB |
| Grade release | STUB | YES | YES | NO | — | — | NO | NO | NO | NO | STUB — returns success message without persisting state |

**Grading: 0 VERIFIED, 1 PARTIAL, 5 IMPLEMENTED UNVERIFIED, 1 STUB**

---

## 2. Machine Learning

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| Logistic regression | YES | YES | YES | NO artifact | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Anomaly detection | YES | YES | YES | NO artifact | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Training data generator | YES | YES | YES | YES (save/load) | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Training API | PARTIAL | YES | YES | NO | PARTIAL | MOCKED | NO | NO | NO | NO | MOCKED — evaluates on training data, hardcoded status |
| Model calibration | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Performance predictor | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |

**ML: 0 VERIFIED, 1 PARTIAL, 1 MOCKED, 4 IMPLEMENTED UNVERIFIED**

---

## 3. Vision Proctoring

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| Face tracker | YES | YES | YES | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Landmark analyzer | YES | YES | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Movement detector | YES | YES | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Head pose | YES | NO | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Vision→risk pipeline | PARTIAL | YES | YES | NO | PARTIAL | PARTIAL | NO | NO | NO | NO | PARTIAL — vision API returns risk contribution but not connected to live camera in tested flow |
| Live webcam in exam UI | YES | NO | YES | NO | YES | YES | NO | NO | NO | NO | IMPLEMENTED UNVERIFIED |

**Vision: 0 VERIFIED, 2 PARTIAL, 4 IMPLEMENTED UNVERIFIED**

---

## 4. Behavioral Biometrics

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| Keystroke dynamics | YES | NO | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Mouse dynamics | YES | NO | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Behavioral feature extractor | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Browser monitoring | YES | NO | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Audio processing | YES | NO | NO | NO | YES | YES | YES | NO | NO | NO | IMPLEMENTED UNVERIFIED |
| Keystroke/mouse→API | NO | NO | PARTIAL | NO | NO | NO | NO | NO | NO | NO | PARTIAL — browser JS tracks tab/fullscreen but keystroke/mouse not wired to API |

**Biometrics: 0 VERIFIED, 1 PARTIAL, 5 IMPLEMENTED UNVERIFIED**

---

## 5. Agent Controller

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| Agent state machine | YES | YES | YES | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Evidence memory | YES | YES | YES | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Risk engine | YES | YES | YES | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Risk decay formula | YES | NO | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Agent controller loop | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Agent→evidence→risk chain | PARTIAL | YES | PARTIAL | NO | PARTIAL | PARTIAL | NO | NO | NO | NO | PARTIAL — individual components work but the full chain hasn't been demonstrated end-to-end |

**Agent: 0 VERIFIED, 1 PARTIAL, 5 IMPLEMENTED UNVERIFIED**

---

## 6. Database & Persistence

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| SQLite sessions | YES | YES | YES | YES | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Questions CRUD | YES | YES | YES | YES | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Answers persistence | PARTIAL | PARTIAL | YES | PARTIAL | YES | YES | YES | YES | NO | NO | PARTIAL — grading submit doesn't persist answers to DB |
| Events persistence | YES | YES | YES | YES | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Auth (users/roles/orgs) | YES | YES | YES | YES | YES | YES | PARTIAL | NO | NO | NO | IMPLEMENTED UNVERIFIED |

**Database: 0 VERIFIED, 2 PARTIAL, 3 IMPLEMENTED UNVERIFIED**

---

## 7. Reports

| Feature | Backend | API | Frontend | Persistence | Real Input | Real Output | Unit | Integration | Browser | E2E | Status |
|---------|---------|-----|----------|-------------|------------|-------------|------|-------------|---------|-----|--------|
| Student report generator | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Examiner report generator | YES | YES | NO | NO | YES | YES | YES | YES | NO | NO | IMPLEMENTED UNVERIFIED |
| Report API | YES | YES | NO | NO | YES | YES | PARTIAL | NO | NO | NO | IMPLEMENTED UNVERIFIED |

**Reports: 0 VERIFIED, 0 PARTIAL, 3 IMPLEMENTED UNVERIFIED**

---

## 8. REST API

| Feature | Status |
|---------|--------|
| 56 API endpoints exist | IMPLEMENTED UNVERIFIED |
| Authentication on protected routes | PARTIAL — login/register work; token validation exists but not all routes enforce it |
| Error handling | IMPLEMENTED UNVERIFIED |
| CORS configured | YES — verified in code |

**API: 0 VERIFIED, 1 PARTIAL, 1 IMPLEMENTED UNVERIFIED**

---

## 9. Frontend UI

| Feature | Status | Notes |
|---------|--------|-------|
| Landing page | IMPLEMENTED UNVERIFIED | Exists, not browser-tested |
| Login/Register | IMPLEMENTED UNVERIFIED | Forms exist, not browser-tested |
| Dashboard | IMPLEMENTED UNVERIFIED | Sidebar + tabs, not browser-tested |
| Student exam (webcam, questions, timer) | IMPLEMENTED UNVERIFIED | Complex page, not browser-tested |
| Student results | IMPLEMENTED UNVERIFIED | Not browser-tested |
| Examiner grading | IMPLEMENTED UNVERIFIED | Not browser-tested |
| Examiner create exam | IMPLEMENTED UNVERIFIED | Not browser-tested |
| Live monitor | IMPLEMENTED UNVERIFIED | Not browser-tested |
| CSS theme (740 lines) | IMPLEMENTED UNVERIFIED | Not browser-tested |
| JavaScript app framework | IMPLEMENTED UNVERIFIED | Not browser-tested |
| Navigation between pages | IMPLEMENTED UNVERIFIED | Not browser-tested |
| API error handling in JS | PARTIAL | Try/catch exists but no user-friendly error UI |

**Frontend: 0 VERIFIED, 1 PARTIAL, 10 IMPLEMENTED UNVERIFIED**

---

## 10. End-to-End Pipeline (THE critical test)

| Feature | Status |
|---------|--------|
| Student opens browser → camera starts → exam loads | NOT TESTED |
| Camera → face detection → landmarks → features | NOT TESTED |
| Features → risk engine → risk score update | NOT TESTED |
| Risk → evidence creation → examiner sees it | NOT TESTED |
| Student submits → grading runs → result persisted | NOT TESTED |
| Examiner opens → sees student results + evidence | NOT TESTED |
| Full exam lifecycle (create → take → grade → review) | NOT TESTED |

**End-to-End: 0/7 VERIFIED**

---

## Summary

| Status | Count |
|--------|-------|
| VERIFIED | 0 |
| PARTIAL | 5 |
| IMPLEMENTED BUT UNVERIFIED | ~30 |
| MOCKED | 1 |
| STUB | 1 |
| NOT IMPLEMENTED | 7 (E2E flows) |
| **TOTAL FEATURES AUDITED** | **~44** |

### What "IMPLEMENTED UNVERIFIED" means

Code exists. Unit tests pass. But no one has confirmed:
- The page renders in a browser
- The API returns correct data when called from a browser
- Camera/mic permissions work
- Data flows from one component to another
- Persisted data survives process restart
- The examiner can see what the student produced

### What needs to happen to move items to [VERIFIED]

1. Run the Flask server locally
2. Open each page in a browser
3. Exercise each workflow
4. Confirm data flows correctly
5. Confirm persistence works
6. Confirm no console errors
7. Fix anything that breaks
8. Repeat until the full pipeline works

This is the work that remains.
