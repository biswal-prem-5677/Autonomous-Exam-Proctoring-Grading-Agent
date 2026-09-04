# References — Open Source Projects, Platforms & Datasets

## Open-Source Proctoring Projects

### 1. Proctora
**GitHub:** https://github.com/Proctora/proctora

**Stack:** Flask, OpenCV, MediaPipe, SQLite

**What it has:**
- Webcam monitoring with MediaPipe face detection
- Microphone noise detection
- Fullscreen exam enforcement
- Tab/window switching detection
- Violation logging with database-backed sessions
- Developer-oriented foundation

**What it lacks:**
- No ML risk scoring (rule-based only)
- No temporal evidence fusion
- No anomaly detection
- No grading integration
- No performance prediction
- No explainable risk scores

**Our advantage:** We add supervised ML, anomaly detection, temporal fusion, automated grading, and performance prediction — all mathematically implemented locally.

---

### 2. OpenProctor
**GitHub:** https://github.com/OpenProctor/OpenProctor

**Stack:** Next.js, TypeScript, BlazeFace, WebRTC

**What it has:**
- Browser-side camera capture
- Microphone monitoring
- Screen sharing
- BlazeFace for face detection
- Media streaming
- Server-side storage

**What it lacks:**
- No ML prediction
- No anomaly detection
- No grading
- Cloud-dependent
- No temporal analysis

**Our advantage:** Fully local, no cloud dependency, ML-powered risk prediction, integrated grading.

---

### 3. Open edX Proctoring
**GitHub:** https://github.com/edx/edx-proctoring

**What it has:**
- LMS integration architecture
- Proctored exam lifecycle management
- Exam state machine (created, started, submitted, verified)
- Backend service design
- Multiple proctoring provider support

**What it lacks:**
- No ML-based detection
- No real-time analysis
- No grading integration

**Our advantage:** We studied their state machine design and improved it with ML-driven transitions and evidence-based reasoning.

---

### 4. Safe Exam Browser (SEB)
**Website:** https://safeexambrowser.org/

**What it has:**
- Lockdown browser
- Fullscreen enforcement
- URL whitelisting
- Application blocking
- Screen monitoring
- Multiple monitor detection

**What it lacks:**
- No behavioral analysis
- No ML detection
- No grading

**Our advantage:** We separate prevention (SEB's job) from detection (our job), focusing on behavioral intelligence rather than lockdown.

---

### 5. ASAP-AES Workbench
**GitHub:** https://github.com/benizi/ ASAP-AES-Workbench

**What it has:**
- Automated essay scoring pipeline
- Feature engineering for text
- Cross-validation framework
- Model comparison
- Uses ASAP dataset

**Our advantage:** We use similar mathematical approaches (TF-IDF + cosine) but extend to short answers with keyword/concept coverage and integrate with proctoring.

---

## Commercial Platforms

### 6. Proctorio
**Features:** Configurable lockdown, webcam/screen/audio monitoring, post-exam review

**Weakness:** Cloud-dependent, expensive, black-box AI, privacy concerns

### 7. Honorlock
**Features:** AI monitoring, human review, multi-signal detection (eye, phone, speech, browser)

**Weakness:** Subscription model, cloud processing, not open

### 8. ExamSoft / ExamMonitor
**Features:** Identity verification, video/audio recording, automated flagging, time-stamped incidents

**Weakness:** No ML prediction, purely rule-based, expensive

### 9. ProctorU
**Features:** Live human proctoring + AI

**Weakness:** Human-dependent, expensive, scheduling constraints

### 10. Respondus LockDown Browser
**Features:** Browser lockdown, screen recording, lockdown + Monitor

**Weakness:** Rule-based only, no ML, no grading integration

---

## Datasets

### 11. ASAP (Automated Student Assessment Prize)
**URL:** https://www.kaggle.com/c/asap-aes

**Content:** 8 essay sets, ~13,000 student essays, human-scored
**Use:** Automated essay scoring benchmark

### 12. ASAP 2.0
**Content:** ~24,000 student argumentative essays, prompt/context info
**Use:** Modern, larger-scale essay scoring

### 13. OULAD (Open University Learning Analytics Dataset)
**URL:** https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education

**Content:** 7 courses, 32,000+ students, assessments + VLE interactions
**Use:** Student performance prediction, at-risk identification

### 14. UCI Student Performance Dataset
**URL:** https://archive.ics.uci.edu/ml/datasets/student+performance

**Content:** Student grades, demographics, study habits
**Use:** Performance prediction baseline

### 15. KEEL (Knowledge Extraction based on Evolutionary Learning)
**URL:** https://sci2s.ugr.es/keel/

**Content:** Repository of datasets for evolutionary algorithms
**Use:** Model comparison baselines

### 16. ExamBehavior-Local (OUR DATASET)
**Status:** To be generated

**Content:** Controlled behavioral sessions with labeled events
- Face presence/absence
- Multiple face events
- Head pose measurements
- Keyboard timing (hold time, inter-key latency)
- Mouse dynamics (velocity, acceleration)
- Browser events (tab switch, fullscreen exit)
- Audio RMS energy
- Event labels: NORMAL, LOOKING_AWAY, ABSENCE, MULTIPLE_PERSON, AUDIO_ACTIVITY, TAB_SWITCH, UNUSUAL_TYPING, MULTI_SIGNAL_EVENT

**Generation:** `src/ml/training_data.py` (to be built)

---

## Key Research Papers

See `docs/research/research_papers.md` for the full annotated bibliography.
