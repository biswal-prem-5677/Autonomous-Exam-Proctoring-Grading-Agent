# SECURITY.md — Security & Privacy Documentation

## Privacy-First Design

This system is designed with privacy as a core principle. All processing happens locally — no data leaves the machine running the exam.

### Data Residency
- **No cloud uploads**: All exam data, video, audio, and keystrokes are processed locally
- **No external APIs**: Zero calls to external services (no OpenAI, no cloud vision, no speech APIs)
- **SQLite storage**: All data stored in local SQLite database at `data/exam_agent.db`
- **Local ML**: All models run inference on the local machine

### Data Collected (During Exam)

| Data Type | Purpose | Retention |
|-----------|---------|-----------|
| Webcam frames | Face presence, count, head pose | Session only |
| Keystrokes | Typing rate, idle detection, paste detection | Session only |
| Mouse events | Movement patterns, click frequency | Session only |
| Audio levels | RMS energy, volume detection (optional) | Session only |
| Exam answers | Grading | Until exam ends |
| Proctoring events | Evidence log for reports | Until reports generated |

### Data NOT Collected
- No facial recognition database
- No voice recordings
- No biometric templates
- No browsing history
- No screen recordings (unless explicitly enabled)
- No personal information beyond student ID

---

## Threat Model

### What This System Protects Against
1. **Cheating detection**: Multi-signal behavioral analysis with explainable evidence
2. **Collusion detection**: Multiple face detection
3. **External resource use**: Tab switching, copy-paste detection
4. **Impersonation**: Face presence monitoring

### What This System Does NOT Protect Against
1. **Screen sharing software** (e.g., Skype, Discord screen share)
2. **Secondary devices** (phone, tablet not in frame)
3. **Pre-written notes** (physical or digital)
4. **VPN/proxy use** (not applicable — fully local)
5. **VM escape** (if running inside a VM)

### Limitations
- This system is a **risk indicator**, not definitive proof
- Risk scores are designed for **human review**
- False positives are expected and acceptable — the system errs on the side of flagging for review rather than missing suspicious behavior
- The multi-signal approach reduces (but doesn't eliminate) false positives

---

## Security Measures

### Input Validation
- All signal inputs are validated before processing
- Missing/None values handled with safe defaults
- Feature vectors are clamped to valid ranges

### Safe Defaults
- Camera index: 0 (default webcam)
- Confidence threshold: 0.7 (conservative)
- Risk escalation: 0.7 (requires strong evidence)
- Review threshold: 0.5 (multiple signals required)

### Error Handling
- Graceful degradation if MediaPipe unavailable (falls back to Haar cascades)
- Graceful degradation if audio libraries unavailable
- No unhandled exceptions in the agent loop
- Safe state transitions in the agent state machine

### Data Sanitization
- No raw frames stored — only extracted features
- No audio samples stored — only RMS energy values
- Evidence entries contain only event type, confidence, and timestamp
- Reports contain no raw sensor data

---

## Ethical Guidelines

### Risk Score ≠ Proof of Cheating

> **Risk scores are evidence indicators, NOT proof of cheating.**

The system is designed to assist human examiners, not replace them. All flagged cases require human review before any action is taken.

### Explainability

Every risk score is accompanied by:
- Contributing evidence list
- Individual signal severities
- Timestamped event log
- ML prediction probability (if model available)

This allows examiners to understand WHY the system flagged a student.

### Fairness

- No demographic data collected (no race, gender, age analysis)
- Thresholds are calibrated for general populations
- The system does not make irreversible decisions
- Students can request evidence review

### Transparency

- All algorithms are open-source
- Mathematical formulas are documented in code comments
- Weightings are configurable in `config/default.yaml`
- Test suite is publicly available

---

## Development Security

### No Secrets in Code
- No API keys hardcoded
- `.env.example` provided for any future environment variables
- No credentials in git

### Dependencies
- All dependencies pinned to specific versions in `requirements.txt`
- Known CVEs reviewed for core dependencies
- Minimal dependency tree

### Code Review
- All changes should be reviewed before merging
- Test suite must pass before deployment
- No new external dependencies without discussion

---

## Compliance Notes

This system is designed for educational institutions. Before deployment:

1. **Check local regulations** on proctoring software (GDPR, FERPA, etc.)
2. **Obtain informed consent** from students before monitoring
3. **Define data retention policy** — how long are proctoring records kept?
4. **Appeal process** — students must have a way to contest flags
5. **Accessibility** — ensure the system doesn't disadvantage students with disabilities

---

## Incident Response

If a security issue is found:

1. Report to: [maintainer email]
2. Do NOT open a public GitHub issue for security vulnerabilities
3. Response SLA: 72 hours
4. Fix release target: 7 days
