# Research Papers — Annotated Bibliography

## Category 1: Proctoring & Academic Integrity (MUST READ)

### 1. Ensuring Academic Integrity Through Automated Online Exam Proctoring: A Decade-Long Systematic Review (2026)
**Authors:** Discover Education
**Summary:** Reviews 80 peer-reviewed studies from 2014–2024 covering ML/DL approaches, facial analysis, gaze, head pose, audio, keystrokes, hardware, privacy, and fairness.
**Key Finding:** The dominant approaches include CNN/RNN/DL, SVM/RF, eye movement, head pose, facial behavior, audio and environmental monitoring.
**Relevance:** Maps the entire proctoring landscape — identifies what's already done and what gaps remain.

### 2. Digital Proctoring in Higher Education: A Systematic Literature Review
**Summary:** Analyzes 115 publications around system development, adoption, student performance, legal issues, ethics, security and privacy.
**Relevance:** Provides the student-side perspective — privacy, technical difficulties, fairness concerns are major themes.

### 3. What is the Student Experience of Remote Proctoring? A Pragmatic Scoping Review (2024)
**Summary:** Reviews 21 studies and finds recurring negative themes: privacy concerns, technical problems, fairness issues, and stress.
**Key Finding:** Students don't just care about accuracy — they care about privacy, stress, and fairness.
**Our Implication:** Build privacy-first local architecture as a differentiator.

### 4. A Comprehensive Review of Academic Dishonesty in Automated Proctoring in the AI Era
**Summary:** Covers gaze, head pose, facial recognition, authentication, activity recognition, lockdown features.
**Relevance:** Maps the feature landscape across commercial systems.

---

## Category 2: Behavioral Biometrics (CRITICAL FOR YOUR PROJECT)

### 5. Behavioral Biometrics for Remote Exam Integrity: Continuous Authenticity Assessment via Keystroke Dynamics (2025)
**Summary:** Uses keystroke dynamics for continuous authentication during remote exams. Compares Random Forest, Isolation Forest, and One-Class SVM.
**Key Contribution:** Shows that keystroke behavior can serve as both authentication and anomaly detection.
**Our Use:** Implement keystroke dynamics as a core behavioral feature.

### 6. Keystroke Dynamics: Concepts, Techniques, and Applications (2025)
**Authors:** ACM Computing Surveys
**Summary:** Comprehensive survey of keystroke biometrics — typing patterns, hold times, latency, continuous authentication.
**Our Use:** Theoretical foundation for our keyboard feature module.

### 7. An Intelligent Approach for Fair Assessment of Online Laboratory Examinations Based on Student's Mouse Interaction Behavior
**Summary:** Explores mouse dynamics for cheating detection in online laboratory exams.
**Our Use:** Validates mouse dynamics as a legitimate detection signal.

---

## Category 3: Automated Grading (MUST READ)

### 8. Automated Short Answer Grading: A Systematic Review (2025)
**Summary:** Reviews 110 ASAG studies from 2014–2024. Shows transition from rule/feature-based → deep learning → transformers. Identifies gaps in validity, reliability, multilingual coverage.
**Key Finding:** Don't claim "AI replaces teachers." Instead: "algorithm produces explainable preliminary score for human review."
**Our Approach:** TF-IDF + cosine + keyword coverage (mathematical, explainable, no LLM).

### 9. Automated Essay Scoring: A Siamese Bidirectional LSTM Neural Network Architecture
**Summary:** Shows neural approach to essay scoring using ASAP dataset.
**Why We Don't Use It:** Too complex, requires training data we don't have, black-box.
**Our Approach:** Stay with TF-IDF baseline that's mathematically explainable.

---

## Category 4: Student Performance Prediction (IMPORTANT)

### 10. Machine Learning for Student Performance Prediction: A Systematic Review (2023)
**Summary:** Reviews 162 papers. Most common: Decision Trees, Random Forest, Naive Bayes, ANN, SVM.
**Features:** grades, LMS activity, demographics, historical records.
**Our Use:** Connect exam behavior + question performance + time → future prediction.

### 11. Predicting Student Success: A State-of-the-Art Systematic Review
**Summary:** Identifies regression, Random Forest, SVM, gradient boosting as top methods.
**Key Finding:** Performance factors and evaluation metrics are well-established.

### 12. Predicting Student Performance: A Comprehensive Review of ML, DL, and Explainable AI (2024)
**Summary:** Highlights explainability as an emerging but limited area.
**Our Implication:** Build explainable predictions with feature importance.

---

## Category 5: Multi-Signal & Temporal Analysis

### 13. Hybrid Deep Learning Approaches for Multi-Modal Student Behavior Analysis in E-Learning
**Summary:** Combines vision + interaction + LMS data for behavior understanding.
**Key Finding:** Multi-modal fusion outperforms single-signal approaches.

### 14. Temporal Pattern Mining in Educational Data
**Summary:** Shows how temporal patterns (trends, sequences) improve prediction.
**Our Implication:** Build temporal feature aggregation into the agent.

---

## Category 6: Fairness & Privacy

### 15. Fairness in AI: A Systematic Review of Student Performance Prediction
**Summary:** Analyzes bias in educational ML — demographic parity, equalized odds.
**Key Finding:** Most systems don't report fairness metrics.
**Our Implication:** Run controlled tests under different conditions and report FPR/FNR.

### 16. Privacy-Preserving Educational Data Mining
**Summary:** Techniques for privacy-aware learning.
**Our Implication:** All processing stays local — privacy by design.

---

## Papers Worth Skimming

| Paper | Why |
|-------|-----|
| "Face Liveness Detection" | Spoofing detection concepts |
| "Gaze Estimation via Convolutional Neural Networks" | Eye tracking math |
| "Anomaly Detection in Time Series" | Temporal anomaly fundamentals |
| "Random Forest vs Gradient Boosting" | Model comparison methodology |
| "SHAP Values for Model Interpretability" | Explainability techniques |

---

## Our Research Contribution

After reviewing 50+ papers, our unique contribution is:

1. **Locally executable** — no cloud, no external APIs
2. **Multimodal temporal fusion** — not just single-frame rules
3. **Explainable risk** — shows contributing evidence, not just score
4. **Behavioral biometrics** — keystroke + mouse as continuous signals
5. **Integrated system** — proctoring + grading + performance in one agent
6. **Privacy-first** — all processing on-device

---

## What We Don't Need to Implement

- Custom CNN face detector (MediaPipe/OpenCV is sufficient)
- Speech recognition (audio energy/RMS is enough)
- LLM-based grading (TF-IDF is mathematically explainable)
- Full essay scoring (short-answer TF-IDF is enough)
- Object detection (not core to our scope)
- 360° room surveillance (privacy concerns)

---

## How We Compare

| Aspect | Existing Systems | Our System |
|--------|-----------------|-------------|
| Face detection | ✅ | ✅ (MediaPipe) |
| Head pose | ✅ | ✅ |
| Audio | ✅ | ✅ (energy-based, not speech) |
| Browser monitoring | ✅ | ✅ (tab/window) |
| ML risk prediction | Rare | ✅ (Logistic + Anomaly) |
| Temporal fusion | Rare | ✅ |
| Explainable risk | Rare | ✅ |
| Keystroke dynamics | Rare | ✅ |
| Mouse biometrics | Rare | ✅ |
| Local-only | Rare | ✅ |
| No external APIs | Rare | ✅ |
| Integrated grading | Rare | ✅ |
| Performance prediction | Separate | ✅ |
