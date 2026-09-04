# System Features — Technical Specification

## Feature Classification Matrix

| Feature ID | Feature | Category | Priority | Complexity | Dependencies |
|-----------|---------|----------|----------|------------|--------------|
| F-01 | Question Bank Management | Exam Engine | P0 | Low | None |
| F-02 | Exam Session Lifecycle | Exam Engine | P0 | Medium | F-01 |
| F-03 | Adaptive Timer | Exam Engine | P0 | Low | F-02 |
| F-04 | Answer Persistence | Exam Engine | P0 | Low | F-02 |
| F-05 | Face Presence Detection | Vision | P0 | Medium | Sensors |
| F-06 | Multi-Person Detection | Vision | P0 | Medium | F-05 |
| F-07 | Head Pose Estimation | Vision | P1 | Medium | F-05 |
| F-08 | Gaze Deviation Detection | Vision | P1 | High | F-07 |
| F-09 | Facial Movement Tracking | Vision | P1 | Low | F-05 |
| F-10 | RMS Energy Analysis | Audio | P2 | Low | Audio Sensor |
| F-11 | FFT-Based Spectral Analysis | Audio | P2 | Medium | F-10 |
| F-12 | Zero-Crossing Rate | Audio | P2 | Low | F-10 |
| F-13 | Keystroke Timing Analysis | Biometrics | P0 | Medium | Keyboard Sensor |
| F-14 | Typing Rhythm Profiling | Biometrics | P1 | Medium | F-13 |
| F-15 | Keystroke Anomaly Detection | Biometrics | P1 | High | F-13, F-14 |
| F-16 | Mouse Velocity/Acceleration | Biometrics | P0 | Low | Mouse Sensor |
| F-17 | Mouse Trajectory Analysis | Biometrics | P1 | Medium | F-16 |
| F-18 | Click Pattern Analysis | Biometrics | P2 | Low | Mouse Sensor |
| F-19 | Tab/Window Focus Detection | Environment | P0 | Low | Browser API |
| F-20 | Fullscreen Enforcement | Environment | P1 | Low | Browser API |
| F-21 | Copy/Paste Detection | Environment | P1 | Low | Browser API |
| F-22 | 22-Dimensional Feature Extraction | Features | P0 | Medium | F-05 through F-21 |
| F-23 | Z-Score Normalization | Features | P0 | Low | F-22 |
| F-24 | Rolling Window Aggregation | Features | P0 | Medium | F-22 |
| F-25 | Logistic Regression Classifier | ML | P0 | High | F-24 |
| F-26 | Z-Score Anomaly Detection | ML | P0 | Medium | F-24 |
| F-27 | Mahalanobis Distance Anomaly | ML | P1 | High | F-26 |
| F-28 | Temporal Risk Fusion | ML | P0 | Medium | F-25, F-26 |
| F-29 | Risk Decay Mechanism | ML | P0 | Low | F-28 |
| F-30 | Multi-Signal Convergence | ML | P0 | Medium | F-28 |
| F-31 | MCQ Automated Grading | Grading | P0 | Low | None |
| F-32 | Numerical Tolerance Grading | Grading | P0 | Low | None |
| F-33 | TF-IDF Vectorization | Grading | P0 | Medium | Text Processing |
| F-34 | Cosine Similarity Scoring | Grading | P0 | Low | F-33 |
| F-35 | Keyword Coverage Analysis | Grading | P0 | Low | Text Processing |
| F-36 | Concept Matching | Grading | P1 | High | NLP Pipeline |
| F-37 | Formula Pattern Matching | Grading | P1 | Medium | Text Processing |
| F-38 | Partial Credit Scoring | Grading | P0 | Medium | F-31 through F-37 |
| F-39 | Explainable Score Breakdown | Grading | P1 | Low | F-38 |
| F-40 | Agent State Machine | Agent | P0 | Low | None |
| F-41 | Observe-Act Loop | Agent | P0 | Medium | F-40 |
| F-42 | Evidence Memory System | Agent | P0 | Medium | F-40 |
| F-43 | Decision Engine | Agent | P0 | High | F-40 through F-42 |
| F-44 | Explainable Evidence Log | Agent | P0 | Medium | F-42 |
| F-45 | Linear Regression (Performance) | Prediction | P1 | Low | Historical Data |
| F-46 | Random Forest Regression | Prediction | P1 | Medium | F-45 |
| F-47 | Topic-Wise Knowledge Profiling | Prediction | P1 | Medium | F-45 |
| F-48 | At-Risk Student Classification | Prediction | P1 | Medium | F-45, F-47 |
| F-49 | Question Difficulty Calibration | Prediction | P2 | Low | Historical Data |
| F-50 | Student Report Generation | Reports | P0 | Low | F-39, F-43 |
| F-51 | Examiner Dashboard | Reports | P0 | Medium | F-50 |
| F-52 | Evidence Timeline Visualization | Reports | P1 | Low | F-44 |
| F-53 | Risk Score Distribution | Reports | P1 | Low | F-43 |

---

## Module Specifications

### Module A: Exam Management Engine

#### F-01: Question Bank Management

**Purpose:** Centralized storage and retrieval of examination content with type-aware metadata.

**Data Model:**

```yaml
Question:
  question_id: UUID
  exam_id: UUID
  text: str
  type: enum[MCQ, NUMERICAL, SHORT_ANSWER, TRUE_FALSE]
  marks: float
  difficulty: enum[EASY, MEDIUM, HARD]
  topic: str
  options: list[str] | null          # For MCQ/TrueFalse
  correct_answer: str | float | null  # Type-aware
  keywords: list[str]                 # For short-answer grading
  expected_concepts: list[str]        # Concept coverage
  tolerance: float                    # For numerical (default: 0.02)
  model_answer: str                   # Reference answer for TF-IDF
```

**Operations:**
- `add_question(question)` → inserts with validation
- `get_questions_by_exam(exam_id)` → ordered retrieval
- `get_questions_by_topic(topic)` → topic filtering
- `update_difficulty(question_id, actual_correct_ratio)` → adaptive difficulty

#### F-02: Exam Session Lifecycle

**Purpose:** Manages the complete lifecycle of an examination session from creation to archival.

**State Machine:**

```
CREATED → STARTED → IN_PROGRESS → SUBMITTED → EVALUATED → ARCHIVED
                  ↓               ↓
               PAUSED         TIMED_OUT
```

**Session Attributes:**
- Duration tracking with configurable tolerance
- Auto-submit on time expiry
- Answer persistence at configurable intervals
- State persistence for crash recovery

#### F-03: Adaptive Timer

**Implementation:**

```python
def remaining_time(exam) -> timedelta:
    elapsed = datetime.now() - exam.start_time
    remaining = exam.duration - elapsed
    if remaining <= timedelta(0):
        trigger_auto_submit(exam)
    return max(remaining, timedelta(0))
```

**Features:**
- Real-time countdown display
- Configurable pre-expiry warnings (5-min, 1-min)
- Automatic answer save before forced submission

---

### Module B: Computer Vision Engine

#### F-05: Face Presence Detection

**Algorithm:**

```
For each frame f_t at time t:
  faces = mediapipe_face_detector.detect(f_t)
  F_t = {f_1, f_2, ..., f_n}  where n = len(faces)

  if n == 0:
    emit_event(FACE_ABSENT)
    absence_duration += delta_t
  elif n == 1:
    absence_duration = 0
    confidence = faces[0].score
  else:
    emit_event(MULTIPLE_PERSON)
```

**Mathematical Properties:**
- Detection threshold: `confidence > 0.7`
- Absence event threshold: `absence_duration > 10s`
- False positive mitigation: temporal persistence (3 consecutive frames)

#### F-06: Multi-Person Detection

**Risk Model:**

$$P(\text{Cheating} \mid n > 1) \neq 1$$

Instead:

$$R_{\text{face}} = w_1 \cdot \mathbb{I}[n > 1] + w_2 \cdot \frac{\text{duration}}{\text{window}} + w_3 \cdot \text{frequency}$$

Where $w_1, w_2, w_3$ are learned weights.

**Evidence Object:**

```json
{
  "timestamp": "00:42:17",
  "event": "MULTIPLE_PERSON",
  "face_count": 2,
  "duration": 4.8,
  "confidence": 0.91,
  "severity": 0.50,
  "risk_contribution": 22
}
```

#### F-07: Head Pose Estimation

**Mathematical Model:**

Using facial landmarks (MediaPipe 468-point mesh):

$$\vec{L} = P_{\text{left-eye}} - P_{\text{nose}}$$
$$\vec{R} = P_{\text{right-eye}} - P_{\text{nose}}$$

**Horizontal Orientation:**

$$H = \frac{x_{\text{nose}} - x_{\text{face-center}}}{\text{face\_width}}$$

**Vertical Orientation:**

$$V = \frac{y_{\text{nose}} - y_{\text{face-center}}}{\text{face\_height}}$$

**Event Trigger:**

```python
if abs(H) > HEAD_TURN_THRESHOLD:
    looking_side = True
    direction = "left" if H > 0 else "right"

if abs(V) > HEAD_TURN_THRESHOLD:
    looking_up_down = True
    direction = "up" if V > 0 else "down"
```

#### F-08: Gaze Deviation Detection

**Temporal Feature Extraction:**

```python
# Per window W (default: 10 seconds)
LookingAwayRate = N_away / N_frames
Duration_away    = Σ duration_i for each away event
Frequency_away   = N_events / window_duration
```

**Persistence Score:**

$$P_{\text{gaze}} = \alpha_1 \cdot \text{rate} + \alpha_2 \cdot \text{duration\_ratio} + \alpha_3 \cdot \text{frequency}$$

---

### Module C: Behavioral Biometrics Engine

#### F-13: Keystroke Timing Analysis

**Keystroke Feature Vector:**

$$K = [H_{\mu}, H_{\sigma}, L_{\mu}, L_{\sigma}, R_{\text{burst}}, R_{\text{idle}}]$$

Where:
- $H_i = t_{\text{release},i} - t_{\text{press},i}$ (hold time)
- $L_i = t_{\text{press},i+1} - t_{\text{release},i}$ (inter-key latency)
- $R_{\text{burst}} = N_{\text{keys in 500ms window}} / 500\text{ms}$
- $R_{\text{idle}} = \mathbb{I}[\Delta t > 120\text{s}]$

**Statistical Profiling:**

$$\mu_H = \frac{1}{N}\sum_{i=1}^{N} H_i$$
$$\sigma_H = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(H_i - \mu_H)^2}$$

**Anomaly Detection:**

$$z_{H_i} = \frac{H_i - \mu_H}{\sigma_H}$$

If $|z_{H_i}| > 3$, flag as outlier keystroke.

#### F-16: Mouse Dynamics Analysis

**Velocity:**

$$v_t = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\Delta t}$$

**Acceleration:**

$$a_t = \frac{v_t - v_{t-1}}{\Delta t}$$

**Feature Vector:**

$$M = [d_{\text{total}}, v_{\mu}, v_{\sigma}, a_{\mu}, a_{\sigma}, c_{\text{rate}}, t_{\text{idle}}]$$

**Trajectory Classification:**

| Pattern | Indication |
|---------|-----------|
| Smooth, continuous movement | Normal reading |
| Frequent micro-adjustments | Possible screen-reading |
| Rapid, long jumps | Possible switching between screens |
| Idle + sudden burst | Possible external reference lookup |

---

### Module D: ML Risk Prediction Engine

#### F-25: Logistic Regression (Scratch)

**Hypothesis:**

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**Cost Function (with L2 Regularization):**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)}) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

**Gradient Descent:**

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

**Decision Boundary:**

$$P(\text{Cheating}) = \begin{cases} < 0.3 & \text{Low Risk} \\ 0.3 - 0.6 & \text{Medium Risk} \\ > 0.6 & \text{High Risk} \end{cases}$$

#### F-26: Z-Score Anomaly Detection

**Per-Feature Z-Score:**

$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

**Aggregated Anomaly Score:**

$$A = \sqrt{\sum_{i=1}^{n} z_i^2}$$

**Threshold (Chi-Squared, df=n, p=0.95):**

$$T = \chi^2_{n, 0.95}$$

Flag: $A > T$

#### F-27: Mahalanobis Distance Anomaly

**Distance:**

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Where $\Sigma$ is the covariance matrix of normal behavior.

**Threshold:**

$$D_M(x) > T \Rightarrow \text{Anomalous}$$

**Advantage over Z-score:** Accounts for feature correlations.

#### F-28: Temporal Risk Fusion

**Multi-Signal Convergence:**

$$C = I_{\text{vision}} + I_{\text{audio}} + I_{\text{interaction}} + I_{\text{temporal}}$$

Each $I \in \{0, 1\}$ (binary evidence indicator) or $\mathbb{R}$ (continuous intensity).

**Fused Risk:**

$$R_{\text{fused}} = w_1 R_{\text{ML}} + w_2 R_{\text{anomaly}} + w_3 R_{\text{rules}}$$

With $\sum w_i = 1$.

#### F-29: Risk Decay

**Decay Formula:**

$$R_t = \alpha \cdot R_{t-1} + (1 - \alpha) \cdot E_t$$

Default: $\alpha = 0.8$

**Behavior:**
- Continuous suspicious events: Risk accumulates
- Normal behavior continues: Risk gradually decreases
- Complete silence: Risk → 0 over ~50 time windows

#### F-30: Multi-Signal Confirmation

**Escalation Gating:**

```python
def should_escalate(current_risk, evidence_categories, persistence_count):
    if current_risk < ESCALATION_THRESHOLD:
        return False
    if len(evidence_categories) < MIN_CATEGORIES_FOR_ESCALATION:  # >= 2
        return False
    if persistence_count < MIN_PERSISTENCE:  # >= 3 consecutive
        return False
    return True
```

**Categories:** Vision, Audio, Keyboard, Mouse, Browser, Temporal

---

### Module E: Autonomous Agent

#### F-40: State Machine

```
States: NORMAL → SUSPICIOUS → MEDIUM_RISK → HIGH_RISK → REVIEW_REQUIRED
                          ↑__________________________|
```

**Transitions:**

| From | Event | To |
|------|-------|----|
| NORMAL | Risk > 0.3 × persistence | SUSPICIOUS |
| NORMAL | Risk > 0.6 × immediate | MEDIUM_RISK |
| SUSPICIOUS | Risk > 0.5 × persistence | MEDIUM_RISK |
| SUSPICIOUS | Risk < 0.2 × decay | NORMAL |
| MEDIUM_RISK | Risk > 0.7 × persistent | HIGH_RISK |
| MEDIUM_RISK | Risk < 0.3 × decay | SUSPICIOUS |
| HIGH_RISK | Risk > 0.8 × confirmed | REVIEW_REQUIRED |
| HIGH_RISK | Risk < 0.4 × decay | MEDIUM_RISK |

#### F-41: Observe-Act Loop

```
┌─────────────────────────────────────────────────────┐
│                  AGENT LOOP                          │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ OBSERVE  │───▶│ PERCEIVE │───▶│FEATUREIZE│     │
│  └──────────┘    └──────────┘    └────┬─────┘     │
│       ▲                               │            │
│       │    ┌──────────┐    ┌──────────┐│            │
│       └────│   ACT    │◀──│  DECIDE  ││            │
│            └────┬─────┘    └────┬─────┘│            │
│                 │               │      │            │
│            ┌────┴───────────────┘      │            │
│            │   ┌──────────┐   ┌────────▼─────┐      │
│            └──▶│  REASON  │◀──│  PREDICT    │       │
│                └──────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────┘
```

**Per-Iteration Time Budget:** ≤100ms

#### F-42: Evidence Memory System

**Sliding Window Buffer:**

```python
class EvidenceMemory:
    def __init__(self, window_size: int = 300):  # 5 min at 10Hz
        self.buffer = deque(maxlen=window_size)
    
    def add(self, evidence: Evidence):
        self.buffer.append(evidence)
    
    def decay(self, alpha: float = 0.8):
        for e in self.buffer:
            e.decayed_weight *= alpha
```

**Decay Function (per evidence item):**

$$w_i^{(t)} = w_i^{(t-1)} \cdot \alpha^{\Delta t}$$

Where $\Delta t$ is the elapsed time since last update.

#### F-43: Decision Engine

**Decision Matrix:**

| Risk Score | Evidence Count | Persistence | Action |
|-----------|---------------|-------------|--------|
| < 0.3 | Any | Any | CONTINUE |
| 0.3 - 0.5 | 1 | < 3 windows | CONTINUE (elevated monitoring) |
| 0.3 - 0.5 | ≥ 2 | ≥ 3 | FLAG for review |
| 0.5 - 0.7 | Any | Any | FLAG for review |
| > 0.7 | ≥ 2 | ≥ 3 | ESCALATE |
| Any | Critical (no face > 30s, multiple faces) | Immediate | ESCALATE |

---

### Module F: Automated Grading

#### F-31: MCQ Automated Grading

**Exact Match Algorithm:**

```python
def grade_mcq(student_answer, correct_answer, marks):
    return marks if student_answer.strip().lower() == correct_answer.strip().lower() else 0.0
```

**Statistical Properties:**
- Precision: 100% (deterministic)
- Recall: N/A
- False Positive Rate: 0%

#### F-32: Numerical Tolerance Grading

**Relative Error:**

$$\epsilon_{\text{rel}} = \frac{|a_{\text{student}} - a_{\text{correct}}|}{|a_{\text{correct}}|}$$

**Scoring Function:**

$$S = \begin{cases}
m & \epsilon_{\text{rel}} \leq \tau \\
m \cdot \max\left(0, 1 - \frac{\epsilon_{\text{rel}} - \tau}{1 - \tau}\right) & \tau < \epsilon_{\text{rel}} < 1 \\
0 & \epsilon_{\text{rel}} \geq 1
\end{cases}$$

Where $\tau$ is the tolerance threshold (default: 0.02).

#### F-33: TF-IDF Vectorization

**Term Frequency:**

$$TF(t, d) = \frac{f_{t,d}}{\sum_{k \in d} f_{k,d}}$$

**Inverse Document Frequency:**

$$IDF(t) = \log\left(\frac{N}{1 + |\{d \in D : t \in d\}|}\right)$$

**TF-IDF Weight:**

$$w_{t,d} = TF(t, d) \times IDF(t)$$

**Implementation Details:**
- Stop-word removal (NLTK corpus)
- Porter stemming
- n-gram range: (1, 2)
- Sublinear TF scaling: $1 + \log(TF)$

#### F-34: Cosine Similarity Scoring

$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\|_2 \cdot \|B\|_2} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Score Mapping:**

| Similarity | Interpretation | Grade |
|-----------|---------------|-------|
| ≥ 0.85 | Near-identical | Full marks |
| 0.70 - 0.85 | Very similar | 80-95% |
| 0.55 - 0.70 | Moderately similar | 55-75% |
| 0.40 - 0.55 | Weak similarity | 40-55% |
| 0.25 - 0.40 | Vague relation | 20-40% |
| < 0.25 | Unrelated | 0-20% |

#### F-35: Keyword Coverage Analysis

**Coverage Metric:**

$$\text{Coverage} = \frac{|\{k \in K_{\text{expected}}\} \cap \{k \in K_{\text{student}}\}|}{|K_{\text{expected}}|}$$

**Semantic Expansion:**

For each keyword, expand with synonyms:
$$K_{\text{expanded}} = \bigcup_{k \in K} \{k\} \cup \text{Synonyms}(k)$$

#### F-38: Partial Credit Scoring

**Weighted Combination:**

$$S_{\text{final}} = w_1 S_{\text{tfidf}} + w_2 S_{\text{coverage}} + w_3 S_{\text{concept}} + w_4 S_{\text{structure}}$$

Default weights: $w_1=0.40, w_2=0.25, w_3=0.25, w_4=0.10$

**Final Grade:**

$$\text{Grade} = S_{\text{final}} \times M$$

Where $M$ is the maximum marks for the question.

#### F-39: Explainable Score Breakdown

**Report Format:**

```
Question 7: "Explain Ohm's Law"
─────────────────────────────────────────────────────
Component              Score    Weight    Contribution
─────────────────────────────────────────────────────
Semantic Similarity    0.72    40%       0.288
Keyword Coverage       0.75    25%       0.188
Concept Match          0.80    25%       0.200
Structure Quality      0.65    10%       0.065
────────────────────────────────────���────────────────
Combined Score         0.74    100%
Final Grade            7.4/10
─────────────────────────────────────────────────────
```

---

### Module G: Student Performance Intelligence

#### F-45: Linear Regression Performance Model

**Model:**

$$\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i$$

**Features:**

$$X = [S_{\text{prev}}, T_{\text{avg}}, A_{\text{topic}}, D_{\text{avg}}, R_{\text{exam}}]$$

Where:
- $S_{\text{prev}}$ = previous exam scores
- $T_{\text{avg}}$ = average time per question
- $A_{\text{topic}}$ = topic-wise accuracy
- $D_{\text{avg}}$ = average question difficulty
- $R_{\text{exam}}$ = proctoring risk score (optional)

#### F-47: Topic-Wise Knowledge Profiling

**Knowledge Vector:**

$$K_s = [k_{\text{topic}_1}, k_{\text{topic}_2}, ..., k_{\text{topic}_m}]$$

**Per-Topic Score:**

$$k_t = \frac{\sum_{i \in Q_t} w_i \cdot \mathbb{I}[\text{correct}_i]}{\sum_{i \in Q_t} w_i}$$

Where $w_i$ is the marks weight of question $i$.

**Classification:**

| Accuracy | Classification |
|----------|---------------|
| ≥ 80% | Strong |
| 60% - 80% | Moderate |
| 40% - 60% | Developing |
| < 40% | Weak |

#### F-49: Question Difficulty Calibration

**Difficulty Index:**

$$D_j = 1 - \frac{C_j}{N_j}$$

Where:
- $C_j$ = number of students who answered question $j$ correctly
- $N_j$ = total students who attempted question $j$

**Discrimination Index:**

$$DISC_j = \frac{U_j - L_j}{N/2}$$

Where $U_j$ = upper group correct, $L_j$ = lower group correct.

---

### Module H: Reporting & Evidence

#### F-50: Student Report

**Report Structure:**

```
╔═══════════════════════════════════════════════════════╗
║              EXAMINATION INTELLIGENCE REPORT           ║
╠═══════════════════════════════════════════════════════╣
║ Student: STU001  |  Exam: Final Mathematics  |  2026  ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  ACADEMIC PERFORMANCE                                 ║
║  ─────────────────────                                ║
║  Total Score:        78 / 100                          ║
║  Percentage:         78.0%                             ║
║  Correct:            16 / 20                           ║
║  Time Efficiency:    92%                               ║
║                                                       ║
║  TOPIC ANALYSIS                                       ║
║  ──────────────                                       ║
║  Algebra:          ████████████░░  87%  [Strong]      ║
║  Geometry:         ███████████░░░  83%  [Strong]      ║
║  Calculus:         ████████░░░░░░  66%  [Moderate]    ║
║  Probability:      ██████░░░░░░░░  51%  [Developing]  ║
║  Statistics:       █████░░░░░░░░░  43%  [Weak]        ║
║                                                       ║
║  PROCTORING SUMMARY                                    ║
║  ──────────────────                                   ║
║  Risk Score:       18%  ████░░░░░░  LOW RISK         ║
║  State:            NORMAL                              ║
║  Events:          3                                    ║
║  Confidence:       92%                                  ║
║                                                       ║
║  PREDICTED PERFORMANCE                                ║
║  ────────────────────                                 ║
║  Next Exam Est.:   83 / 100                            ║
║  Recommended Focus: Probability, Statistics            ║
╚═══════════════════════════════════════════════════════╝
```

#### F-51: Examiner Dashboard

**Dashboard Panels:**

1. **Session Overview** — Active students, exam status, time remaining
2. **Risk Distribution** — Histogram of student risk scores
3. **Flagged Students Table** — Sorted by risk, with evidence summary
4. **Event Timeline** — Time-series of events across all students
5. **Grade Distribution** — Score histogram with mean/std
6. **Topic Analysis** — Class-level topic performance heatmap

---

## Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| Signal processing latency | < 100ms | Per batch |
| Feature extraction | < 50ms | Per window |
| ML inference | < 10ms | Per prediction |
| Risk calculation | < 5ms | Per update |
| Report generation | < 2s | Per student |
| Dashboard refresh | < 1s | Per poll |
| Memory usage | < 500MB | Peak during exam |
| Test coverage | ≥ 85% | Line coverage |

## Scalability Constraints

| Constraint | Limit | Mitigation |
|-----------|-------|------------|
| Concurrent students | 1 per machine | Multi-machine deployment |
| Session duration | ≤ 4 hours | Auto-archive |
| Evidence buffer | 10,000 events per session | Ring buffer |
| Feature vector size | 22 dimensions (fixed) | No expansion |
| Model training data | ≤ 10,000 samples | Incremental learning |
