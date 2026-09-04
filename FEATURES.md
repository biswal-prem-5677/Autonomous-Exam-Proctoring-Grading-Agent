# FEATURES.md — Feature List

## Core Features

### 1. Exam Management
- Student registration and selection
- Exam configuration (duration, questions, rules)
- Question bank with multiple types
- Real-time timer with auto-submit
- Session management per student

### 2. Proctoring Sensors

#### Vision Engine
- **Face Presence Detection**: Detects if student is at desk
  - Uses MediaPipe face detection (local)
  - Confidence threshold: 0.7
  - Math: N-face count per frame
- **Multiple Face Detection**: Identifies if more than one person present
  - Weight: 0.50 severity
- **Head Pose Estimation**: Yaw, pitch, roll tracking
  - Triggers event when |yaw| > 45°
- **Movement Analysis**: Frame-to-frame displacement tracking

#### Keyboard Sensor
- Keystroke rate calculation
- Idle detection (>120s triggers alert)
- Copy/paste detection (high severity 0.40)

#### Mouse Sensor
- Movement velocity and distance
- Click frequency
- Idle detection

#### Audio Sensor (Optional)
- RMS energy calculation
- Volume in dB
- Zero crossing rate
- Spectral centroid

### 3. Behavioral Feature Engineering

Extracts a 22-dimensional feature vector per time window:
```
X = [face_present, face_confidence, face_count_norm, absence_ratio,
     yaw_abs, pitch_abs, roll_abs,
     rms_energy, volume_db_norm, zcr, volume_threshold, spectral_centroid,
     keyboard_idle_norm, keystroke_rate_norm, idle_keyboard_flag,
     mouse_speed_norm, mouse_distance_norm, click_count_norm, mouse_idle_norm,
     multi_face_events_norm,
     session_progress, total_event_rate]
```

### 4. ML Risk Prediction (Module A)

**Logistic Regression (scratch implementation):**
- Sigmoid: σ(z) = 1 / (1 + e^(-z))
- Binary cross-entropy loss with L2 regularization
- Gradient descent optimization
- Output: P(Cheating | X) ∈ [0, 1]

### 5. Anomaly Detection (Module B)

**Z-score + Mahalanobis distance:**
- Z-score: z_i = (x_i - μ_i) / σ_i
- Mahalanobis: D_M(x) = sqrt((x-μ)^T Σ^(-1) (x-μ))
- Chi-squared threshold for statistical significance
- Output: anomaly_score ∈ [0, 1]

### 6. Automated Grading (Module C)

#### MCQ Grading
- Exact match: score = marks if correct, else 0
- True/False handled same as MCQ

#### Numerical Grading
- Relative tolerance: |answer - correct| / |correct| ≤ ε
- Default tolerance: 2%
- Partial credit for near-misses

#### Short Answer Grading
- **TF-IDF Cosine Similarity**: 60% weight
  - TF(t,d) = count(t,d) / Σ_k count(k,d)
  - IDF(t) = log(N / (1 + DF(t)))
  - TF-IDF(t,d) = TF(t,d) × IDF(t)
  - Cosine: (A·B) / (||A|| × ||B||)
- **Keyword Coverage**: 40% weight
  - Coverage = matched_keywords / total_keywords
- Combined: S = 0.6 × cosine + 0.4 × coverage
- Final score = S × question_marks

### 7. Risk Fusion Engine

Combines three evidence sources:
```
R = 0.40 × base_risk + 0.35 × severity_score + 0.25 × convergence_factor
```
Where convergence_factor = active_signal_categories / 5

With ML refinement (if model available):
```
R = 0.60 × R + 0.40 × logistic_proba
```

### 8. Autonomous Agent (Module E)

**State Machine:**
```
NORMAL → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED
```

**Agent Loop (per time window):**
```
OBSERVE → PERCEIVE → FEATUREIZE → PREDICT → REASON → DECIDE → ACT → REPEAT
```

**Risk Decay:**
```
R_t = α × R_{t-1} + (1-α) × E_t
```
Prevents permanent scoring from one event.

**Multi-Signal Confirmation:**
Requires ≥2 independent signal categories before escalation.

### 9. Evidence Timeline

Each event is recorded with:
- Timestamp
- Event type
- Confidence score
- Source (vision/audio/keyboard/mouse/temporal)
- Decayed contribution to risk

### 10. Grading & Analytics

- Total score and percentage
- Per-question breakdown
- Section/topic analysis
- Correct count

### 11. Performance Prediction (Module D)

**Linear Regression:**
```
Ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```
Features: previous scores, time per question, topic accuracy, difficulty.

**Random Forest Regression** for comparison.

### 12. Reporting

#### Student Report
- Score summary
- Per-question breakdown
- Proctoring summary (risk score + state)
- Recommendations for weak topics

#### Examiner Report
- Total students, mean score, pass rate
- Risk analysis (mean risk, flagged count)
- Flagged students list with reasons
- Full text report

#### Evidence Log
- Timestamped event sequence
- Severity classification
- Source attribution

---

## Feature Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| MCQ Grading | High | Low | P0 |
| Face Detection | High | Medium | P0 |
| Risk Score | High | Medium | P0 |
| Agent State Machine | High | Low | P0 |
| Evidence Log | High | Low | P0 |
| Numerical Grading | High | Low | P0 |
| TF-IDF Grading | Medium | Medium | P1 |
| Anomaly Detection | Medium | Medium | P1 |
| Risk Decay | Medium | Low | P1 |
| Temporal Features | Medium | Medium | P1 |
| Performance Prediction | Medium | Medium | P2 |
| Audio Analysis | Low | Medium | P2 |
| Examiner Dashboard | High | Low | P1 |
| Premium UI | Medium | Medium | P1 |
| Ablation Study | Medium | High | P3 |
