# HOW_TO_USE.md — Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/biswal-prem-5677/Autonomous-Exam-Proctoring-Grading-Agent.git
cd Autonomous-Exam-Proctoring-Grading-Agent

# Create virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: CLI Demo (No UI)

```bash
# Run the full demo with sample data
python main.py demo

# Save reports to JSON file
python main.py demo --output reports/demo_results.json

# Grade answers from a JSON file
python main.py grade answers.json

# Run test suite
python main.py test
```

### Option 2: Student Exam Interface

```bash
streamlit run app/student/exam.py
```

This opens the student exam portal in your browser. Steps:
1. Select your student identity from the dropdown
2. Set exam duration (10-120 minutes)
3. Click "Start Exam"
4. Answer questions (MCQ, numerical, or text)
5. Navigate using Previous/Next buttons
6. Submit when done

### Option 3: Examiner Dashboard

```bash
streamlit run app/examiner/dashboard.py
```

This opens the examiner monitoring dashboard. View:
- Score distribution across students
- Risk analysis summary
- Per-student details
- Flagged students requiring review

---

## Creating Custom Exams

### Step 1: Define Questions

Edit `data/sample_data.py` or create your own JSON:

```json
[
  {
    "id": "q1",
    "type": "mcq",
    "text": "What is 2 + 2?",
    "options": ["3", "4", "5", "6"],
    "correct_answer": "4",
    "marks": 1.0,
    "keywords": [],
    "metadata": {"topic": "mathematics"}
  },
  {
    "id": "q2",
    "type": "numerical",
    "text": "Calculate the area of a circle with radius 5",
    "options": null,
    "correct_answer": 78.54,
    "marks": 2.0,
    "tolerance": 0.02,
    "metadata": {"topic": "geometry"}
  },
  {
    "id": "q3",
    "type": "short_answer",
    "text": "Explain the concept of recursion",
    "options": null,
    "correct_answer": "A function that calls itself",
    "marks": 3.0,
    "keywords": ["function", "calls itself", "base case"],
    "metadata": {"topic": "programming"}
  }
]
```

### Step 2: Define Student Answers

```json
{
  "student_001": {
    "student_id": "student_001",
    "name": "John Doe",
    "answers": {
      "q1": "4",
      "q2": 78.5,
      "q3": "A function that calls itself to solve problems"
    }
  }
}
```

### Step 3: Run Grading

```python
from src.models.models import Question, QuestionType
from src.grading.scorer import GradingOrchestrator

# Load questions
questions = [
    Question(id=q["id"], type=QuestionType(q["type"]), text=q["text"],
             options=q.get("options"), correct_answer=q.get("correct_answer"),
             marks=q.get("marks", 1.0), keywords=q.get("keywords", []),
             tolerance=q.get("tolerance", 0.02))
    for q in questions_data
]

# Grade
orchestrator = GradingOrchestrator()
reference = {q.id: q.correct_answer for q in questions}
results = orchestrator.grade_all(questions, student_answers, reference)

print(f"Score: {results['total_score']:.1f} / {results['total_max']:.1f}")
print(f"Percentage: {results['percentage']:.1f}%")
```

---

## Proctoring Setup

### Webcam Monitoring

The system uses MediaPipe for local face detection. No API keys needed.

```python
from src.sensors.webcam import WebcamSensor

sensor = WebcamSensor(camera_index=0, confidence_threshold=0.7)
sensor.start()

# Capture frames
frame = sensor.get_frame()
faces = sensor.detect_faces(frame)
print(f"Faces detected: {len(faces)}")

sensor.stop()
```

### Keyboard & Mouse Tracking

```python
from src.sensors.keyboard import KeyboardSensor
from src.sensors.mouse import MouseSensor

kb = KeyboardSensor()
mouse = MouseSensor()

# Update with new events
kb.update(events_list)
mouse.update(events_list)

# Get statistics
print(f"Keystrokes/min: {kb.get_events_per_minute()}")
print(f"Idle seconds: {kb.get_idle_seconds()}")
print(f"Paste detected: {kb.was_paste_detected()}")
```

### Running Full Proctoring Agent

```python
from src.agent.controller import ProctoringAgent
from src.agent.memory import EvidenceMemory
from src.ml.logistic import LogisticRegressionScratch

# Initialize agent
agent = ProctoringAgent(config={
    "risk": {
        "escalation_threshold": 0.7,
        "review_threshold": 0.5,
        "multi_signal_required": 2,
    },
    "temporal_window": 30,
})

# Process signals
signals = {
    "face_present": True,
    "face_confidence": 0.9,
    "face_count": 1,
    "keyboard_idle_seconds": 5,
    "mouse_idle_seconds": 3,
    "tab_switch": False,
    "paste_detected": False,
    "audio_features": {"volume_db": -60, "rms_energy": 0.1},
    "head_pose": {"yaw": 5, "pitch": 2, "roll": 1},
}

result = agent.process_signals(signals)
print(f"Risk: {result['risk_score']:.4f}")
print(f"State: {result['state']}")
print(f"Events: {result['events_detected']}")

# Check agent status
status = agent.get_status()
print(status)

# Reset for next student
agent.reset()
```

---

## Training ML Models

### Logistic Regression

```python
import numpy as np
from src.ml.logistic import LogisticRegressionScratch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare data: X = features, y = labels (0=normal, 1=suspicious)
X = np.array([...])  # shape: (n_samples, n_features)
y = np.array([...])  # shape: (n_samples,)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")

# Predict
probabilities = model.predict_proba(X_new)
predictions = model.predict(X_new)
```

### Anomaly Detection

```python
from src.ml.anomaly import AnomalyDetector

# Fit on normal behavior data
normal_data = np.array([...])  # shape: (n_normal_samples, n_features)
detector = AnomalyDetector(zscore_threshold=2.5)
detector.fit(normal_data)

# Check new samples
is_anomaly, details = detector.is_anomaly(new_sample, method="combined")
score = detector.score(new_sample)
```

---

## Configuration

Edit `config/default.yaml` to customize:

```yaml
risk:
  alpha_decay: 0.8          # Risk memory (higher = slower decay)
  escalation_threshold: 0.7  # When to escalate
  review_threshold: 0.5      # When to flag for review

grading:
  numerical_tolerance: 0.02  # 2% tolerance for numerical answers
  tfidf_min_similarity: 0.4  # Minimum TF-IDF similarity

ml:
  logistic:
    learning_rate: 0.01
    max_iterations: 1000
```

---

## API Reference

See individual module files for detailed API documentation.

### Key Classes

| Class | Purpose |
|-------|---------|
| `ProctoringAgent` | Main agent controller |
| `RiskEngine` | Evidence fusion and risk scoring |
| `EvidenceMemory` | Temporal event storage with decay |
| `AgentStateMachine` | State transitions |
| `GradingOrchestrator` | Routes questions to graders |
| `LogisticRegressionScratch` | Self-implemented logistic regression |
| `AnomalyDetector` | Z-score + Mahalanobis detection |
| `TFIDFScorer` | TF-IDF vectorization and cosine similarity |
| `TemporalFeatureAggregator` | Rolling window aggregation |
| `BehavioralFeatureExtractor` | Signal-to-vector conversion |

---

## Troubleshooting

**Camera not working?**
- Ensure no other app is using the webcam
- Try changing `camera_index` in WebcamSensor(0) → WebcamSensor(1)

**Tests failing?**
```bash
pip install pytest
python main.py test
```

**Streamlit not found?**
```bash
pip install streamlit
```

**MediaPipe import error?**
- The system gracefully falls back to Haar cascades if MediaPipe is unavailable
- Install: `pip install mediapipe`

---

## Project Structure Reference

```
autonomous-exam-agent/
├── main.py                 # CLI entry point
├── app/
│   ├── student/exam.py     # Student UI
│   └── examiner/dashboard.py # Examiner UI
├── src/
│   ├── agent/              # Autonomous agent (state, risk, memory, controller)
│   ├── audio/              # Audio processing
│   ├── exam/               # Exam engine, questions, session
│   ├── features/           # Feature extraction and temporal aggregation
│   ├── grading/            # MCQ, numerical, TF-IDF, similarity
│   ├── ml/                 # Logistic regression, anomaly, regression
│   ├── models/             # Data models (Question, ExamSession)
│   ├── prediction/         # Performance, difficulty, knowledge
│   ├── reports/            # Evidence log, student/examiner reports
│   ├── sensors/            # Webcam, keyboard, mouse
│   ├── utils/              # Configuration utilities
│   └── vision/             # Face, landmarks, head pose, movement
├── data/
│   └── sample_data.py      # Demo data
├── config/
│   └── default.yaml        # Configuration
├── tests/
│   └── test_agent.py       # Test suite
└── requirements.txt        # Dependencies
```
