# FLOW.md — Data Flow & Control Flow

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXAM SESSION                                │
│  Student logs in → Exam starts → Timer runs → Questions displayed   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
    ┌────────┐           ┌──────────┐          ┌──────────┐
    │ Webcam │           │ Keyboard │          │   Mouse  │
    └───┬────┘           │  Events  │          │  Events  │
        │                └────┬─────┘          └────┬─────┘
        │                     │                      │
        ▼                     ▼                      ▼
   Face Detection       Keystroke Rate        Movement Velocity
   Face Count           Idle Detection        Click Frequency
   Head Pose            Paste Detection       Idle Detection
   Movement             Copy Detection        Right-Click Spam
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │  FEATURE ENGINE  │
                   │  (22-dim vector) │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌───────────┐
        │  Logistic │  │ Anomaly  │  │  Feature  │
        │  Regression│  │ Detector │  │ Temporal  │
        │  (Module A)│  │ (Module B)│  │ Fusion    │
        └────┬──────┘  └────┬─────┘  └─────┬─────┘
             │              │              │
             │    ┌─────────┴─────────┐    │
             │    │                   │    │
             ▼    ▼                   ▼    ▼
        ┌──────────────────────────────────┐
        │      TEMPORAL FUSION ENGINE       │
        │  R_t = α·R_{t-1} + (1-α)·E_t   │
        │  Multi-signal convergence check   │
        └────────────────┬─────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  AGENT DECISION   │
              │   STATE MACHINE   │
              └────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    ┌──────────────┐       ┌────────────────┐
    │  NORMAL      │       │ SUSPICIOUS     │
    │  Continue    │       │ Flag for review│
    └──────────────┘       └────────────────┘
          │                         │
          │    ┌────────────────────┘
          │    ▼
          │  ┌──────────────┐
          │  │  HIGH_RISK   │
          │  │  Escalate    │
          │  └──────────────┘
          │
          └──────────────────┐
                             │
                             ▼
                    ┌────────────────┐
                    │   EXAM ENDS     │
                    └───────┬────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ GRADING ENGINE │
                   └───────┬────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   MCQ    │    │ Numerical│    │ Short    │
    │  Exact   │    │ Tolerance│    │ TF-IDF + │
    │  Match   │    │  ±2%     │    │ Keywords │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  TOTAL SCORE  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ PERFORMANCE  │
                  │  PREDICTION  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ FINAL REPORT │
                  │  (Student +  │
                  │  Examiner)   │
                  └──────────────┘
```

## Agent Control Flow

```
START
 │
 ▼
[OBSERVE] ──── Raw signals from sensors arrive
 │
 ▼
[PERCEIVE] ── Extract structured events from signals
 │              • face_absent / multiple_faces
 │              • head_turned
 │              • loud_audio
 │              • paste_detected / copy_paste
 │              • idle_keyboard / idle_mouse
 │              • tab_switch
 │
 ▼
[FEATUREIZE] ── Build 22-dim feature vector
 │
 ▼
[PREDICT] ──── Run ML models
 │               • LogisticRegression → cheating probability
 │               • AnomalyDetector → anomaly score
 │               • TemporalAggregator → trend
 │
 ▼
[REASON] ───── Fuse all evidence
 │               • EvidenceMemory.add_event()
 │               • RiskEngine.compute_risk()
 │               • ML prediction refinement
 │               • Multi-signal convergence
 │
 ▼
[DECIDE] ───── State transition
 │               NORMAL → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED
 │
 ▼
[ACT] ───────── Record evidence, trigger callbacks
 │               • EvidenceLog entry
 │               • Risk score update
 │               • Alert if HIGH_RISK
 │
 ▼
 CONTINUE? ──── Yes → back to OBSERVE (next time window)
 │               No → generate final report
 ▼
END
```

## Grading Pipeline Flow

```
STUDENT ANSWER
     │
     ▼
┌───────────────────┐
│  Question Type?   │
└─────────┬─────────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
   MCQ  Num   Short
    │     │     │
    ▼     ▼     ▼
 Exact  Error  TF-IDF
 Match  Tol    Cosine
    │     │     +
    │     │    Keyword
    │     │    Coverage
    │     │     │
    └─────┼─────┘
          │
          ▼
   ┌─────────────┐
   │   SCORE      │
   │  Per Q       │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ AGGREGATE    │
   │ Total Score  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ PERCENTAGE   │
   │  (0-100%)   │
   └─────────────┘
```

## Risk Computation Flow

```
EVIDENCE EVENTS
     │
     ▼
┌──────────────────┐
│  Event Weights   │
│  face_absent:0.3 │
│  multiple:0.5    │
│  head_turned:0.2 │ ... (18 event types)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Severity Score   │
│ Σ(weight × min(count,5)) / 2
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Convergence      │
│ Active signals / 5
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Base Risk        │
│ 0.40·base +      │
│ 0.35·severity +  │
│ 0.25·convergence │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ML Refinement    │ (if logistic model attached)
│ 0.60·base +      │
│ 0.40·logistic    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FINAL RISK       │ ∈ [0, 1]
│ (clamped)        │
└──────────────────┘
```

## Temporal Evidence Memory Flow

```
NEW EVENT arrives
     │
     ▼
┌──────────────────┐
│ Apply decay to   │ R_cumulative *= alpha (default 0.8)
│ existing risk    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Add new event    │ R_cumulative += (1-alpha) × confidence
│ score            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prune old events │ Remove entries outside 300s window
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
��� Return current   │ min(R_cumulative, 1.0)
│ risk score       │
└──────────────────┘
```
