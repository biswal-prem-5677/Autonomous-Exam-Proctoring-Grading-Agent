# EXECUTION_RULES.md — Rules for All Development Sessions

## Purpose

These rules MUST be followed in every development session to ensure consistent progress and completion.

---

## Mandatory Rules

### 1. No External APIs
- ❌ DO NOT add OpenAI, Gemini, Claude, or any LLM API calls
- ❌ DO NOT add cloud vision APIs (Google Vision, AWS Rekognition)
- ❌ DO NOT add cloud speech APIs
- ✅ DO use: NumPy, SciPy, scikit-learn, OpenCV, MediaPipe (local)
- ✅ DO implement algorithms from scratch when possible

### 2. Risk = Evidence, Not Proof
- ❌ DO NOT claim the system "detects cheating"
- ❌ DO NOT make irreversible accusations
- ✅ DO say "risk indicator" or "evidence for review"
- ✅ DO require human review before action

### 3. Mathematics-First
- ✅ DO explain formulas in comments
- ✅ DO use mathematically grounded approaches
- ✅ DO prefer interpretable models (logistic regression) over black boxes

### 4. GitHub Updates
- ✅ DO push changes immediately after any update
- ✅ DO write meaningful commit messages
- ✅ DO verify tests pass before push

---

## Completion Rules

### Before Any Push
- [ ] Run `python main.py test` and verify all tests pass
- [ ] Run `python main.py demo` to verify end-to-end works
- [ ] Update CONTEXT.md if any config/structure changed
- [ ] Commit with descriptive message

### Before Declaring "Done"
- [ ] All tests passing
- [ ] Demo runs without errors
- [ ] Documentation reflects current state
- [ ] GitHub is updated

---

## UI Rules

### Style Guidelines
- ❌ AVOID: Default blue Streamlit colors
- ❌ AVOID: AI-generated looking designs
- ✅ USE: Sophisticated palette — deep purple (#4A0E4E), emerald (#134E4A), charcoal (#1E1E1E)
- ✅ USE: Clean, minimal design
- ✅ USE: Clear typography and spacing

### Accessibility
- ✅ Ensure text is readable
- ✅ Use sufficient color contrast
- ✅ Don't rely solely on color for meaning

---

## Session Workflow

### Start of Session
1. Read CONTEXT.md to understand current state
2. Check TASKS.md for pending items
3. Start with highest priority incomplete item

### During Session
1. Make incremental changes
2. Test frequently
3. Update docs if API changed
4. Push immediately after meaningful progress

### End of Session
1. Run full test suite
2. Verify demo works
3. Push all changes
4. Update CONTEXT.md with session summary

---

## What's In Scope

### Can Add
- New ML algorithms (mathematical, local)
- Additional question types
- Better visualizations
- Performance optimizations
- Bug fixes
- Documentation improvements
- UI enhancements (premium styling)

### Don't Add
- External API integrations
- Cloud services
- New dependencies without discussion
- Over-ambitious features that delay completion

---

## Priority Order

Current priorities (in order):

1. ✅ Premium UI styling (replace blue tones)
2. 🔄 Verify all components work together
3. ⏳ Push to GitHub
4. ⏳ Final testing and polish

---

## Quick Reference Commands

```bash
# Run demo
python main.py demo

# Run tests
python main.py test

# Student UI
streamlit run app/student/exam.py

# Examiner Dashboard
streamlit run app/examiner/dashboard.py

# Git push
git add .
git commit -m "Description"
git push origin main
```

---

## Enforcement

Any session that:
- Adds external APIs
- Makes claims about "detecting cheating"
- Leaves tests failing
- Forgets to push

...must be corrected before moving forward.

---

*Last updated: 2026-09-04*
