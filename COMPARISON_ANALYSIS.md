# Comparison Analysis: Old vs New Results
## Video: 09-22(1-NR)

### Summary

**OLD Results:**
- matched_shots_accuracy: **90.28%**
- Total Detected: 82
- Matched Correct: 65
- Matched Incorrect: 7
  - Made → Missed: 4
  - Missed → Made: 3

**NEW Results (After Improvements):**
- matched_shots_accuracy: **87.50%** ⚠️
- Total Detected: 82
- Matched Correct: 63
- Matched Incorrect: 9
  - Made → Missed: 2 ✅ (Improved by 2)
  - Missed → Made: 7 ❌ (Worsened by 4)

**Overall Impact: -2.78% accuracy decrease**

---

## Analysis

### ✅ **Improvements (Made→Missed Fixed)**
- **Made → Missed errors reduced**: 4 → 2 (50% reduction)
- This confirms our **Improvement 1.1** (moderate overlap handling) is working
- The system is now better at recognizing made shots with moderate overlap

### ❌ **Regressions (Missed→Made Increased)**
- **Missed → Made errors increased**: 3 → 7 (133% increase)
- This is a significant regression

### Root Cause Analysis

**New Missed→Made Errors (7 cases):**
- Most have high overlap and meet made shot criteria
- But they're actually misses (rim hits that look like made shots)

**Issue Identified:**
The improvements made the system MORE lenient for:
1. Steep entry shots (>=40°) - now requires rim bounce check, but may not be strict enough
2. Moderate overlap shots (50-70%) - new path that accepts them as made
3. Fast shots - stricter validation helps, but may miss edge cases

**The problem:** Some of these improvements are being too aggressive and classifying rim hits as made shots.

---

## Specific Issues

### Issue 1: Moderate Overlap Path Too Lenient
**Improvement 1.1** added a new path for 50-70% overlap shots:
```python
elif avg_overlap >= 50 and avg_overlap < 70 and not is_rim_bounce:
    if downward_consistency > 0.7 and upward_consistency < 0.3:
        outcome = "made"
```

**Problem:** This may be catching rim hits that have moderate overlap and downward movement initially, but then bounce out.

### Issue 2: Steep Entry Validation Not Strict Enough
**Improvement 2.1** enhanced steep entry detection, but the logic flow may be allowing too many through:
- Rim bounce confidence check (0.4 threshold) may be too high
- Weak downward continuation check may not be strict enough

### Issue 3: Overlap Pattern Analysis May Not Be Applied Everywhere
The new overlap pattern analysis is only applied in specific decision factors, but may need to be checked earlier in the decision tree.

---

## Recommendations

### Quick Fixes Needed:

1. **Tighten Moderate Overlap Criteria**
   - Increase required downward consistency from 0.7 to 0.8
   - Add additional check: require minimum frames at 85%+ overlap
   - Check overlap pattern for sudden drops

2. **Stricter Steep Entry Validation**
   - Lower rim bounce confidence threshold from 0.4 to 0.3
   - Require stronger downward movement (>40px instead of >30px)
   - Always check overlap pattern for steep entries

3. **Add Overlap Pattern Check Earlier**
   - Check overlap pattern for all shots with high overlap (not just moderate)
   - If sudden drop detected, require additional validation before classifying as made

4. **Review Decision Factor Order**
   - The new moderate overlap path may be catching cases before overlap pattern analysis
   - Consider moving overlap pattern check earlier in decision tree

---

## Next Steps

1. Analyze the 7 new Missed→Made errors in detail to identify common patterns
2. Adjust thresholds based on actual error cases
3. Add additional validation checks for rim hit detection
4. Test on additional videos to ensure changes don't introduce new issues

**Expected Outcome After Fixes:**
- Should maintain the 2 Made→Missed improvements
- Should reduce Missed→Made errors back to 3 or fewer
- Target: 90%+ matched_shots_accuracy

