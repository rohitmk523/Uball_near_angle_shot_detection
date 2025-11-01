# Fixes Applied to Improve matched_shots_accuracy

## Summary

Based on comparison of old (90.28%) vs new (87.50%) results, three critical fixes were applied to address regressions caused by the initial improvements.

---

## Problem Identified

**Initial improvements helped with Made→Missed errors but introduced Missed→Made regressions:**

- ✅ Made→Missed: 4 → 2 (50% reduction)
- ❌ Missed→Made: 3 → 7 (133% increase)
- ⚠️ Overall accuracy: 90.28% → 87.50% (regression)

**Root Cause:** The improvements checked `downward_consistency` but not actual movement direction (`downward_movement`). In our coordinate system:
- `downward_movement > 0` = ball moving down (made shot)
- `downward_movement < 0` = ball moving UP (rim bounce)

---

## Fixes Applied

### **Fix #1: Decision Factor 1 - Rim Hit Pattern Detection**

**Location:** Lines 744-769 in `shot_detection.py`

**Problem:** 
- Error case: 6 frames at 100% overlap, but average overlap only 41.6%
- `downward_movement = -202px` (ball moving UP!)
- Ball was sitting on rim (high max overlap) then bouncing out

**Solution:**
```python
if avg_overlap < 50 and downward_movement <= 0:
    # Rim hit: high max overlap but low avg, ball moving up
    outcome = "missed"
    outcome_reason = "rim_hit_high_max_low_avg"
```

**Expected Impact:** Fixes `perfect_overlap_layup` false positives

---

### **Fix #2: Improvement 1.1 - Moderate Overlap Direction Check**

**Location:** Lines 907-926 in `shot_detection.py`

**Problem:**
- Moderate overlap (50-70%) path was too lenient
- Only checked `downward_consistency > 0.7`, not actual direction
- Could accept shots with upward movement if consistency was met

**Solution:**
```python
if (downward_consistency > 0.7 and 
    upward_consistency < 0.3 and 
    downward_movement > 0 and  # Actually moving down
    post_hoop_analysis['ball_continues_down']):  # Continues down flag
    outcome = "made"
```

**Expected Impact:** Prevents false positives in moderate overlap range

---

### **Fix #3: Improvement 1.2 - Steep Entry Clean Swish Direction Check**

**Location:** Lines 928-948 in `shot_detection.py`

**Problem:**
- Most critical issue identified
- Error case: Entry angle 76.4°, Overlap 49.7%, `downward_movement = -102px` (UP!)
- Only checked `downward_consistency > 0.6`, not actual direction
- Classified as made even though ball moved upward

**Solution:**
```python
if (downward_consistency > 0.6 and 
    bounce_confidence < 0.3 and 
    downward_movement > 0 and  # Actually moving down (NOT up!)
    post_hoop_analysis['ball_continues_down']):  # Continues down flag
    outcome = "made"
```

**Expected Impact:** Fixes `steep_entry_clean_swish` false positives

---

## Technical Details

### **Understanding Movement Direction**

In our coordinate system (standard computer vision):
- Y-axis increases downward (0 at top, higher values at bottom)
- `downward_movement = last_y - first_y`
- **Positive value** = ball moved down (made shot indicator)
- **Negative value** = ball moved up (rim bounce indicator)

### **Why This Matters**

Example from actual error:
```
downward_movement = -102px
downward_consistency = 0.62

Interpretation:
- Ball moved 102 pixels UPWARD (negative)
- 62% of frame-to-frame movements were "downward" 
  (but overall trajectory was UP!)
```

**Lesson:** Consistency alone is not enough. We must check:
1. Overall direction (`downward_movement > 0`)
2. Continuation flag (`ball_continues_down = True`)
3. Consistency (`downward_consistency > threshold`)

---

## Expected Results After Fixes

| Metric | Before Fixes | After Fixes (Expected) |
|--------|--------------|------------------------|
| matched_shots_accuracy | 87.50% | 90-92% |
| Missed→Made errors | 7 | 3-4 |
| Made→Missed errors | 2 | 1-2 |
| Total mismatches | 9 | 4-6 |

---

## Testing Instructions

1. **Re-run detection on the same video:**
   ```bash
   python main.py --action video --video_path input/09-22/game1_nearright.mp4
   ```

2. **Compare results:**
   - Check new accuracy_analysis.json
   - Verify matched_shots_accuracy improved
   - Confirm Missed→Made errors reduced

3. **Expected outcomes:**
   - `steep_entry_clean_swish` errors should be eliminated (0 cases)
   - `perfect_overlap_layup` false positives should be caught
   - Overall accuracy should exceed 90%

---

## Changes Made to Repository

**Files Modified:**
- `shot_detection.py` - Applied three fixes for movement direction validation

**Files Added:**
- `COMPARISON_ANALYSIS.md` - Detailed comparison of old vs new results
- `CURRENT_ANALYSIS.md` - Analysis of issues and fix recommendations
- `FIXES_APPLIED.md` - This file

**Commits:**
1. Initial improvements (commit a0295e5)
2. Fixes for regressions (commit 8f47496)

---

## Key Takeaway

**Always validate actual movement direction, not just consistency.**

The fixes ensure that:
- ✅ Ball must actually move down (`downward_movement > 0`)
- ✅ Ball must continue down (`ball_continues_down = True`)
- ✅ Movement must be consistent (`downward_consistency > threshold`)

All three conditions must be met for a made shot classification.

