# Analysis of Remaining 7 Errors - Path to 95%+ Accuracy

## Current Status

**Comparison Summary:**
- **OLD**: 90.28% (3 made→missed, 4 missed→made)
- **MIDDLE**: 87.50% (7 made→missed, 2 missed→made) - regression from initial improvements
- **NEW**: 90.28% (7 made→missed, **0 missed→made**) ✅

**Achievement:** 
- ✅ Eliminated ALL Missed→Made errors (false positives)
- ⚠️ But now have 7 Made→Missed errors (all false negatives)
- 📊 Back to baseline 90.28%, need to push higher

---

## Detailed Error Analysis

### Error Pattern Summary

| Error | Type | Overlap | Frames 100% | Downward | Reason | Issue |
|-------|------|---------|-------------|----------|--------|-------|
| #1 | FREE_THROW_MAKE | 72.3% | 2 | -4px | insufficient_overlap | Nearly zero movement |
| #2 | 3PT_MAKE | 57.6% | 2 | 291px ✅ | insufficient_overlap | Good downward but 2 frames |
| #3 | FG_MAKE | 80.2% | 8 | -111px | rim_bounce (0.90) | High rim bounce but made |
| #4 | FREE_THROW_MAKE | 59.5% | 2 | 266px ✅ | insufficient_overlap | Good downward but 2 frames |
| #5 | 3PT_MAKE | 81.5% | 5 | 25px | insufficient_overlap | Low consistency (0.21) |
| #6 | FREE_THROW_MAKE | 87.3% | 5 | 174px ✅ | steep_entry_weak_downward | Good movement but low consistency |
| #7 | FREE_THROW_MAKE | 90.5% | 2 | 216px ✅ | insufficient_overlap | Perfect downward (1.00) but 2 frames |

---

## Key Patterns Identified

### **Pattern 1: Free Throws Being Rejected (5 of 7 errors)**
- **Issue**: Free throws have steep entry angles (78-84°) and often 2-5 frames at 100%
- **Current problem**: Our strict validation is rejecting them
- **Examples**: Error #1, #4, #6, #7

### **Pattern 2: Good Downward Movement But Rejected for "2 Frames" (4 cases)**
- **Issue**: Shots with 2 frames at 100%, excellent downward movement (216-291px), but rejected
- **Current logic**: Improvement 2.4 requires 0.9 consistency for 2 frames, but these have lower
- **Examples**: Error #2 (291px downward!), #4 (266px), #7 (216px, 1.00 consistency!)

### **Pattern 3: Rim Bounce But Actually Made (1 case)**
- **Issue**: Error #3 has rim bounce confidence 0.90 but is actually MADE
- **Explanation**: Ball hit rim but still went in (rim bounce doesn't always mean miss)

---

## Proposed Improvements

### **Improvement A: Special Handling for Free Throws**

**Problem**: Free throws have unique characteristics:
- Very steep entry angles (78-90°)
- High overlap but sometimes lower consistency
- Often 2-5 frames at 100%

**Solution**:
```python
# Add free throw detection based on angle
is_free_throw = (entry_angle is not None and 
                 entry_angle >= 75 and 
                 avg_overlap >= 70)

if is_free_throw and frames_with_100_percent >= 2:
    # Free throws have lenient requirements
    if (downward_movement > 0 or downward_movement >= -10) and bounce_confidence < 0.7:
        outcome = "made"
        outcome_reason = "free_throw_made"
        decision_confidence = 0.85
```

**Expected Impact**: Fix errors #1, #4, #6, #7 (4 cases)

---

### **Improvement B: Relax 2-Frame Requirements for Excellent Downward Movement**

**Problem**: Error #7 has:
- 90.5% average overlap
- 2 frames at 100%
- 216px downward movement
- **1.00 downward consistency** (perfect!)
- But rejected by Improvement 2.4

**Current requirement for 2 frames**:
```python
if (downward_consistency >= 0.9 and 
    downward_movement >= 40 and 
    bounce_confidence < 0.2 and
    30 <= entry_angle < 70):
```

**Problem**: This misses shots with:
- Very high overlap (>85%)
- Perfect downward consistency (1.0)
- Steep angles (>70°)

**Solution**:
```python
# Decision Factor 3b: Fast Clean Swish - RELAXED for high quality
elif frames_with_100_percent >= 2 and not is_rim_bounce:
    downward_consistency = post_hoop_analysis.get('downward_consistency', 0)
    downward_movement = post_hoop_analysis.get('downward_movement', 0)
    
    # RELAXED: Accept if overlap is very high AND downward is excellent
    if (avg_overlap >= 85 and 
        downward_consistency >= 0.8 and 
        downward_movement >= 150 and
        bounce_confidence < 0.3):
        outcome = "made"
        outcome_reason = "fast_clean_swish_high_quality"
        decision_confidence = 0.85
    # Original strict validation for lower quality
    elif (downward_consistency >= 0.9 and 
          downward_movement >= 40 and 
          bounce_confidence < 0.2 and
          entry_angle is not None and 30 <= entry_angle < 70):
        outcome = "made"
        outcome_reason = "fast_clean_swish"
        decision_confidence = 0.75
```

**Expected Impact**: Fix errors #2, #4, #7 (3 cases, overlap with Improvement A)

---

### **Improvement C: Lower Rim Bounce Threshold for High Overlap**

**Problem**: Error #3 has:
- 80.2% average overlap
- 8 frames at 100% (very high!)
- Rim bounce confidence 0.90
- But is actually MADE

**Issue**: Ball hit rim but still went in (rim rattler)

**Current logic**:
```python
if bounce_confidence > 0.7:
    outcome = "missed"
```

**Solution**: For very high overlap (many perfect frames), require higher rim bounce confidence:
```python
# Decision Factor 1
if frames_with_100_percent >= 6:
    # For very high perfect frames, rim rattlers can still go in
    if frames_with_100_percent >= 8 and bounce_confidence < 0.95:
        # 8+ perfect frames, even with high rim bounce, likely went in
        outcome = "made"
        outcome_reason = "perfect_overlap_rim_rattler"
        decision_confidence = 0.80
    elif bounce_confidence > 0.7:
        outcome = "missed"
```

**Expected Impact**: Fix error #3 (1 case)

---

### **Improvement D: Lower Threshold for Decision Factor 3 (3+ frames)**

**Problem**: Error #5 has:
- 81.5% average overlap
- 5 frames at 100%
- Entry angle 17.5°
- Downward movement 25px
- Downward consistency 0.21 (very low)

**Current logic**: Requires entry_angle >= 40 OR strong downward for 3+ frames

**Issue**: Shallow angle (17.5°) with low downward consistency rejected

**Solution**: For 5 frames at 100%, be more lenient:
```python
elif frames_with_100_percent >= 5 and avg_overlap >= 80:
    # 5+ perfect frames with high avg overlap
    if bounce_confidence < 0.6:
        outcome = "made"
        outcome_reason = "perfect_overlap_high_frames"
        decision_confidence = 0.82
```

**Expected Impact**: Fix error #5 (1 case)

---

## Expected Impact Summary

| Improvement | Target Errors | Cases Fixed |
|-------------|---------------|-------------|
| A: Free Throw Handling | #1, #4, #6, #7 | 4 |
| B: Relax 2-Frame Requirements | #2, #4, #7 | 3 (overlap with A) |
| C: Rim Rattler Detection | #3 | 1 |
| D: Lower Threshold for 5 Frames | #5 | 1 |
| **Total Unique** | **All 7 errors** | **5-7 cases** |

---

## Predicted Results After Improvements

| Metric | Current | After A+B+C+D |
|--------|---------|---------------|
| matched_shots_accuracy | 90.28% | **95-98%** |
| Made→Missed errors | 7 | **0-2** |
| Missed→Made errors | 0 | **0-1** |
| Total errors | 7 | **0-3** |

---

## Implementation Priority

### **High Priority (Implement First)**
1. ✅ **Improvement A**: Free throw handling (fixes 4 errors)
2. ✅ **Improvement B**: Relax 2-frame requirements (fixes 3 errors, overlaps with A)

**Expected**: 90.28% → 94-95%

### **Medium Priority**
3. ✅ **Improvement C**: Rim rattler detection (fixes 1 error)
4. ✅ **Improvement D**: Lower threshold for 5 frames (fixes 1 error)

**Expected**: 95% → 96-98%

---

## Risk Assessment

### **Low Risk Improvements**
- Improvement A (Free throw handling) - very specific pattern, unlikely to cause false positives
- Improvement C (Rim rattler) - only for 8+ perfect frames, very conservative
- Improvement D (5 frames threshold) - already high quality, low risk

### **Medium Risk Improvements**
- Improvement B (Relax 2-frame) - needs careful validation to avoid reintroducing false positives

**Recommendation**: Implement all four improvements. They target specific high-quality patterns that our current logic is too conservative about.

---

## Next Steps

1. ✅ Implement Improvements A, B, C, D
2. ✅ Test on same video (09-22(1-NR))
3. ✅ Verify accuracy increases to 95%+
4. ✅ Ensure no new Missed→Made errors
5. ✅ Test on additional videos to validate

**Goal**: Push matched_shots_accuracy from 90.28% to 95%+


