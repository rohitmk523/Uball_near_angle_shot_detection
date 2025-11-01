# Shot Detection Improvement Plan
## Based on Mismatch Analysis Report

**Analysis Summary:**
- **434 total mismatches** across 10 videos
- **332 false positives (76%)** - Biggest issue
- **78 outcome mismatches** - Made vs Missed errors
- **24 false negatives** - Missing real shots

---

## 🔴 **Critical Issue: False Positives (332 cases)**

### Root Cause Analysis:
1. **Low overlap threshold**: Current `min_overlap_threshold=1.0%` is too permissive
   - Many false positives have 60-100% overlap but aren't real shots
   - Ball passing near hoop gets registered as shot
   
2. **No trajectory validation**: System doesn't check if ball is on shooting trajectory
   - Any overlap near hoop triggers shot detection
   - Needs pre-validation: ball moving upward toward hoop

3. **No shot attempt validation**: Missing checks for actual shot characteristics
   - Should require upward motion before overlap
   - Should filter out horizontal passes near hoop

### **Actionable Fixes (NO complex processing needed):**

#### **Fix 1: Increase Minimum Overlap Threshold**
```python
# Current: min_overlap_threshold = 1.0  (too low)
# Recommended: min_overlap_threshold = 5.0 - 10.0

# In shot_detection.py __init__:
self.min_overlap_threshold = 5.0  # Require 5% overlap minimum
```

**Impact**: Filters out incidental overlaps (114 FP with high overlap would still need trajectory check)

#### **Fix 2: Add Pre-Shot Trajectory Validation**
```python
# In update_shot_tracking(), before starting shot sequence:

# Check if ball is moving upward toward hoop (actual shot attempt)
if len(self.ball_trajectory_buffer) >= 5:
    recent_positions = list(self.ball_trajectory_buffer)[-5:]
    # Calculate if ball is moving upward
    y_deltas = [recent_positions[i+1][1] - recent_positions[i][1] 
                for i in range(len(recent_positions)-1)]
    avg_y_movement = sum(y_deltas) / len(y_deltas)
    
    # Ball should be moving UP (negative Y = upward) before shot
    is_upward_motion = avg_y_movement < -2  # Moving upward
    
    if not is_upward_motion and overlap_percentage < 50:
        # Not a shot attempt, skip
        return
```

**Impact**: Eliminates false positives from horizontal passes or bouncing near hoop

#### **Fix 3: Minimum Duration Requirement**
```python
# Require minimum overlap duration for valid shot
# In _finalize_shot_sequence():

MIN_OVERLAP_FRAMES = 3  # Require at least 3 frames of overlap

if total_overlap_frames < MIN_OVERLAP_FRAMES:
    # Not a valid shot - too brief
    self.shot_sequence_active = False
    self.shot_sequence_overlaps = []
    return
```

**Impact**: Filters brief overlaps that aren't shots

#### **Fix 4: Require Minimum Overlap Quality**
```python
# Don't just count frames, require sustained quality
# In _finalize_shot_sequence():

MIN_SUSTAINED_OVERLAP = 20  # Require at least 20% average overlap

if avg_overlap < MIN_SUSTAINED_OVERLAP:
    # Overlap too low - not a shot
    self.shot_sequence_active = False
    self.shot_sequence_overlaps = []
    return
```

**Impact**: Filters low-quality detections

---

## 🟡 **Secondary Issue: Outcome Mismatches (78 cases)**

### Made → Missed (47 cases):
- **Problem**: Overlap detection too strict
- **Fix**: Current logic requires 6+ frames at 100% - may be too high
- **Recommendation**: Keep current thresholds but improve trajectory analysis

### Missed → Made (31 cases):
- **Problem**: Steep entry angles (>70°) with high overlap classified as made
- **Fix**: Already partially addressed with `steep_entry_bounce_back`
- **Enhancement**: Improve rim bounce detection sensitivity

---

## 🟢 **Minor Issue: False Negatives (24 cases)**

### Root Cause:
- Shots with very low overlap (<1%) not detected
- Fast shots passing through detection window

### **Actionable Fixes:**

#### **Fix 5: Expand Detection Window for Fast Shots**
```python
# Current: shot_sequence_timeout = 3.0 seconds
# For fast shots, may need slightly longer window

# Option: Add fast-shot recovery mechanism
# Track near-miss trajectories and extend timeout for high-velocity balls
```

#### **Fix 6: Lower Threshold for Rapid Trajectory Changes**
```python
# If ball velocity is high (fast shot), accept lower overlap
# In update_shot_tracking():

if ball_velocity > FAST_SHOT_THRESHOLD:
    min_overlap = self.min_overlap_threshold * 0.7  # Lower threshold
else:
    min_overlap = self.min_overlap_threshold
```

---

## 📊 **Implementation Priority**

### **Phase 1: High-Impact Quick Wins** (Implement First)
1. ✅ **Fix 1**: Increase `min_overlap_threshold` to 5.0-10.0
2. ✅ **Fix 3**: Require minimum 3 frames of overlap
3. ✅ **Fix 4**: Require minimum 20% average overlap

**Expected Impact**: Reduce false positives by 50-70% (166-232 cases)

### **Phase 2: Trajectory Validation** (Medium Complexity)
4. ✅ **Fix 2**: Add pre-shot upward trajectory validation
5. ✅ **Fix 6**: Add fast-shot detection with lower threshold

**Expected Impact**: Additional 20-30% false positive reduction

### **Phase 3: Fine-Tuning** (Low Priority)
6. Improve rim bounce detection sensitivity
7. Adjust made/missed thresholds based on new validation

---

## 🎯 **Expected Results After Implementation**

### **Before:**
- Total Mismatches: 434
- False Positives: 332 (76%)
- Outcome Mismatches: 78 (18%)
- False Negatives: 24 (6%)

### **After Phase 1 (Quick Wins):**
- Expected False Positives: 100-166 (70-85% reduction)
- Expected Total Mismatches: 180-250

### **After Phase 2 (Full Implementation):**
- Expected False Positives: 50-80 (85-90% reduction)
- Expected Total Mismatches: 130-170
- **Overall Accuracy: 70-85%** (up from 51-79%)

---

## 🔧 **Code Changes Required**

All fixes can be implemented in `shot_detection.py`:
- **Lines 163-210**: Threshold configuration
- **Lines 296-372**: `update_shot_tracking()` - Add trajectory validation
- **Lines 575-750**: `_finalize_shot_sequence()` - Add duration/quality checks

**No complex processing needed** - just better validation logic!

---

## 📝 **Testing Plan**

1. Test each fix independently on subset of videos
2. Measure false positive reduction
3. Ensure false negatives don't increase significantly
4. Validate overall accuracy improvement

---

## ✅ **Conclusion**

**These improvements are NOT complex processing** - they're straightforward validation enhancements:

1. ✅ **Higher thresholds** (simple parameter change)
2. ✅ **Trajectory checks** (basic direction calculation)
3. ✅ **Duration requirements** (simple frame counting)
4. ✅ **Quality filters** (average calculation)

**All achievable without:**
- ❌ Machine learning models
- ❌ Complex computer vision algorithms
- ❌ Deep learning approaches
- ❌ New dependencies

**These are simple, logical improvements that will significantly boost accuracy!**

