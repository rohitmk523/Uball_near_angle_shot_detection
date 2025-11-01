# Matched Shots Accuracy Improvement Plan

## 📊 **Current Situation Analysis**

Based on mismatch analysis report (lines 60-72):

**Total Outcome Mismatches: 78 cases**
- **Made → Missed: 47 cases** (made shots incorrectly classified as missed)
- **Missed → Made: 31 cases** (missed shots incorrectly classified as made)

**Current matched_shots_accuracy issues:**
- **insufficient_overlap: 24 times** → Made shots with moderate overlap classified as missed
- **perfect_overlap_steep_entry: 22 times** → Missed shots with steep entry and high overlap classified as made
- **perfect_overlap_layup: 9 times** → Missed shots with layup-like overlap classified as made
- **fast_clean_swish: 8 times** → Fast missed shots classified as made
- **steep_entry_bounce_back: 6 times** → Made shots with bounce-back classified as missed
- **perfect_overlap: 6 times** → Missed shots with perfect overlap classified as made

---

## 🎯 **Problem Analysis**

### **Issue 1: Made Shots Classified as Missed (47 cases)**

**Root Causes:**
1. **insufficient_overlap (24 cases)**: Made shots with 30-55% average overlap are rejected
   - Current logic requires 3+ frames at 100% or strong downward continuation
   - Many made shots have moderate overlap but still go in
   - Example: 30.7% overlap, 0 frames at 100%, 88° entry angle → classified as missed

2. **steep_entry_bounce_back (6 cases)**: Steep entry (>70°) shots with bounce-back classified as missed
   - Current logic treats steep entry bounce-back as rim bounce (miss)
   - But some made shots with steep entry can show slight bounce-back
   - Need better distinction between rim bounce vs slight bounce on made shot

3. **Low overlap threshold too strict**: System is too conservative for made shots
   - Requires 6+ frames at 100% OR (4+ at 100% AND 7+ at 95%) for high confidence
   - Many valid made shots don't meet this threshold
   - Especially affects shots with 50-70% average overlap

**Sample Pattern:**
- Overlap: 30-55%
- Frames with 100%: 0-2
- Entry angle: 20-90°
- Classification: MISSED (but actually MADE)

---

### **Issue 2: Missed Shots Classified as Made (31 cases)**

**Root Causes:**
1. **perfect_overlap_steep_entry (22 cases)**: Steep entry angles (>40°) with high overlap classified as made
   - Current logic: `entry_angle >= 40` → classified as made
   - But steep entries can hit rim and bounce out (miss)
   - Example: 66.4% overlap, 3 frames at 100%, 19° entry → classified as made

2. **perfect_overlap_layup (9 cases)**: Layup-like patterns classified as made
   - High overlap (80-100%), multiple perfect frames
   - But ball can still miss (rim out, bounce out)
   - Need better rim bounce detection

3. **fast_clean_swish (8 cases)**: Fast shots with strong downward continuation classified as made
   - Current logic: 2+ frames at 100% + strong downward → made
   - But fast missed shots can also have strong downward movement
   - Need additional validation beyond downward continuation

**Sample Pattern:**
- Overlap: 66-100%
- Frames with 100%: 3-7
- Entry angle: 6-86°
- Rim bounce confidence: 0.0-0.7 (low)
- Classification: MADE (but actually MISSED)

---

## ✅ **Improvement Plan**

### **Phase 1: Fix Made→Missed Errors (47 cases)**

#### **Improvement 1.1: Lower Threshold for Moderate Overlap Made Shots**

**Problem**: Made shots with 30-55% overlap are rejected

**Solution**: Add special handling for moderate overlap with consistent trajectory

**Implementation**:
```
Current: Requires 3+ frames at 100% for made classification
New: If 50-70% avg overlap + consistent downward movement + no bounce-back → MADE
```

**Decision Logic Addition**:
- If `avg_overlap >= 50%` AND `avg_overlap < 70%`:
  - Check for consistent downward movement (downward_consistency > 0.7)
  - Check for no bounce-back (upward_consistency < 0.3)
  - If both true → Classify as MADE with confidence 0.70-0.75
  - Reason: "moderate_overlap_consistent_downward"

**Expected Impact**: Fix 15-20 of 24 "insufficient_overlap" cases

---

#### **Improvement 1.2: Improve Entry Angle Consideration for Made Shots**

**Problem**: Steep entry angles (70-90°) with moderate overlap are too strict

**Solution**: Adjust threshold based on entry angle

**Implementation**:
```
For steep entries (>70°): Lower overlap requirement
- Steep entries naturally have less overlap but can still go in
- Current: Same threshold for all angles
- New: Steep entries need lower overlap threshold
```

**Decision Logic Addition**:
- If `entry_angle >= 70°` AND `avg_overlap >= 40%`:
  - Require: downward_consistency > 0.6 AND no rim bounce
  - If true → Classify as MADE with confidence 0.72-0.80
  - Reason: "steep_entry_clean_swish"

**Expected Impact**: Fix 5-8 of remaining "insufficient_overlap" cases

---

#### **Improvement 1.3: Better Rim Bounce vs Made Shot Bounce Distinction**

**Problem**: Slight bounce-back on made shots classified as rim bounce (miss)

**Solution**: Distinguish between strong rim bounce vs slight deflection

**Implementation**:
```
Current: Any upward movement → rim bounce
New: Require significant upward movement (>20px) AND consistency
```

**Enhanced Rim Bounce Detection**:
- Require `upward_movement > 20px` (was 15px)
- Require `upward_consistency > 0.6` (was 0.5)
- If both not met → Don't classify as rim bounce
- This allows slight deflections on made shots

**Expected Impact**: Fix 3-5 "steep_entry_bounce_back" cases

---

### **Phase 2: Fix Missed→Made Errors (31 cases)**

#### **Improvement 2.1: Enhance Steep Entry Rim Detection**

**Problem**: Steep entry (>40°) with high overlap classified as made, but these hit rim

**Solution**: Improve rim bounce detection for steep entries

**Implementation**:
```
Current: entry_angle >= 40° → classified as made (if 3+ frames at 100%)
New: For steep entries, require additional validation
```

**Decision Logic Modification**:
- If `entry_angle >= 40°` AND `frames_with_100_percent >= 3`:
  - **ALWAYS** check rim bounce confidence first
  - If `rim_bounce_confidence > 0.4` → MISSED
  - If `rim_bounce_confidence < 0.4`:
    - Require strong downward continuation (downward_consistency > 0.8)
    - Require significant downward movement (>30px)
    - If both not met → MISSED (likely rim hit)
    - If both met → MADE (clean steep shot)

**Expected Impact**: Fix 15-18 of 22 "perfect_overlap_steep_entry" cases

---

#### **Improvement 2.2: Improve Post-Hoop Trajectory Analysis for Rim Hits**

**Problem**: Rim hits can have strong downward movement initially, then bounce

**Solution**: Analyze longer post-hoop trajectory window

**Implementation**:
```
Current: Analyzes 2-3 frames after overlap
New: Extend analysis window to 5-7 frames after overlap
```

**Enhanced Post-Hoop Analysis**:
- Track ball position for longer after hoop interaction
- Detect "deceleration + reversal" pattern (rim hit signature)
- If downward movement but then stops/slows → likely rim hit (MISSED)
- If downward movement continues consistently → likely made (MADE)

**Expected Impact**: Fix 5-7 "perfect_overlap_layup" and "perfect_overlap" cases

---

#### **Improvement 2.3: Add Overlap Pattern Analysis**

**Problem**: High overlap can occur on rim hits (ball sits on rim before bouncing)

**Solution**: Analyze overlap pattern, not just max/average

**Implementation**:
```
Current: Uses max_overlap and avg_overlap
New: Analyze overlap pattern for "peak then drop" (rim hit signature)
```

**Pattern Detection**:
- Rim hits often show: Peak overlap → Gradual decrease → Sudden drop
- Made shots often show: Peak overlap → Sustained overlap → Gradual decrease
- If overlap pattern shows sudden drop (>30% in 2 frames) → likely rim hit

**Expected Impact**: Fix 4-6 additional cases

---

#### **Improvement 2.4: Stricter Fast Shot Validation**

**Problem**: Fast shots with 2 frames at 100% + downward continuation classified as made

**Solution**: Require additional validation for fast shots

**Implementation**:
```
Current: 2 frames at 100% + strong downward → MADE
New: For fast shots, require additional factors
```

**Enhanced Fast Shot Logic**:
- If `frames_with_100_percent == 2` (fast shot):
  - Require: `downward_consistency >= 0.9` (was 0.8)
  - Require: `downward_movement >= 40px` (was just checking if >0)
  - Require: `rim_bounce_confidence < 0.2` (very low bounce risk)
  - Require: `entry_angle >= 30°` OR `entry_angle < 70°` (not too steep, not too shallow)
  - If all not met → MISSED (likely rim hit)

**Expected Impact**: Fix 4-6 "fast_clean_swish" cases

---

## 📈 **Expected Impact Summary**

### **Phase 1: Fix Made→Missed (47 cases)**

| Improvement | Cases Fixed | Reason |
|------------|-------------|--------|
| 1.1: Moderate Overlap Handling | 15-20 | insufficient_overlap |
| 1.2: Steep Entry Adjustment | 5-8 | insufficient_overlap |
| 1.3: Bounce Distinction | 3-5 | steep_entry_bounce_back |
| **Total Phase 1** | **23-33** | **~50-70% of Made→Missed errors** |

### **Phase 2: Fix Missed→Made (31 cases)**

| Improvement | Cases Fixed | Reason |
|------------|-------------|--------|
| 2.1: Steep Entry Rim Detection | 15-18 | perfect_overlap_steep_entry |
| 2.2: Extended Post-Hoop Analysis | 5-7 | perfect_overlap_layup, perfect_overlap |
| 2.3: Overlap Pattern Analysis | 4-6 | Various |
| 2.4: Stricter Fast Shot Validation | 4-6 | fast_clean_swish |
| **Total Phase 2** | **28-37** | **~90-100% of Missed→Made errors** |

### **Overall Expected Impact**

**Before:**
- Total Outcome Mismatches: 78
- Made→Missed: 47
- Missed→Made: 31
- **matched_shots_accuracy: ~86%** (assuming 78 mismatches out of ~550 total matches)

**After Phase 1:**
- Remaining Made→Missed: 14-24
- Remaining Missed→Made: 31
- Total Outcome Mismatches: 45-55
- **matched_shots_accuracy: ~90-92%**

**After Phase 2:**
- Remaining Made→Missed: 14-24
- Remaining Missed→Made: 0-3
- Total Outcome Mismatches: 14-27
- **matched_shots_accuracy: ~95-97%**

---

## 🎯 **Priority Implementation Order**

### **High Priority (Implement First)**
1. ✅ **Improvement 2.1**: Steep Entry Rim Detection (fixes 15-18 cases)
2. ✅ **Improvement 1.1**: Moderate Overlap Handling (fixes 15-20 cases)

**Expected**: Fix 30-38 cases (~40-50% of total)

### **Medium Priority (Implement Second)**
3. ✅ **Improvement 1.2**: Steep Entry Adjustment (fixes 5-8 cases)
4. ✅ **Improvement 2.2**: Extended Post-Hoop Analysis (fixes 5-7 cases)
5. ✅ **Improvement 2.4**: Stricter Fast Shot Validation (fixes 4-6 cases)

**Expected**: Additional 14-21 cases fixed

### **Low Priority (Fine-Tuning)**
6. ✅ **Improvement 1.3**: Bounce Distinction (fixes 3-5 cases)
7. ✅ **Improvement 2.3**: Overlap Pattern Analysis (fixes 4-6 cases)

**Expected**: Additional 7-11 cases fixed

---

## 🔧 **Implementation Notes**

### **Where to Modify in shot_detection.py**

1. **`_finalize_shot_sequence()` method** (around line 575):
   - Modify decision logic (lines 630-750)
   - Add new decision branches for moderate overlap
   - Enhance steep entry handling

2. **`_enhanced_rim_bounce_detection()` method** (around line 538):
   - Improve rim bounce confidence calculation
   - Better distinction between rim bounce vs deflection

3. **`_analyze_post_hoop_trajectory()` method** (around line 492):
   - Extend analysis window
   - Add deceleration/reversal pattern detection

4. **New method**: `_analyze_overlap_pattern()`:
   - Detect overlap pattern for rim hit signature
   - Add to decision logic

---

## ✅ **Conclusion**

**These improvements target matched_shots_accuracy specifically:**
- Focus on fixing outcome mismatches (made vs missed classification)
- Address root causes identified in mismatch analysis
- Expected improvement: **86% → 95-97% accuracy**

**All improvements are logical enhancements to existing decision logic:**
- No complex processing required
- Uses existing trajectory and overlap data
- Adjusts thresholds and adds validation checks
- Can be tested incrementally

**Priority: Implement Phase 1 & Phase 2 high-priority items first for maximum impact!**

