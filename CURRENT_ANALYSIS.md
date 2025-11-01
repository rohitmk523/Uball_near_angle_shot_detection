# Current Analysis: Comparison of Old vs New Results
## Video: 09-22(1-NR)

---

## 📊 **Results Summary**

| Metric | OLD | NEW | Change |
|--------|-----|-----|--------|
| **matched_shots_accuracy** | **90.28%** | **87.50%** | ⚠️ **-2.78%** |
| Total Detected | 82 | 82 | - |
| Matched Correct | 65 | 63 | -2 |
| Matched Incorrect | 7 | 9 | +2 |
| Made → Missed | 4 | 2 | ✅ -2 (50% reduction) |
| Missed → Made | 3 | 7 | ❌ +4 (133% increase) |

---

## ✅ **What's Working (Improvements)**

### **Made → Missed Errors: Fixed**
- **Reduced from 4 to 2 (50% reduction)**
- Improvement 1.1 (moderate overlap handling) is working correctly
- The system is now better at recognizing made shots with 50-70% overlap
- Examples that are now correctly classified as made:
  - Shots with moderate overlap but consistent downward movement
  - Shots that previously needed 3+ frames at 100% but can now be detected with moderate overlap

---

## ❌ **What's Broken (Regressions)**

### **Missed → Made Errors: Increased**
- **Increased from 3 to 7 (133% increase)**
- This is causing the overall accuracy decrease

### **Error Analysis:**

#### **Error #1: `steep_entry_clean_swish`**
- **Issue**: Our Improvement 1.2 is TOO LENIENT
- **Details**:
  - Entry angle: 76.4° (>70°, very steep)
  - Overlap: 49.7% (meets 40-50% threshold)
  - Downward consistency: 0.62 (meets >0.6 requirement)
  - **PROBLEM**: `downward_movement = -102px` (NEGATIVE = ball moving UPWARD!)
  - `continues_down = False` (ball is NOT continuing down)
  - `bounces_back = False` (but movement is upward!)

**Root Cause**: The code checks `downward_consistency > 0.6` but doesn't verify that `downward_movement > 0` (actually moving down).

#### **Error #2: `perfect_overlap_layup`**
- **Issue**: High overlap frames but low average overlap = rim hit
- **Details**:
  - Entry angle: 76.7° (>70°)
  - Frames at 100%: 6 (high!)
  - Average overlap: 41.6% (low - rim sitting)
  - Max overlap: 100%
  - **PROBLEM**: `downward_movement = -202px` (NEGATIVE = ball bounced UP!)
  - `downward_consistency = 0.35` (only 35% consistent downward)
  - `upward_consistency = 0.57` (57% upward movement!)
  - `continues_down = False`

**Root Cause**: Decision Factor 1 (6+ frames at 100%) is catching rim hits. The ball sits on rim (high max overlap) but then bounces out (upward movement).

---

## 🔧 **Fixes Needed**

### **Fix #1: Steep Entry Clean Swish - Add Direction Check**

**Current Code** (Improvement 1.2):
```python
elif (entry_angle >= 70 and avg_overlap >= 40 and avg_overlap < 50):
    if downward_consistency > 0.6 and bounce_confidence < 0.3:
        outcome = "made"  # ❌ TOO LENIENT
```

**Fixed Code**:
```python
elif (entry_angle >= 70 and avg_overlap >= 40 and avg_overlap < 50):
    downward_movement = post_hoop_analysis.get('downward_movement', 0)
    if (downward_consistency > 0.6 and 
        bounce_confidence < 0.3 and 
        downward_movement > 0 and  # ✅ CHECK: Actually moving down
        post_hoop_analysis['ball_continues_down']):  # ✅ CHECK: Continues down
        outcome = "made"
```

### **Fix #2: Decision Factor 1 - Add Pattern Check**

**Current Code**:
```python
if frames_with_100_percent >= 6 or (frames_with_100_percent >= 4 and frames_with_95_percent >= 7):
    # Only checks rim bounce
    outcome = "made"  # ❌ Can catch rim hits
```

**Fixed Code**:
```python
if frames_with_100_percent >= 6 or (frames_with_100_percent >= 4 and frames_with_95_percent >= 7):
    # Check for rim hit pattern (high max but low avg = rim sitting)
    if avg_overlap < 50 and post_hoop_analysis.get('downward_movement', 0) <= 0:
        # Rim hit: high max overlap but low avg, and ball moving up
        outcome = "missed"
        outcome_reason = "rim_hit_high_max_low_avg"
    elif steep_entry_bounce_back or is_rim_bounce:
        outcome = "missed"
    else:
        outcome = "made"
```

### **Fix #3: Check Actual Movement Direction Everywhere**

Add checks for `downward_movement > 0` (actually moving down) in:
- Improvement 1.1 (moderate overlap)
- Improvement 1.2 (steep entry clean swish)
- All decision factors that use downward continuation

---

## 📈 **Expected Impact After Fixes**

| Metric | Current | Expected After Fixes |
|--------|---------|---------------------|
| matched_shots_accuracy | 87.50% | 90-92% |
| Missed → Made | 7 | 3-4 |
| Made → Missed | 2 | 1-2 |
| Total Mismatches | 9 | 4-6 |

---

## ✅ **Next Steps**

1. **Fix the three issues identified above**
2. **Re-test on the same video**
3. **Verify accuracy improvement**
4. **Test on additional videos to ensure no regressions**

The improvements are on the right track, but need these validation fixes to prevent false positives (missed shots classified as made).

