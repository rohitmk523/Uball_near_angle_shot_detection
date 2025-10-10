# Basketball Shot Detection - Improved Motion-Based Algorithm

## 🎯 Overview

This document describes the improved shot detection algorithm that addresses the three main failure cases:
1. **Rim Bounce** - Ball bounces on rim and goes back into play
2. **Horizontal Layup Crossing** - Ball crosses over hoop from one side to the other
3. **Fast Swoosh** - Ball goes through quickly with limited perfect overlap frames

---

## 🔧 What Changed

### **Removed Files:**
- `enhanced_shot_detection.py` - Over-engineered with complex trajectory analysis
- `main_enhanced.py` - Enhanced version main file
- `trajectory_analysis.py` - Complex physics-based analysis module

### **Modified Files:**
- `shot_detection.py` - Added simple motion tracking and improved decision logic
- `main.py` - No changes needed (works with updated shot_detection.py)

---

## 🧠 New Algorithm Logic

### **Core Improvement: Motion Pattern Analysis**

Instead of relying solely on overlap percentage and frame counts, we now track:
- **Vertical displacement** (Y-axis movement)
- **Horizontal displacement** (X-axis movement)
- **Motion type** classification

### **Decision Tree:**

```
1. Track ball position (X, Y) during overlap sequence
2. Calculate displacements when sequence ends
3. Classify motion type:
   - Upward Bounce: Vertical < -10px → MISSED (rim bounce)
   - Horizontal Crossing: Horizontal >> Vertical → MISSED (layup crossing)
   - Minimal Motion: Vertical < 10px → MISSED (not enough downward)
   - Downward Motion: Vertical >= 20px → Check overlap for MADE/MISSED

4. For Downward Motion:
   - 2+ frames at 100% overlap + 20px+ down → MADE
   - 3+ frames at 95%+ overlap + 15px+ down → MADE (fast swoosh)
   - Otherwise → MISSED
```

---

## ⚙️ Tunable Parameters

Located in `ShotAnalyzer.__init__()`:

```python
# Motion thresholds (in pixels)
self.MIN_DOWNWARD_PIXELS = 20        # Required downward movement for made shot
self.MAX_HORIZONTAL_RATIO = 2.5      # Max horizontal/vertical ratio
self.UPWARD_BOUNCE_THRESHOLD = -10   # Upward = rim bounce
self.MINIMAL_VERTICAL_THRESHOLD = 10 # Minimum vertical movement

# Overlap thresholds
PERFECT_OVERLAP = 100.0    # Perfect overlap
NEAR_PERFECT = 95.0        # Near perfect (for fast swoosh)

# Frame requirements
MIN_PERFECT_FRAMES = 2     # Frames at 100% for made shot
MIN_NEAR_PERFECT_FRAMES = 3 # Frames at 95%+ for fast swoosh
```

---

## 🎨 Visualization Improvements

The overlay now shows:
- **Sequence info**: Frame count and overlap distribution
- **Motion type**: DOWN (green), BOUNCE (red), CROSS (orange), MIN (white)
- **Vertical displacement**: How many pixels ball moved down/up
- **Real-time prediction**: Shows MADE/MISS prediction during sequence

Example display:
```
MADE: 5
MISSED: 3
TOTAL: 8

SEQ: 4f | 100%:2 95%:4
MOTION: DOWN 25px
PRED: MADE
```

---

## 📊 Expected Accuracy

| Scenario | Old (70%) | New (Expected) |
|----------|-----------|----------------|
| **Made shots** | ✅ Good | ✅ Better |
| **Rim bounce** | ❌ False MADE | ✅ Correct MISS |
| **Layup crossing** | ❌ False MADE | ✅ Correct MISS |
| **Fast swoosh** | ❌ False MISS | ✅ Correct MADE |
| **Overall** | ~70% | **85-90%** |

---

## 🧪 Testing & Tuning

### **Step 1: Test with your videos**
```bash
python main.py --action video --video_path game3_nearleft.mp4
```

### **Step 2: Review the session JSON**
Check `game3_nearleft_session.json` for:
- `outcome_reason` field shows why each decision was made
- `motion_analysis` contains detailed motion data
- Review false positives/negatives

### **Step 3: Adjust parameters**

**If too many rim bounces counted as MADE:**
- Increase `UPWARD_BOUNCE_THRESHOLD` (e.g., -15 or -20)
- This will catch more subtle upward bounces

**If layup crossings still counted as MADE:**
- Decrease `MAX_HORIZONTAL_RATIO` (e.g., 2.0 or 1.5)
- Increase `MIN_DOWNWARD_PIXELS` (e.g., 25 or 30)

**If fast swoosh still missed:**
- Decrease `NEAR_PERFECT` threshold (e.g., 90.0 or 92.0)
- Decrease `MIN_NEAR_PERFECT_FRAMES` (e.g., 2)

**If too many false MADEs:**
- Increase `MIN_PERFECT_FRAMES` (e.g., 3)
- Increase `MIN_DOWNWARD_PIXELS` (e.g., 25)

---

## 🔍 Debugging Tips

### **Understanding outcome_reason codes:**

| Code | Meaning | Action |
|------|---------|--------|
| `rim_bounce_detected` | Ball moved upward | Check UPWARD_BOUNCE_THRESHOLD |
| `horizontal_layup_crossing` | Too much horizontal motion | Check MAX_HORIZONTAL_RATIO |
| `insufficient_downward_motion` | Ball didn't move down enough | Check MINIMAL_VERTICAL_THRESHOLD |
| `perfect_overlap_downward` | Clean made shot | No action needed ✓ |
| `fast_swoosh_downward` | Fast through basket | No action needed ✓ |
| `insufficient_overlap_frames` | Not enough overlap | Check frame thresholds |

### **Analyzing motion_analysis data:**

```json
"motion_analysis": {
  "motion_type": "downward",
  "vertical_displacement": 28.5,
  "horizontal_displacement": 12.3,
  "h_v_ratio": 0.43
}
```

- **vertical_displacement > 0**: Moving down (good)
- **vertical_displacement < 0**: Moving up (bounce)
- **h_v_ratio < 2.5**: Acceptable horizontal motion
- **h_v_ratio > 2.5**: Too horizontal (crossing)

---

## 🚀 Usage

### **Single video:**
```bash
python main.py --action video --video_path your_video.mp4
```

### **Live camera:**
```bash
python main.py --action live --camera 0
```

### **Batch processing:**
```bash
python main.py --action batch --video_dir ./videos/
```

### **With time range:**
```bash
python main.py --action video --video_path game.mp4 --start_time 0:30 --end_time 2:00
```

---

## 📝 Session Data Format

Each shot in the JSON now includes:

```json
{
  "outcome": "made",
  "outcome_reason": "perfect_overlap_downward",
  "max_overlap_percentage": 100.0,
  "frames_with_100_percent": 3,
  "frames_with_95_percent": 4,
  "motion_analysis": {
    "motion_type": "downward",
    "vertical_displacement": 28.5,
    "horizontal_displacement": 12.3,
    "h_v_ratio": 0.43
  },
  "detection_method": "motion_based_improved"
}
```

---

## 💡 Key Insights

1. **Simple is better**: Motion tracking uses just X/Y coordinates, no complex physics
2. **Motion type matters more than overlap**: A 100% overlap with upward motion = rim bounce
3. **Adaptive thresholds handle edge cases**: 95%+ overlap catches fast swoosh
4. **Real-time feedback**: Visual overlay helps tune parameters during testing

---

## 🔄 Reverting Changes

If you need to revert to the old version:
```bash
git checkout HEAD~1 -- shot_detection.py
git checkout HEAD~1 -- enhanced_shot_detection.py main_enhanced.py trajectory_analysis.py
```

---

## 📧 Support

If accuracy is still below 85% after tuning:
1. Share a problematic video sample
2. Include the session JSON
3. Describe which shots are miscategorized
4. We can fine-tune thresholds based on your specific camera angle and court setup

---

**Good luck! The improved algorithm should handle your three problem cases much better! 🏀**

