# Current Shot Detection Settings

## ⚙️ **Active Configuration**

### **Timeouts (Reverted to Original)**
```python
self.shot_sequence_timeout = 3.0  # Main timeout
mini_timeout = 1.0  # When no overlaps detected
```

**Why reverted:** Faster timeouts (1.0s and 0.3s) caused too many shots to be registered.

---

### **Size Validation (Simplified)**
```python
valid_size = 0.2 <= size_ratio <= 1.2  # Very lenient
```

**No position filtering** - removed complex vertical distance checks.

---

### **Decision Logic (Simple)**
```python
# 1. Rim bounce (ball moved up 10+ pixels)
if is_rim_bounce:
    → MISSED

# 2. Standard made (2+ frames at 100%)
elif frames_with_100_percent >= 2:
    → MADE

# 3. Fast swoosh (3+ frames at 95%+)
elif frames_with_95_percent >= 3:
    → MADE

# 4. Everything else
else:
    → MISSED
```

---

## 📊 **Current Behavior**

| Setting | Value | Effect |
|---------|-------|--------|
| **Main timeout** | 3.0s | Groups overlaps into one shot |
| **Mini timeout** | 1.0s | Finalizes after no overlap |
| **Size check** | 0.2-1.2 ratio | Very lenient |
| **Perfect frames** | 2+ at 100% | For MADE |
| **Fast swoosh** | 3+ at 95%+ | For MADE |
| **Rim bounce** | -10px upward | For MISSED |

---

## 🎯 **What This Means**

✅ **Advantages:**
- Fewer false positives (shots grouped properly)
- Standard shot detection works well
- Fast swoosh detection active
- Rim bounce detection active

⚠️ **Trade-offs:**
- Some rim hits may not register (if they leave shooting zone quickly)
- Longer wait before shot is finalized (1-3 seconds)

---

## 🧪 **Testing**

```bash
python main.py --action video --video_path game3_nearleft.mp4 --model runs/detect/basketball_yolo11n3/weights/best.pt
```

**Check:**
- Shots should be registered at reasonable intervals
- No excessive duplicate detections
- Made/missed based on 2+ perfect frames or 3+ fast swoosh

---

## ⚙️ **If You Need Adjustments**

### **Too many shots still:**
```python
# Increase minimum frames requirement
frames_with_100_percent >= 3  # from 2
frames_with_95_percent >= 4   # from 3
```

### **Missing some shots:**
```python
# Reduce frame requirements
frames_with_100_percent >= 1  # from 2
frames_with_95_percent >= 2   # from 3
```

### **Rim hits not counting:**
```python
# Reduce timeouts (but may increase false positives)
self.shot_sequence_timeout = 2.0  # from 3.0
mini_timeout = 0.5  # from 1.0
```

---

## 📝 **Summary**

**Current setup:** 
- Conservative timeouts (3s main, 1s mini)
- Simple size validation (0.2-1.2)
- Clean decision logic (2+ perfect or 3+ swoosh)
- Rim bounce detection enabled

**Best for:** Reducing false positives and duplicate detections.

**Test and adjust frame requirements if needed!** 🎯

