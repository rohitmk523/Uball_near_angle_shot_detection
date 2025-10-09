# Enhanced Basketball Shot Detection

This enhanced version implements advanced trajectory analysis and multi-frame context analysis to improve shot detection accuracy from ~70% to 85-90%+.

## New Features

### 🎯 **Trajectory Analysis**
- **Velocity Analysis**: Tracks ball speed and direction changes
- **Direction Consistency**: Measures trajectory smoothness (0-1 score)
- **Upward Bounce Detection**: Identifies rim bounces that falsely appear as made shots
- **Clean Downward Motion**: Detects fast swooshes that may be missed

### 🔍 **Multi-Frame Context Analysis**
- **Approach Analysis**: Examines ball movement before overlap
- **Shot Sequence Analysis**: Analyzes trajectory through hoop area
- **Exit Analysis**: Tracks ball behavior after overlap
- **Pattern Recognition**: Identifies shot patterns (clean_made_shot, rim_bounce, fast_swoosh, clear_miss)

### 🧠 **Enhanced Decision Logic**
- **Physics-Based Validation**: Uses trajectory physics to validate shots
- **False Positive Reduction**: Catches rim bounces that get 100% overlap
- **False Negative Reduction**: Detects fast swooshes with minimal overlap
- **Confidence Scoring**: Multi-factor confidence calculation

## Quick Start

### Enhanced Live Detection
```bash
python main_enhanced.py --action live --model runs/detect/basketball_yolo11n3/weights/best.pt
```

### Enhanced Video Processing
```bash
python main_enhanced.py --action video --video_path your_game.mp4 --model runs/detect/basketball_yolo11n3/weights/best.pt
```

### Compare Approaches
```bash
# Run original approach
python main.py --action video --video_path your_game.mp4

# Run enhanced approach
python main_enhanced.py --action video --video_path your_game.mp4

# Compare results
python compare_approaches.py --original your_game_session.json --enhanced your_game_enhanced_session.json
```

## Key Improvements

### Problem 1: Ball swooshes through basket (False Negatives)
**Solution**: Fast swoosh detection using trajectory analysis
- Detects high-speed downward motion with good overlap (≥85%)
- Validates trajectory consistency even with few frames
- Accounts for ball disappearing below hoop quickly

### Problem 2: Ball bounces on rim (False Positives)  
**Solution**: Rim bounce detection using multiple methods
- **Trajectory Analysis**: Detects upward velocity changes indicating bounce
- **Context Analysis**: Tracks ball reappearing above rim level
- **Physics Validation**: Checks for bounce patterns in velocity

## Enhanced Parameters

```python
# Trajectory Analysis
direction_consistency_threshold = 0.6    # Minimum trajectory consistency
bounce_detection_threshold = 50         # Pixels for upward bounce detection
min_downward_velocity = 10              # Pixels/second for downward motion

# Context Analysis  
pre_frame_count = 10                    # Frames to analyze before shot
post_frame_count = 10                   # Frames to analyze after shot
disappearance_threshold = 5             # Frames to confirm ball disappeared

# Enhanced Decision Logic
confidence_threshold = 0.75             # Overall confidence threshold (adjustable)
```

## Usage Examples

### Adjust Confidence Threshold
```bash
# More conservative (fewer false positives)
python main_enhanced.py --action live --confidence_threshold 0.85

# More aggressive (catch more shots)
python main_enhanced.py --action live --confidence_threshold 0.65
```

### Process Specific Video Segment
```bash
python main_enhanced.py --action video --video_path game.mp4 --start_time 2:30 --end_time 5:45
```

### Batch Process Multiple Videos
```bash
python main_enhanced.py --action batch --video_dir ./game_videos/ --output_dir ./enhanced_results/
```

## Output Analysis

### Enhanced JSON Output
```json
{
  "outcome": "made",
  "enhanced_confidence": 0.87,
  "trajectory_analysis": {
    "direction_consistency": 0.82,
    "has_upward_bounce": false,
    "shows_clean_downward_motion": true,
    "trajectory_smoothness": 0.75
  },
  "context_analysis": {
    "overall_pattern": "clean_made_shot",
    "approach_analysis": {"approaching_from_above": true},
    "exit_analysis": {"type": "ball_disappeared", "disappeared_below_hoop": true}
  }
}
```

### Comparison Report
The comparison tool provides detailed analysis:
- Detection accuracy comparison
- Shot decision agreements/disagreements  
- Enhanced feature utilization
- Performance improvements

## Architecture

```
enhanced_shot_detection.py          # Main enhanced analyzer
├── trajectory_analysis.py          # Trajectory analysis components
├── main_enhanced.py                # Enhanced application entry point
├── compare_approaches.py           # Comparison tool
└── README_ENHANCED.md              # This documentation
```

## Expected Improvements

Based on the analysis of your game data, the enhanced approach should:

1. **Reduce False Positives**: Rim bounces with 100% overlap → correctly identified as misses
2. **Reduce False Negatives**: Fast swooshes with minimal overlap → correctly identified as makes  
3. **Improve Overall Accuracy**: From ~70% to 85-90%+
4. **Provide Better Insights**: Detailed trajectory and context analysis

## Troubleshooting

### If accuracy seems lower initially:
1. **Adjust confidence threshold**: Start with 0.70 and tune up/down
2. **Check trajectory parameters**: Ensure bounce detection threshold fits your camera angle
3. **Validate hoop detection**: Enhanced system requires stable hoop detection

### If missing obvious shots:
- Lower `confidence_threshold` to 0.65-0.70
- Check `min_downward_velocity` parameter
- Ensure sufficient `pre_frame_count` and `post_frame_count`

### If getting false positives:
- Increase `confidence_threshold` to 0.80-0.85  
- Tune `bounce_detection_threshold` for your setup
- Check `direction_consistency_threshold`

## Next Steps

1. **Test the enhanced system** on your game footage
2. **Compare results** using the comparison tool
3. **Fine-tune parameters** based on your specific camera angle and conditions
4. **Analyze the enhanced JSON output** to understand detection reasoning

The enhanced system provides comprehensive shot analysis that should significantly improve accuracy for your basketball shot detection use case.