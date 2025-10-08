# Basketball Shot Detection System

A comprehensive basketball shot detection system using YOLOv11 for object detection and advanced trajectory analysis for shot outcome classification.

## Features

- **Zero-shot Detection**: Test basketball detection using pre-trained YOLO models
- **Custom Training**: Train specialized models for basketball and hoop detection
- **Shot Tracking**: Advanced trajectory analysis for shot detection
- **Made/Missed Classification**: Automated shot outcome determination with confidence scores
- **Real-time Processing**: Live camera feed analysis
- **Batch Processing**: Process multiple videos efficiently
- **Performance Evaluation**: Comprehensive metrics and benchmarking

## Camera Setup

This system is optimized for overhead/near-angle camera placement as shown in the reference image. The GoPro setup provides excellent coverage of the basketball court and hoop area.

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd Uball_near_angle_shot_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project includes a pre-configured basketball dataset with Basketball and Basketball Hoop annotations in YOLOv11 format located in the `basketball_dataset/` directory. The dataset contains:

- **Training set**: Images and labels for model training
- **Validation set**: Images and labels for model validation  
- **Test set**: Images and labels for model evaluation
- **Classes**: Basketball (class 0), Basketball Hoop (class 1)

## Quick Start

### 1. Train Your Model

```bash
# Setup training environment
python custom_training.py --action setup

# Train the model (adjust epochs and batch size as needed)
python custom_training.py --action train --epochs 50 --batch_size 8 --device cpu

# Validate trained model
python custom_training.py --action validate --model_path runs/detect/basketball_yolo11n/weights/best.pt
```

### 2. Test with Single Image/Screenshot

```bash
# Test with trained model on default test image
python test_simple.py --model runs/detect/basketball_yolo11n/weights/best.pt

# Test with specific image and show results
python test_simple.py --model runs/detect/basketball_yolo11n/weights/best.pt --image test.png --show

# Test with pre-trained YOLO model (limited accuracy)
python test_simple.py --image test.png
```

### 3. Live Camera Detection with Shot Tracking

```bash
# Live detection with trained model and shot tracking
python main.py --action live --model runs/detect/basketball_yolo11n/weights/best.pt

# Live detection with specific camera
python main.py --action live --model runs/detect/basketball_yolo11n/weights/best.pt --camera 1
```

### 4. Process Videos

```bash
# Process single video with shot detection and overlay
python main.py --action video --video_path your_video.mp4 --model runs/detect/basketball_yolo11n/weights/best.pt

# Process multiple videos in batch
python main.py --action batch --video_dir videos/ --model runs/detect/basketball_yolo11n/weights/best.pt --output_dir processed/
```

### 5. Export Trained Model

```bash
# Export model to ONNX format for deployment
python custom_training.py --action export --model_path runs/detect/basketball_yolo11n/weights/best.pt --export_format onnx
```

## Custom Model Training

If zero-shot detection doesn't provide sufficient accuracy, train a custom model:

### 1. Set up Data Collection

```bash
# Create dataset structure
python data_collection.py --action organize

# Extract frames from videos
python data_collection.py --action extract --video_dir raw_videos/
```

### 2. Annotate Data

Use one of these annotation tools:
- **LabelImg**: `pip install labelImg` (Recommended for YOLO format)
- **Roboflow**: Online tool with auto-annotation features
- **CVAT**: Professional annotation tool

Classes to annotate:
- `basketball` (class 0)
- `basketball_hoop` (class 1)

### 3. Train Model

```bash
# Setup training pipeline
python custom_training.py --action setup

# Train custom model
python custom_training.py --action train --dataset_yaml basketball_dataset/dataset.yaml
```

### 4. Evaluate Model

```bash
python custom_training.py --action evaluate --model_path runs/detect/train/weights/best.pt --dataset_yaml basketball_dataset/dataset.yaml
```

## Project Structure

```
├── main.py                 # Main application entry point
├── shot_detection.py       # Core shot detection and tracking logic
├── custom_training.py      # Model training pipeline
├── data_collection.py      # Data collection and organization
├── evaluation.py           # Evaluation metrics and benchmarking
├── zero_shot_test.py       # Zero-shot detection testing
├── test_simple.py          # Simple detection test
├── requirements.txt        # Python dependencies
└── basketball_dataset/     # Dataset directory (created during setup)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml
```

## Usage Examples

### Basic Video Processing

```python
from shot_detection import ShotAnalyzer

# Initialize analyzer
analyzer = ShotAnalyzer('yolo11n.pt')  # or path to custom model

# Process video
results = analyzer.process_video('basketball_game.mp4', 'output_with_tracking.mp4')

print(f"Detected {len(results)} shots")
for shot in results:
    print(f"Frame {shot['frame']}: {shot['result']['outcome']} (confidence: {shot['result']['confidence']:.3f})")
```

### Custom Model Training

```python
from custom_training import BasketballTrainer

# Setup trainer
trainer = BasketballTrainer('basketball_dataset/dataset.yaml')

# Train model
results = trainer.train_model(
    model_size='n',  # nano model for speed
    epochs=100,
    batch_size=16
)

# Validate
val_results = trainer.validate_model()
```

### Performance Evaluation

```python
from evaluation import ShotDetectionEvaluator

# Load ground truth and predictions
evaluator = ShotDetectionEvaluator()
evaluator.load_ground_truth('ground_truth.json')
evaluator.load_predictions('predictions.json')

# Calculate metrics
evaluator.calculate_detection_metrics()
evaluator.calculate_shot_classification_metrics()

# Generate report
evaluator.generate_report('evaluation_report.json')
```

## Configuration

Key parameters in `shot_detection.py`:

```python
# Shot detection sensitivity
min_shot_distance = 100          # Minimum movement for shot detection
peak_detection_frames = 5        # Frames to confirm trajectory peak
hoop_proximity_threshold = 50    # Pixels from hoop for made shot

# Model confidence thresholds
basketball_confidence = 0.5      # Minimum confidence for basketball detection
hoop_confidence = 0.5           # Minimum confidence for hoop detection
```

## Performance Benchmarks

### Zero-shot YOLO Performance
- Basketball detection: Limited (depends on sports ball class in COCO)
- Hoop detection: Poor (no specific hoop class in COCO)
- **Recommendation**: Use custom training for production

### Custom Model Performance (Expected)
- Basketball detection: >90% accuracy with proper training data
- Hoop detection: >85% accuracy with proper training data
- Inference speed: ~30-60 FPS on modern GPUs

## Evaluation Metrics

### Object Detection
- **Precision**: Percentage of detected objects that are correct
- **Recall**: Percentage of actual objects that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision across confidence thresholds

### Shot Classification
- **Accuracy**: Percentage of correctly classified shots
- **Precision/Recall**: Per-class performance for made/missed shots
- **Confidence Distribution**: Analysis of prediction confidence scores

### Trajectory Tracking
- **Mean Trajectory Error**: Average pixel distance from ground truth
- **Temporal Consistency**: Frame-to-frame tracking stability

## Troubleshooting

### Common Issues

1. **No objects detected**: 
   - Check camera angle and lighting
   - Adjust confidence thresholds
   - Consider custom model training

2. **Poor shot classification**:
   - Verify hoop position detection
   - Adjust trajectory analysis parameters
   - Increase training data for custom model

3. **Slow processing**:
   - Use smaller YOLO model (yolo11n vs yolo11x)
   - Reduce video resolution
   - Use GPU acceleration

### Debug Mode

Enable debug output:
```python
# In shot_detection.py, set verbose=True
results = self.model(frame, conf=0.5, verbose=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- YOLOv11 by Ultralytics
- OpenCV for computer vision operations
- Basketball community for testing and feedback