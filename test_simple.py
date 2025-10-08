#!/usr/bin/env python3
"""
Simple Basketball Detection Test

This script tests basketball and hoop detection on a single image using either
a pre-trained YOLO model or a custom trained model.

Usage:
    python test_simple.py                                    # Test with default image and model
    python test_simple.py --image test.png                   # Test with specific image
    python test_simple.py --model runs/detect/train/weights/best.pt  # Test with custom model
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

class BasketballDetector:
    def __init__(self, model_path='yolo11n.pt'):
        """Initialize basketball detector with specified model"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.basketball_confidence = 0.5
        self.hoop_confidence = 0.5
        
        # Class names for custom model (Basketball=0, Basketball Hoop=1)
        self.custom_classes = {0: 'Basketball', 1: 'Basketball Hoop'}
        
    def detect_objects(self, image_path, save_results=True):
        """Detect basketball and hoop in image"""
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None
            
        print(f"Analyzing image: {image_path}")
        print(f"Using model: {self.model_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
            
        # Run detection
        results = self.model(image, conf=0.3, verbose=False)
        
        # Process results
        detections = {
            'basketball': [],
            'basketball_hoop': [],
            'other_objects': []
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Determine object type
                    if 'best.pt' in str(self.model_path) or 'custom' in str(self.model_path):
                        # Custom model classes
                        if class_id == 0 and confidence >= self.basketball_confidence:
                            detections['basketball'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'center': [int((x1+x2)/2), int((y1+y2)/2)]
                            })
                        elif class_id == 1 and confidence >= self.hoop_confidence:
                            detections['basketball_hoop'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'center': [int((x1+x2)/2), int((y1+y2)/2)]
                            })
                    else:
                        # Pre-trained model (COCO classes)
                        class_name = self.model.names[class_id].lower()
                        if 'ball' in class_name and confidence >= self.basketball_confidence:
                            detections['basketball'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'center': [int((x1+x2)/2), int((y1+y2)/2)],
                                'class_name': class_name
                            })
                        else:
                            detections['other_objects'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_name': class_name
                            })
        
        # Draw detections on image
        annotated_image = self.draw_detections(image.copy(), detections)
        
        # Save results if requested
        if save_results:
            output_path = Path(image_path).stem + '_detected.jpg'
            cv2.imwrite(output_path, annotated_image)
            print(f"✓ Results saved to: {output_path}")
            
            # Save detection data
            json_path = Path(image_path).stem + '_detections.json'
            detection_data = {
                'timestamp': datetime.now().isoformat(),
                'image_path': str(image_path),
                'model_path': str(self.model_path),
                'detections': detections,
                'summary': {
                    'basketball_count': len(detections['basketball']),
                    'hoop_count': len(detections['basketball_hoop']),
                    'other_objects_count': len(detections['other_objects'])
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
            print(f"✓ Detection data saved to: {json_path}")
        
        return detections, annotated_image
        
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        
        # Colors for different objects
        colors = {
            'basketball': (0, 165, 255),      # Orange
            'basketball_hoop': (0, 255, 0),   # Green
            'other_objects': (255, 0, 0)      # Blue
        }
        
        # Draw basketball detections
        for obj in detections['basketball']:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colors['basketball'], 2)
            
            # Draw label
            label = f"Basketball: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), colors['basketball'], -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center = obj['center']
            cv2.circle(image, center, 5, colors['basketball'], -1)
        
        # Draw hoop detections
        for obj in detections['basketball_hoop']:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colors['basketball_hoop'], 2)
            
            # Draw label
            label = f"Hoop: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), colors['basketball_hoop'], -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center = obj['center']
            cv2.circle(image, center, 5, colors['basketball_hoop'], -1)
        
        # Draw other objects (for pre-trained models)
        for obj in detections['other_objects']:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            class_name = obj.get('class_name', 'object')
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colors['other_objects'], 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), colors['other_objects'], -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
        
    def analyze_shot_possibility(self, detections):
        """Analyze if a shot scenario is possible based on detections"""
        
        basketball_count = len(detections['basketball'])
        hoop_count = len(detections['basketball_hoop'])
        
        print("\n" + "="*50)
        print("SHOT ANALYSIS")
        print("="*50)
        
        if basketball_count == 0:
            print("❌ No basketball detected")
            print("   - Check lighting and image quality")
            print("   - Consider using custom trained model")
            return False
            
        if hoop_count == 0:
            print("❌ No basketball hoop detected")
            print("   - Hoop may be out of frame or occluded")
            print("   - Pre-trained models have limited hoop detection")
            print("   - Recommend using custom trained model")
            return False
            
        # Calculate distances between balls and hoops
        for i, ball in enumerate(detections['basketball']):
            ball_center = ball['center']
            
            for j, hoop in enumerate(detections['basketball_hoop']):
                hoop_center = hoop['center']
                
                # Calculate distance
                distance = np.sqrt((ball_center[0] - hoop_center[0])**2 + 
                                 (ball_center[1] - hoop_center[1])**2)
                
                print(f"✓ Ball {i+1} to Hoop {j+1}: {distance:.1f} pixels")
                
                # Determine shot scenario
                if distance < 100:
                    print(f"   → CLOSE RANGE: Ball near hoop (possible made shot)")
                elif distance < 300:
                    print(f"   → MEDIUM RANGE: Ball in shooting range")
                else:
                    print(f"   → LONG RANGE: Ball far from hoop")
        
        print(f"\n✓ Detection Summary:")
        print(f"   - Basketballs detected: {basketball_count}")
        print(f"   - Hoops detected: {hoop_count}")
        print(f"   - Shot analysis: POSSIBLE")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Test Basketball Detection')
    parser.add_argument('--image', type=str, default='test.png',
                       help='Path to test image')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save detection results')
    parser.add_argument('--show', action='store_true',
                       help='Display result image')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASKETBALL DETECTION TEST")
    print("="*60)
    
    # Initialize detector
    detector = BasketballDetector(args.model)
    
    # Run detection
    result = detector.detect_objects(args.image, save_results=not args.no_save)
    
    if result is not None:
        detections, annotated_image = result
        
        # Analyze shot possibility
        detector.analyze_shot_possibility(detections)
        
        # Show image if requested
        if args.show:
            cv2.imshow('Basketball Detection', annotated_image)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        print(f"\n🏀 Detection test completed!")
        
        if not args.no_save:
            print("Check the generated files:")
            print(f"   - Annotated image: {Path(args.image).stem}_detected.jpg")
            print(f"   - Detection data: {Path(args.image).stem}_detections.json")
    else:
        print("❌ Detection test failed")

if __name__ == "__main__":
    main()