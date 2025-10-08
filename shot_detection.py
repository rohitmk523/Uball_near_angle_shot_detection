#!/usr/bin/env python3
"""
Basketball Shot Detection and Analysis

This module provides comprehensive basketball shot detection, tracking, and classification
capabilities using YOLOv11 for object detection and trajectory analysis for shot outcomes.

Features:
- Real-time basketball and hoop detection
- Shot trajectory tracking
- Made/missed shot classification with confidence scoring
- JSON logging of all shots and classifications
- Overlay display with shot counters
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import math

class ShotTracker:
    """Tracks individual shot trajectories and determines outcomes"""
    
    def __init__(self, shot_id, initial_position):
        self.shot_id = shot_id
        self.trajectory = deque(maxlen=30)  # Store last 30 positions
        self.trajectory.append(initial_position)
        self.start_time = time.time()
        self.status = "active"  # active, completed, lost
        self.outcome = None     # made, missed, undetermined
        self.confidence = 0.0
        self.peak_reached = False
        self.frames_since_peak = 0
        self.hoop_proximity_threshold = 80
        self.min_trajectory_length = 10
        
    def update_position(self, position):
        """Update ball position and analyze trajectory"""
        self.trajectory.append(position)
        
        # Check if trajectory peak has been reached
        if len(self.trajectory) >= 5:
            self._check_trajectory_peak()
            
        # Check for shot completion
        if self.peak_reached:
            self.frames_since_peak += 1
            if self.frames_since_peak > 15:  # Wait frames after peak
                self._determine_shot_outcome()
                
    def _check_trajectory_peak(self):
        """Check if ball has reached trajectory peak"""
        if len(self.trajectory) < 5:
            return
            
        recent_y = [pos[1] for pos in list(self.trajectory)[-5:]]
        
        # Peak detected if ball starts moving down after going up
        if not self.peak_reached:
            if len(recent_y) >= 3:
                # Check if there's an upward then downward trend
                mid_point = len(recent_y) // 2
                early_y = sum(recent_y[:mid_point]) / mid_point
                later_y = sum(recent_y[mid_point:]) / (len(recent_y) - mid_point)
                
                if early_y > later_y and abs(early_y - later_y) > 5:
                    self.peak_reached = True
                    
    def _determine_shot_outcome(self):
        """Determine if shot was made or missed based on trajectory"""
        if len(self.trajectory) < self.min_trajectory_length:
            self.outcome = "undetermined"
            self.confidence = 0.0
            self.status = "completed"
            return
            
        # Get final position and trajectory direction
        final_pos = self.trajectory[-1]
        trajectory_list = list(self.trajectory)
        
        # Calculate trajectory slope in final phase
        if len(trajectory_list) >= 5:
            final_positions = trajectory_list[-5:]
            x_coords = [pos[0] for pos in final_positions]
            y_coords = [pos[1] for pos in final_positions]
            
            # Calculate if ball is moving toward or away from basket area
            final_velocity_y = y_coords[-1] - y_coords[0]
            
            # Basic heuristic: if ball trajectory ends in expected hoop area
            # This would need hoop position for accurate determination
            # For now, using trajectory characteristics
            
            if final_velocity_y > 0:  # Ball moving downward at end
                self.outcome = "made"
                self.confidence = 0.7
            else:
                self.outcome = "missed"
                self.confidence = 0.6
        else:
            self.outcome = "undetermined"
            self.confidence = 0.0
            
        self.status = "completed"
        
    def determine_outcome_with_hoop(self, hoop_position):
        """Determine shot outcome using hoop position"""
        if not self.peak_reached or len(self.trajectory) < self.min_trajectory_length:
            return
            
        final_pos = self.trajectory[-1]
        distance_to_hoop = math.sqrt(
            (final_pos[0] - hoop_position[0])**2 + 
            (final_pos[1] - hoop_position[1])**2
        )
        
        # Analyze trajectory relative to hoop
        trajectory_list = list(self.trajectory)
        
        # Check if ball passed through or near hoop area
        min_distance_to_hoop = float('inf')
        for pos in trajectory_list:
            dist = math.sqrt(
                (pos[0] - hoop_position[0])**2 + 
                (pos[1] - hoop_position[1])**2
            )
            min_distance_to_hoop = min(min_distance_to_hoop, dist)
            
        # Determine outcome based on proximity to hoop
        if min_distance_to_hoop < self.hoop_proximity_threshold:
            # Ball passed near/through hoop
            if distance_to_hoop < self.hoop_proximity_threshold * 1.5:
                self.outcome = "made"
                self.confidence = 0.85
            else:
                self.outcome = "made"
                self.confidence = 0.75
        else:
            self.outcome = "missed"
            self.confidence = 0.8
            
        self.status = "completed"

class ShotAnalyzer:
    """Main class for basketball shot detection and analysis"""
    
    def __init__(self, model_path='yolo11n.pt'):
        """Initialize shot analyzer with YOLO model"""
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Detection thresholds
        self.basketball_confidence = 0.5
        self.hoop_confidence = 0.5
        
        # Shot tracking
        self.active_shots = {}
        self.completed_shots = []
        self.shot_counter = 0
        self.next_shot_id = 1
        
        # Statistics
        self.stats = {
            'made_shots': 0,
            'missed_shots': 0,
            'total_shots': 0,
            'undetermined_shots': 0
        }
        
        # Logging
        self.shot_log = []
        self.session_start = datetime.now()
        
    def detect_objects(self, frame):
        """Detect basketball and hoop in frame"""
        results = self.model(frame, conf=0.3, verbose=False)
        
        detections = {
            'basketball': [],
            'basketball_hoop': []
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Determine object type based on model
                    if 'best.pt' in str(self.model_path) or 'custom' in str(self.model_path):
                        # Custom model classes: 0=Basketball, 1=Basketball Hoop
                        if class_id == 0 and confidence >= self.basketball_confidence:
                            detections['basketball'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': [center_x, center_y],
                                'confidence': confidence
                            })
                        elif class_id == 1 and confidence >= self.hoop_confidence:
                            detections['basketball_hoop'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': [center_x, center_y],
                                'confidence': confidence
                            })
                    else:
                        # Pre-trained model (look for sports balls)
                        class_name = self.model.names[class_id].lower()
                        if 'ball' in class_name and confidence >= self.basketball_confidence:
                            detections['basketball'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': [center_x, center_y],
                                'confidence': confidence,
                                'class_name': class_name
                            })
                            
        return detections
        
    def update_shot_tracking(self, detections):
        """Update shot tracking with new detections"""
        basketball_positions = [ball['center'] for ball in detections['basketball']]
        hoop_positions = [hoop['center'] for hoop in detections['basketball_hoop']]
        
        # Update existing shots
        shots_to_remove = []
        for shot_id, shot_tracker in self.active_shots.items():
            # Find closest basketball to this shot
            if basketball_positions:
                distances = [
                    math.sqrt((pos[0] - shot_tracker.trajectory[-1][0])**2 + 
                             (pos[1] - shot_tracker.trajectory[-1][1])**2)
                    for pos in basketball_positions
                ]
                min_distance = min(distances)
                closest_idx = distances.index(min_distance)
                
                if min_distance < 100:  # Ball still being tracked
                    shot_tracker.update_position(basketball_positions[closest_idx])
                    basketball_positions.pop(closest_idx)  # Remove matched ball
                else:
                    # Ball lost, complete the shot
                    if hoop_positions:
                        shot_tracker.determine_outcome_with_hoop(hoop_positions[0])
                    else:
                        shot_tracker._determine_shot_outcome()
                    shots_to_remove.append(shot_id)
            else:
                # No basketballs detected, complete existing shots
                if hoop_positions:
                    shot_tracker.determine_outcome_with_hoop(hoop_positions[0])
                else:
                    shot_tracker._determine_shot_outcome()
                shots_to_remove.append(shot_id)
                
        # Remove completed shots
        for shot_id in shots_to_remove:
            completed_shot = self.active_shots.pop(shot_id)
            self.completed_shots.append(completed_shot)
            self._log_completed_shot(completed_shot)
            
        # Create new shots for unmatched basketballs
        for position in basketball_positions:
            shot_tracker = ShotTracker(self.next_shot_id, position)
            self.active_shots[self.next_shot_id] = shot_tracker
            self.next_shot_id += 1
            
    def _log_completed_shot(self, shot_tracker):
        """Log completed shot to statistics and JSON log"""
        self.stats['total_shots'] += 1
        
        if shot_tracker.outcome == "made":
            self.stats['made_shots'] += 1
        elif shot_tracker.outcome == "missed":
            self.stats['missed_shots'] += 1
        else:
            self.stats['undetermined_shots'] += 1
            
        # Add to shot log
        shot_data = {
            'shot_id': shot_tracker.shot_id,
            'timestamp': datetime.now().isoformat(),
            'duration': time.time() - shot_tracker.start_time,
            'outcome': shot_tracker.outcome,
            'confidence': shot_tracker.confidence,
            'trajectory_length': len(shot_tracker.trajectory),
            'trajectory': list(shot_tracker.trajectory)
        }
        
        self.shot_log.append(shot_data)
        
    def draw_overlay(self, frame, detections):
        """Draw detection overlay and shot statistics"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw detections
        for ball in detections['basketball']:
            x1, y1, x2, y2 = ball['bbox']
            confidence = ball['confidence']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Draw label
            label = f"Ball: {confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Draw center point
            center = ball['center']
            cv2.circle(overlay, center, 5, (0, 165, 255), -1)
            
        for hoop in detections['basketball_hoop']:
            x1, y1, x2, y2 = hoop['bbox']
            confidence = hoop['confidence']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Hoop: {confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Draw shot trajectories
        for shot_tracker in self.active_shots.values():
            trajectory = list(shot_tracker.trajectory)
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(overlay, trajectory[i-1], trajectory[i], (255, 255, 0), 2)
                    
        # Draw statistics overlay
        self._draw_stats_overlay(overlay)
        
        return overlay
        
    def _draw_stats_overlay(self, frame):
        """Draw shot statistics overlay"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent background for stats
        overlay_bg = np.zeros((120, 300, 3), dtype=np.uint8)
        overlay_bg[:] = (0, 0, 0)  # Black background
        
        # Add transparency
        alpha = 0.7
        y_offset = 20
        x_offset = 20
        
        # Draw background rectangle
        cv2.rectangle(frame, (x_offset, y_offset), 
                     (x_offset + 300, y_offset + 120), (0, 0, 0), -1)
        
        # Calculate shooting percentage
        total_determined = self.stats['made_shots'] + self.stats['missed_shots']
        shooting_pct = (self.stats['made_shots'] / total_determined * 100) if total_determined > 0 else 0
        
        # Draw statistics text
        stats_text = [
            f"SHOT TRACKER",
            f"Made: {self.stats['made_shots']}",
            f"Missed: {self.stats['missed_shots']}",
            f"Total: {self.stats['total_shots']}",
            f"Shooting %: {shooting_pct:.1f}%"
        ]
        
        colors = [(255, 255, 255), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 165, 0)]
        
        for i, text in enumerate(stats_text):
            y_pos = y_offset + 25 + (i * 20)
            cv2.putText(frame, text, (x_offset + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                       
        # Show active shots
        if self.active_shots:
            active_text = f"Tracking: {len(self.active_shots)} shots"
            cv2.putText(frame, active_text, (x_offset + 10, y_offset + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    def save_session_data(self, filename=None):
        """Save session data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shot_session_{timestamp}.json"
            
        session_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'total_duration': time.time() - self.session_start.timestamp()
            },
            'statistics': self.stats,
            'shots': self.shot_log
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"✓ Session data saved to: {filename}")
        return filename