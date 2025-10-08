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
    
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.87, min_overlap_threshold=1.0):
        """Initialize shot analyzer with YOLO model and configurable thresholds"""
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Enable GPU acceleration for MacBook M-series chips
        import torch
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"🚀 Using MacBook GPU (MPS) acceleration")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print(f"🚀 Using CUDA GPU acceleration")
        else:
            self.device = 'cpu'
            print("ℹ️ Using CPU")
        
        # Move model to device
        self.model.to(self.device)
        
        # Detection thresholds
        self.basketball_confidence = 0.35
        self.hoop_confidence = 0.5
        
        # Improved shot detection with box overlap
        self.hoop_area = None
        self.hoop_center = None
        self.hoop_size = None
        self.hoop_bbox = None
        
        # Shot sequence grouping to prevent multiple detections
        self.shot_sequence_active = False
        self.shot_sequence_start_time = None
        self.shot_sequence_overlaps = []  # Store all overlaps in current sequence
        self.shot_sequence_timeout = 3.0  # 3 seconds to group overlaps into one shot
        self.current_overlap_percentage = 0
        
        # Configurable thresholds for robustness
        self.confidence_threshold = confidence_threshold  # Threshold for made vs missed (0.87 default)
        self.min_overlap_threshold = min_overlap_threshold  # Minimum overlap to consider as shot (1.0% default)
        
        # Additional robustness features
        self.recent_shots = []  # Track recent shots to prevent duplicates
        self.duplicate_prevention_window = 2.0  # seconds
        self.shot_quality_factors = {
            'ideal_size_ratio_range': (0.45, 0.65),  # Tighter ideal range
            'max_vertical_distance': 80,  # More restrictive
            'sequence_consistency_bonus': 0.03,  # Reduced bonus
            'min_quality_frames': 3  # Require minimum quality frames
        }
        
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
        results = self.model(frame, conf=0.3, verbose=False, device=self.device)
        
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
        """Enhanced shot detection: 1 frame with 100% overlap = made, otherwise missed"""
        basketball_positions = [ball['center'] for ball in detections['basketball']]
        hoop_positions = [hoop['center'] for hoop in detections['basketball_hoop']]
        
        # Update hoop area if hoop detected
        if hoop_positions:
            hoop = detections['basketball_hoop'][0]  # Use first detected hoop
            self.hoop_center = hoop['center']
            self.hoop_bbox = hoop['bbox']  # Store actual hoop bbox for overlap detection
            self.hoop_size = max(self.hoop_bbox[2] - self.hoop_bbox[0], 
                               self.hoop_bbox[3] - self.hoop_bbox[1])
            
            # Define hoop shooting area (expanded around hoop)
            margin = int(self.hoop_size * 0.8)  # 80% of hoop size as margin
            x1 = self.hoop_bbox[0] - margin
            y1 = self.hoop_bbox[1] - margin  
            x2 = self.hoop_bbox[2] + margin
            y2 = self.hoop_bbox[3] + margin
            self.hoop_area = [x1, y1, x2, y2]
            
        current_time = time.time()
        
        # Check for shot sequence timeout
        if (self.shot_sequence_active and 
            self.shot_sequence_start_time and 
            current_time - self.shot_sequence_start_time > self.shot_sequence_timeout):
            self._finalize_shot_sequence()
        
        # Check each basketball for overlap with hoop
        if basketball_positions and self.hoop_area and self.hoop_bbox:
            for ball in detections['basketball']:
                ball_center = ball['center']
                ball_bbox = ball['bbox']
                ball_size = max(ball_bbox[2] - ball_bbox[0], ball_bbox[3] - ball_bbox[1])
                
                # Check if ball is in shooting zone
                in_shooting_zone = (self.hoop_area[0] <= ball_center[0] <= self.hoop_area[2] and 
                                  self.hoop_area[1] <= ball_center[1] <= self.hoop_area[3])
                
                if in_shooting_zone:
                    # Enhanced ball validation for position and size
                    size_ratio = ball_size / self.hoop_size
                    
                    # Check vertical position relative to hoop
                    ball_y = ball_center[1]
                    hoop_y = self.hoop_center[1]
                    vertical_distance = ball_y - hoop_y
                    
                    # Balanced size and position validation - more selective than before
                    # Tighten ranges to reduce false positives while still catching legitimate shots
                    valid_size = 0.35 <= size_ratio <= 0.8  # Tighter range for better accuracy
                    valid_position = vertical_distance <= 100  # Reasonable position constraint
                    
                    # More selective validation for different ball positions relative to hoop
                    if vertical_distance > 40:  # Ball below hoop level
                        # Require appropriate size for below-rim balls
                        valid_size = 0.4 <= size_ratio <= 0.8
                        valid_position = vertical_distance <= 80  # More restrictive for below-rim balls
                    elif vertical_distance < -40:  # Ball above hoop level
                        # Balls above rim should be reasonably sized
                        valid_size = 0.3 <= size_ratio <= 0.7
                        valid_position = vertical_distance >= -100  # Prevent too-high detections
                    else:  # Ball near hoop level (-40 to +40 pixels)
                        # Optimal range for through-basket shots - based on screenshot analysis
                        valid_size = 0.4 <= size_ratio <= 0.75
                    
                    if valid_size and valid_position:
                        
                        # Check overlap percentage between ball and hoop
                        overlap_percentage = self._check_box_overlap(ball_bbox, self.hoop_bbox)
                        self.current_overlap_percentage = overlap_percentage
                        
                        # Group overlaps into shot sequences - configurable overlap threshold
                        if overlap_percentage >= self.min_overlap_threshold:
                            if not self.shot_sequence_active:
                                # Start new shot sequence
                                self.shot_sequence_active = True
                                self.shot_sequence_start_time = current_time
                                self.shot_sequence_overlaps = []
                                
                            # Add overlap to current sequence
                            overlap_data = {
                                'frame_time': current_time,
                                'overlap_percentage': overlap_percentage,
                                'confidence': ball['confidence'],
                                'ball_position': ball_center,
                                'size_ratio': size_ratio
                            }
                            self.shot_sequence_overlaps.append(overlap_data)
                            
        # If no overlaps detected and sequence is active, check for timeout soon
        else:
            if (self.shot_sequence_active and 
                self.shot_sequence_start_time and 
                current_time - self.shot_sequence_start_time > 1.0):  # 1 second mini-timeout
                self._finalize_shot_sequence()
            
    def _check_box_overlap(self, ball_bbox, hoop_bbox):
        """Calculate overlap percentage between ball and hoop bounding boxes"""
        # ball_bbox and hoop_bbox are [x1, y1, x2, y2]
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox
        hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_bbox
        
        # Calculate intersection area
        overlap_x = max(0, min(ball_x2, hoop_x2) - max(ball_x1, hoop_x1))
        overlap_y = max(0, min(ball_y2, hoop_y2) - max(ball_y1, hoop_y1))
        intersection_area = overlap_x * overlap_y
        
        # Calculate ball area
        ball_area = (ball_x2 - ball_x1) * (ball_y2 - ball_y1)
        
        # Return overlap percentage
        if ball_area > 0:
            overlap_percentage = (intersection_area / ball_area) * 100
            return overlap_percentage
        return 0
    
    def _calculate_shot_confidence(self, shot_overlaps, max_overlap):
        """Calculate confidence score based on overlap quality and frame count"""
        if not shot_overlaps:
            return 0.0
            
        # More conservative base confidence - require higher overlap for high base confidence
        if max_overlap >= 100.0:
            base_confidence = 0.7  # Start lower even for perfect overlap
        elif max_overlap >= 95.0:
            base_confidence = 0.5
        elif max_overlap >= 90.0:
            base_confidence = 0.3
        else:
            base_confidence = max_overlap / 100.0 * 0.3  # Very low for <90% overlap
        
        # Count frames with different overlap thresholds
        frames_100_percent = sum(1 for overlap in shot_overlaps if overlap['overlap_percentage'] >= 100.0)
        frames_95_percent = sum(1 for overlap in shot_overlaps if overlap['overlap_percentage'] >= 95.0)
        frames_80_percent = sum(1 for overlap in shot_overlaps if overlap['overlap_percentage'] >= 80.0)
        total_frames = len(shot_overlaps)
        
        # Much more demanding confidence bonuses - require exceptional quality for made shots
        frame_bonus = 0.0
        if frames_100_percent >= 5:  # Require 5+ perfect frames for highest bonus
            frame_bonus += 0.15 + (frames_100_percent - 5) * 0.02  # Smaller bonuses
        elif frames_100_percent >= 3:  # 3-4 perfect frames = good
            frame_bonus += 0.10
        elif frames_100_percent >= 2:  # Two perfect frames = moderate
            frame_bonus += 0.06
        elif frames_100_percent == 1:  # Single 100% frame = small bonus
            frame_bonus += 0.03
            
        if frames_95_percent >= 6:  # Require many high-overlap frames
            frame_bonus += 0.08
        elif frames_95_percent >= 4:  # Some bonus for 4-5 high frames
            frame_bonus += 0.04
        elif frames_95_percent >= 2:  # Minimal bonus for 2-3 high frames
            frame_bonus += 0.02
            
        # Sequence quality bonus - much more demanding
        if total_frames >= 8 and frames_80_percent / total_frames >= 0.8:  # Very demanding
            frame_bonus += 0.04
        elif total_frames >= 6 and frames_80_percent / total_frames >= 0.75:
            frame_bonus += 0.02
            
        # Size ratio consistency bonus - very demanding
        size_ratios = [overlap['size_ratio'] for overlap in shot_overlaps]
        avg_size_ratio = sum(size_ratios) / len(size_ratios)
        if 0.5 <= avg_size_ratio <= 0.6:  # Very tight ideal range
            frame_bonus += 0.05
        elif 0.45 <= avg_size_ratio <= 0.65:  # Good range
            frame_bonus += 0.02
            
        # Final confidence calculation (cap at 1.0)
        final_confidence = min(1.0, base_confidence + frame_bonus)
        return final_confidence

    def _is_duplicate_shot(self, current_time, ball_position):
        """Check if this shot is a duplicate of a recent shot"""
        for recent_shot in self.recent_shots:
            time_diff = current_time - recent_shot['time']
            if time_diff <= self.duplicate_prevention_window:
                # Check if ball position is similar to recent shot
                pos_diff = abs(ball_position[0] - recent_shot['position'][0]) + abs(ball_position[1] - recent_shot['position'][1])
                if pos_diff < 50:  # Within 50 pixels
                    return True
        return False

    def _finalize_shot_sequence(self):
        """Finalize shot sequence and determine outcome based on confidence threshold"""
        if not self.shot_sequence_overlaps:
            return
            
        # Check for duplicate shots
        current_time = time.time()
        max_overlap_position = max(self.shot_sequence_overlaps, key=lambda x: x['overlap_percentage'])['ball_position']
        
        if self._is_duplicate_shot(current_time, max_overlap_position):
            # Reset sequence without logging
            self.shot_sequence_active = False
            self.shot_sequence_start_time = None
            self.shot_sequence_overlaps = []
            return
            
        # Find the maximum overlap percentage in the sequence
        max_overlap = max(overlap['overlap_percentage'] for overlap in self.shot_sequence_overlaps)
        
        # Get the overlap data with maximum percentage
        max_overlap_data = next(overlap for overlap in self.shot_sequence_overlaps 
                               if overlap['overlap_percentage'] == max_overlap)
        
        # Calculate confidence score for this shot
        shot_confidence = self._calculate_shot_confidence(self.shot_sequence_overlaps, max_overlap)
        
        # Count frames with different overlap thresholds for logging
        frames_with_100_percent = sum(1 for overlap in self.shot_sequence_overlaps 
                                    if overlap['overlap_percentage'] >= 100.0)
        frames_with_95_percent = sum(1 for overlap in self.shot_sequence_overlaps 
                                   if overlap['overlap_percentage'] >= 95.0)
        
        # Confidence-based outcome determination
        if max_overlap >= 100.0 and shot_confidence >= self.confidence_threshold:
            outcome = "made"
            self.stats['made_shots'] += 1
        else:
            outcome = "missed"
            self.stats['missed_shots'] += 1
            
        self.stats['total_shots'] += 1
        
        # Log the single shot with comprehensive data including confidence metrics
        shot_data = {
            'frame_time': max_overlap_data['frame_time'],
            'timestamp': datetime.now().isoformat(),
            'outcome': outcome,
            'confidence': max_overlap_data['confidence'],  # Ball detection confidence
            'shot_confidence': shot_confidence,  # New: Overall shot confidence score
            'confidence_threshold': self.confidence_threshold,  # New: Threshold used for decision
            'max_overlap_percentage': max_overlap,
            'frames_with_100_percent': frames_with_100_percent,  # New: Count of perfect overlap frames
            'frames_with_95_percent': frames_with_95_percent,
            'total_overlaps_in_sequence': len(self.shot_sequence_overlaps),
            'sequence_duration': self.shot_sequence_overlaps[-1]['frame_time'] - self.shot_sequence_overlaps[0]['frame_time'],
            'ball_position': max_overlap_data['ball_position'],
            'hoop_center': self.hoop_center,
            'size_ratio': max_overlap_data['size_ratio'],
            'avg_size_ratio': sum(overlap['size_ratio'] for overlap in self.shot_sequence_overlaps) / len(self.shot_sequence_overlaps),  # New: Average size ratio
            'detection_method': 'confidence_based_sequence'  # Updated method name
        }
        self.shot_log.append(shot_data)
        
        # Add to recent shots for duplicate prevention
        self.recent_shots.append({
            'time': current_time,
            'position': max_overlap_data['ball_position'],
            'outcome': outcome
        })
        
        # Clean up old recent shots (keep only within duplicate prevention window)
        self.recent_shots = [shot for shot in self.recent_shots 
                           if current_time - shot['time'] <= self.duplicate_prevention_window]
        
        # Reset sequence tracking
        self.shot_sequence_active = False
        self.shot_sequence_start_time = None
        self.shot_sequence_overlaps = []
        
    def draw_overlay(self, frame, detections):
        """Draw clean detection overlay and shot statistics"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw hoop detection and area
        for hoop in detections['basketball_hoop']:
            x1, y1, x2, y2 = hoop['bbox']
            confidence = hoop['confidence']
            
            # Draw clean hoop box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw hoop area if defined
            if self.hoop_area:
                area_color = (0, 255, 0)
                cv2.rectangle(overlay, (self.hoop_area[0], self.hoop_area[1]), 
                            (self.hoop_area[2], self.hoop_area[3]), area_color, 1)
                
                # Label the shooting zone
                cv2.putText(overlay, "SHOOTING ZONE", 
                          (self.hoop_area[0], self.hoop_area[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_color, 1)
            
        # Draw basketball detection with fixed color box and confidence
        for ball in detections['basketball']:
            x1, y1, x2, y2 = ball['bbox']
            center = ball['center']
            confidence = ball['confidence']
            
            # Fixed color bounding box for ball (orange)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Confidence label on box
            label = f"Ball: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), (0, 165, 255), -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Check if ball is in shooting zone
            if (self.hoop_area and 
                self.hoop_area[0] <= center[0] <= self.hoop_area[2] and 
                self.hoop_area[1] <= center[1] <= self.hoop_area[3]):
                # Highlight ball in shooting zone with thicker border
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                # Check for box overlap with hoop
                if self.hoop_bbox:
                    overlap_pct = self._check_box_overlap(ball['bbox'], self.hoop_bbox)
                    if overlap_pct > 0:
                        # Simple color coding: 95% = green (made), anything else = red (missed)
                        if overlap_pct >= 95.0:
                            color = (0, 255, 0)  # Green for 95%+ overlap (made)
                            label = f"MADE: {overlap_pct:.0f}%"
                        else:
                            color = (0, 0, 255)  # Red for partial overlap (missed)
                            label = f"MISS: {overlap_pct:.0f}%"
                            
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(overlay, label, (x1, y2 + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        # Draw clean statistics overlay
        self._draw_clean_stats_overlay(overlay)
        
        return overlay
        
    def _draw_clean_stats_overlay(self, frame):
        """Draw clean, minimal shot statistics overlay"""
        height, width = frame.shape[:2]
        
        # Calculate shooting percentage
        total_determined = self.stats['made_shots'] + self.stats['missed_shots']
        shooting_pct = (self.stats['made_shots'] / total_determined * 100) if total_determined > 0 else 0
        
        # Position overlay in top-right corner
        x_offset = width - 200
        y_offset = 20
        
        # Semi-transparent background
        overlay_bg = np.zeros((80, 180, 3), dtype=np.uint8)
        alpha = 0.3
        
        # Create overlay region
        overlay_region = frame[y_offset:y_offset+80, x_offset:x_offset+180]
        frame[y_offset:y_offset+80, x_offset:x_offset+180] = cv2.addWeighted(
            overlay_region, 1-alpha, overlay_bg, alpha, 0)
        
        # Clean, minimal text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Shot statistics
        cv2.putText(frame, f"MADE: {self.stats['made_shots']}", 
                   (x_offset + 10, y_offset + 20), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"MISSED: {self.stats['missed_shots']}", 
                   (x_offset + 10, y_offset + 40), font, font_scale, (0, 100, 255), thickness)
        cv2.putText(frame, f"TOTAL: {self.stats['total_shots']}", 
                   (x_offset + 10, y_offset + 60), font, font_scale, (255, 255, 255), thickness)
        
        # Show shot sequence tracking info
        if self.shot_sequence_active:
            overlaps_count = len(self.shot_sequence_overlaps)
            max_overlap = max([o['overlap_percentage'] for o in self.shot_sequence_overlaps]) if self.shot_sequence_overlaps else 0
            frames_95 = sum(1 for o in self.shot_sequence_overlaps if o['overlap_percentage'] >= 95.0)
            
            cv2.putText(frame, f"SEQUENCE: {overlaps_count} overlaps", 
                       (x_offset + 10, y_offset + 80), font, font_scale, (255, 255, 0), thickness)
            
            if max_overlap >= 95.0 and frames_95 >= 2:
                cv2.putText(frame, f"95%+: {frames_95} frames - MADE", 
                           (x_offset + 10, y_offset + 100), font, font_scale, (0, 255, 0), thickness)
            elif max_overlap >= 95.0:
                cv2.putText(frame, f"95%+: {frames_95} frames - NEED 2+", 
                           (x_offset + 10, y_offset + 100), font, font_scale, (255, 165, 0), thickness)
            else:
                cv2.putText(frame, f"MAX: {max_overlap:.0f}% - MISS", 
                           (x_offset + 10, y_offset + 100), font, font_scale, (0, 0, 255), thickness)
        elif self.current_overlap_percentage > 0:
            if self.current_overlap_percentage >= 95.0:
                cv2.putText(frame, f"OVERLAP: {self.current_overlap_percentage:.0f}% - MADE", 
                           (x_offset + 10, y_offset + 80), font, font_scale, (0, 255, 0), thickness)
            else:
                cv2.putText(frame, f"OVERLAP: {self.current_overlap_percentage:.0f}% - MISS", 
                           (x_offset + 10, y_offset + 80), font, font_scale, (0, 0, 255), thickness)
                   
    def _draw_stats_overlay(self, frame):
        """Compatibility method - delegates to clean overlay"""
        self._draw_clean_stats_overlay(frame)
                       
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