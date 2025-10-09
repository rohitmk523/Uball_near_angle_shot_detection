#!/usr/bin/env python3
"""
Enhanced Shot Detection Module

This module integrates trajectory analysis and multi-frame context analysis
to provide improved basketball shot detection with higher accuracy.
"""

import cv2
import numpy as np
import json
import time
import math
from datetime import datetime
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple

from trajectory_analysis import (
    EnhancedTrajectoryAnalyzer, 
    MultiFrameContextAnalyzer, 
    PositionData,
    TrajectoryAnalysis
)

class EnhancedShotSequence:
    """Enhanced shot sequence tracking with trajectory analysis"""
    
    def __init__(self, initial_position: PositionData, sequence_id: int):
        self.sequence_id = sequence_id
        self.positions = deque(maxlen=50)  # Store position history
        self.overlap_frames = []  # Frames with hoop overlap
        self.start_time = initial_position.timestamp
        self.status = "active"  # active, analyzing, completed
        
        # Add initial position
        self.positions.append(initial_position)
        
        # Analysis components
        self.trajectory_analyzer = EnhancedTrajectoryAnalyzer()
        self.context_analyzer = MultiFrameContextAnalyzer()
        
        # Analysis results
        self.trajectory_analysis: Optional[TrajectoryAnalysis] = None
        self.context_analysis: Optional[Dict[str, Any]] = None
        self.final_outcome: Optional[str] = None
        self.confidence_score: float = 0.0
        
    def add_position(self, position_data: PositionData):
        """Add new position data to sequence"""
        self.positions.append(position_data)
        
        # If this position has overlap, add to overlap frames
        if position_data.overlap_percentage > 0:
            self.overlap_frames.append(position_data)
    
    def analyze_sequence(self, hoop_center: Tuple[int, int]) -> Dict[str, Any]:
        """Perform comprehensive sequence analysis"""
        if len(self.positions) < 3:
            return self._insufficient_data_result()
        
        # Convert positions to list for analysis
        position_list = list(self.positions)
        
        # Perform trajectory analysis
        self.trajectory_analysis = self.trajectory_analyzer.analyze_trajectory(position_list)
        
        # Perform context analysis
        overlap_start_idx = self._find_overlap_start_index(position_list)
        overlap_end_idx = self._find_overlap_end_index(position_list)
        
        if overlap_start_idx >= 0 and overlap_end_idx >= 0:
            pre_frames = position_list[:overlap_start_idx] if overlap_start_idx > 0 else []
            shot_frames = position_list[overlap_start_idx:overlap_end_idx+1]
            post_frames = position_list[overlap_end_idx+1:] if overlap_end_idx < len(position_list)-1 else []
            
            self.context_analysis = self.context_analyzer.analyze_shot_context(
                pre_frames, shot_frames, post_frames, hoop_center
            )
        else:
            # No clear overlap detected, analyze full sequence
            mid_point = len(position_list) // 2
            pre_frames = position_list[:mid_point]
            shot_frames = self.overlap_frames if self.overlap_frames else position_list[mid_point-1:mid_point+1]
            post_frames = position_list[mid_point:]
            
            self.context_analysis = self.context_analyzer.analyze_shot_context(
                pre_frames, shot_frames, post_frames, hoop_center
            )
        
        # Determine final outcome
        self.final_outcome, self.confidence_score = self._determine_enhanced_outcome()
        
        return self._create_analysis_result()
    
    def _find_overlap_start_index(self, positions: List[PositionData]) -> int:
        """Find index where overlap with hoop starts"""
        for i, pos in enumerate(positions):
            if pos.overlap_percentage > 0:
                return i
        return -1
    
    def _find_overlap_end_index(self, positions: List[PositionData]) -> int:
        """Find index where overlap with hoop ends"""
        for i in range(len(positions)-1, -1, -1):
            if positions[i].overlap_percentage > 0:
                return i
        return -1
    
    def _determine_enhanced_outcome(self) -> Tuple[str, float]:
        """Determine shot outcome using enhanced analysis"""
        if not self.trajectory_analysis or not self.context_analysis:
            return "undetermined", 0.0
        
        # Get analysis components
        traj = self.trajectory_analysis
        context = self.context_analysis
        
        # Start with base confidence from overlap
        max_overlap = max([pos.overlap_percentage for pos in self.overlap_frames]) if self.overlap_frames else 0
        base_confidence = max_overlap / 100.0
        
        # Enhanced decision logic
        
        # 1. Check for clear made shot patterns
        if self._is_clear_made_shot(traj, context, max_overlap):
            confidence = min(0.95, base_confidence + 0.3)
            return "made", confidence
        
        # 2. Check for clear miss patterns
        if self._is_clear_miss(traj, context, max_overlap):
            confidence = max(0.8, 1.0 - base_confidence)
            return "missed", confidence
        
        # 3. Check for rim bounce (false positive)
        if self._is_rim_bounce(traj, context):
            confidence = 0.85
            return "missed", confidence
        
        # 4. Check for fast swoosh (false negative)
        if self._is_fast_swoosh(traj, context, max_overlap):
            confidence = 0.8
            return "made", confidence
        
        # 5. Fallback to traditional overlap-based decision with enhanced logic
        return self._fallback_decision(max_overlap, traj, context)
    
    def _is_clear_made_shot(self, traj: TrajectoryAnalysis, context: Dict[str, Any], max_overlap: float) -> bool:
        """Check if this is clearly a made shot"""
        # High overlap with good trajectory
        if max_overlap >= 100.0 and traj.direction_consistency > 0.7:
            return True
        
        # Clean downward motion with good overlap
        if traj.shows_clean_downward_motion and max_overlap >= 95.0:
            return True
        
        # Context indicates clean made shot
        if context.get("overall_pattern") == "clean_made_shot":
            return True
        
        # Multiple perfect overlap frames with consistent trajectory
        perfect_frames = len([pos for pos in self.overlap_frames if pos.overlap_percentage >= 100.0])
        if perfect_frames >= 3 and traj.direction_consistency > 0.6:
            return True
        
        return False
    
    def _is_clear_miss(self, traj: TrajectoryAnalysis, context: Dict[str, Any], max_overlap: float) -> bool:
        """Check if this is clearly a missed shot"""
        # Very low overlap
        if max_overlap < 50.0:
            return True
        
        # Context indicates clear miss
        if context.get("overall_pattern") == "clear_miss":
            return True
        
        # Poor trajectory with low overlap
        if max_overlap < 80.0 and traj.direction_consistency < 0.4:
            return True
        
        return False
    
    def _is_rim_bounce(self, traj: TrajectoryAnalysis, context: Dict[str, Any]) -> bool:
        """Check if this is a rim bounce (false positive for made shot)"""
        # Trajectory shows upward bounce
        if traj.has_upward_bounce:
            return True
        
        # Context indicates rim bounce
        if context.get("overall_pattern") == "rim_bounce":
            return True
        
        # Ball reappears above rim level after overlap
        exit_analysis = context.get("exit_analysis", {})
        if (exit_analysis.get("type") == "ball_reappeared" and 
            exit_analysis.get("moved_upward", False)):
            return True
        
        return False
    
    def _is_fast_swoosh(self, traj: TrajectoryAnalysis, context: Dict[str, Any], max_overlap: float) -> bool:
        """Check if this is a fast swoosh (false negative for made shot)"""
        # Context indicates fast swoosh
        if context.get("overall_pattern") == "fast_swoosh":
            return True
        
        # Good overlap but very short sequence with clean trajectory
        if (max_overlap >= 90.0 and 
            len(self.overlap_frames) <= 3 and 
            traj.shows_clean_downward_motion):
            return True
        
        # High speed downward motion with good overlap
        if (traj.average_velocity.dy > 50 and  # Fast downward
            max_overlap >= 85.0 and
            traj.direction_consistency > 0.6):
            return True
        
        return False
    
    def _fallback_decision(self, max_overlap: float, traj: TrajectoryAnalysis, context: Dict[str, Any]) -> Tuple[str, float]:
        """Fallback decision logic with enhanced considerations"""
        # Base decision on overlap with trajectory adjustments
        if max_overlap >= 95.0:
            # High overlap - check trajectory quality
            if traj.direction_consistency > 0.5 and not traj.has_upward_bounce:
                return "made", 0.85
            else:
                # High overlap but poor trajectory - likely rim bounce
                return "missed", 0.75
        
        elif max_overlap >= 80.0:
            # Medium overlap - trajectory is crucial
            if traj.shows_clean_downward_motion and traj.direction_consistency > 0.6:
                return "made", 0.7
            else:
                return "missed", 0.7
        
        else:
            # Low overlap - likely miss unless trajectory suggests otherwise
            if traj.shows_clean_downward_motion and len(self.overlap_frames) >= 2:
                return "made", 0.6  # Fast swoosh case
            else:
                return "missed", 0.8
    
    def _create_analysis_result(self) -> Dict[str, Any]:
        """Create comprehensive analysis result"""
        max_overlap = max([pos.overlap_percentage for pos in self.overlap_frames]) if self.overlap_frames else 0
        
        return {
            "sequence_id": self.sequence_id,
            "outcome": self.final_outcome,
            "confidence_score": self.confidence_score,
            "max_overlap_percentage": max_overlap,
            "overlap_frame_count": len(self.overlap_frames),
            "total_frame_count": len(self.positions),
            "sequence_duration": self.positions[-1].timestamp - self.positions[0].timestamp,
            "trajectory_analysis": {
                "direction_consistency": self.trajectory_analysis.direction_consistency,
                "has_upward_bounce": self.trajectory_analysis.has_upward_bounce,
                "shows_clean_downward_motion": self.trajectory_analysis.shows_clean_downward_motion,
                "trajectory_smoothness": self.trajectory_analysis.trajectory_smoothness,
                "velocity_pattern": self.trajectory_analysis.velocity_pattern
            },
            "context_analysis": self.context_analysis,
            "analysis_method": "enhanced_trajectory_context"
        }
    
    def _insufficient_data_result(self) -> Dict[str, Any]:
        """Return result for insufficient data"""
        return {
            "sequence_id": self.sequence_id,
            "outcome": "undetermined",
            "confidence_score": 0.0,
            "max_overlap_percentage": 0.0,
            "overlap_frame_count": 0,
            "total_frame_count": len(self.positions),
            "sequence_duration": 0.0,
            "analysis_method": "insufficient_data"
        }

class EnhancedShotAnalyzer:
    """Enhanced shot analyzer with trajectory and context analysis"""
    
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.75):
        """Initialize enhanced shot analyzer"""
        # Initialize base YOLO model
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Enable GPU acceleration
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
        
        self.model.to(self.device)
        
        # Detection thresholds
        self.basketball_confidence = 0.35
        self.hoop_confidence = 0.5
        self.confidence_threshold = confidence_threshold
        
        # Hoop tracking
        self.hoop_center = None
        self.hoop_bbox = None
        self.hoop_area = None
        self.hoop_size = None
        
        # Enhanced sequence tracking
        self.active_sequences: Dict[int, EnhancedShotSequence] = {}
        self.sequence_id_counter = 0
        self.sequence_timeout = 2.0  # seconds
        self.max_active_sequences = 3
        self.min_shot_interval = 2.0  # Minimum 2 seconds between shots
        self.last_shot_time = 0.0  # Track last shot to prevent duplicates
        
        # Ball tracking for sequence association
        self.ball_position_history = deque(maxlen=10)
        self.position_match_threshold = 100  # pixels
        
        # Statistics and logging
        self.stats = {
            'made_shots': 0,
            'missed_shots': 0,
            'total_shots': 0,
            'undetermined_shots': 0
        }
        
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
        """Enhanced shot tracking with trajectory analysis"""
        current_time = time.time()
        
        # Update hoop position
        self._update_hoop_tracking(detections)
        
        # Process basketball detections
        basketball_positions = []
        for ball in detections['basketball']:
            if self._is_valid_ball_detection(ball):
                ball_center = ball['center']
                ball_bbox = ball['bbox']
                
                # Calculate overlap with hoop if hoop detected
                overlap_percentage = 0.0
                if self.hoop_bbox:
                    overlap_percentage = self._calculate_overlap(ball_bbox, self.hoop_bbox)
                
                # Calculate size ratio
                ball_size = max(ball_bbox[2] - ball_bbox[0], ball_bbox[3] - ball_bbox[1])
                size_ratio = ball_size / self.hoop_size if self.hoop_size else 0.0
                
                # Create position data
                position_data = PositionData(
                    position=tuple(ball_center),
                    timestamp=current_time,
                    confidence=ball['confidence'],
                    size_ratio=size_ratio,
                    overlap_percentage=overlap_percentage
                )
                
                basketball_positions.append(position_data)
        
        # Update ball position history
        if basketball_positions:
            self.ball_position_history.extend(basketball_positions)
        
        # Update active sequences
        self._update_active_sequences(basketball_positions, current_time)
        
        # Clean up timed out sequences
        self._cleanup_sequences(current_time)
    
    def _update_hoop_tracking(self, detections):
        """Update hoop position and area"""
        if detections['basketball_hoop']:
            hoop = detections['basketball_hoop'][0]  # Use first detected hoop
            self.hoop_center = tuple(hoop['center'])
            self.hoop_bbox = hoop['bbox']
            self.hoop_size = max(self.hoop_bbox[2] - self.hoop_bbox[0], 
                               self.hoop_bbox[3] - self.hoop_bbox[1])
            
            # Define shooting area around hoop
            margin = int(self.hoop_size * 0.8)
            x1 = self.hoop_bbox[0] - margin
            y1 = self.hoop_bbox[1] - margin
            x2 = self.hoop_bbox[2] + margin
            y2 = self.hoop_bbox[3] + margin
            self.hoop_area = [x1, y1, x2, y2]
    
    def _is_valid_ball_detection(self, ball):
        """Validate ball detection for shot tracking"""
        if not self.hoop_area or not self.hoop_center:
            return False
        
        ball_center = ball['center']
        ball_bbox = ball['bbox']
        
        # Check if ball is in shooting zone
        in_shooting_zone = (self.hoop_area[0] <= ball_center[0] <= self.hoop_area[2] and 
                          self.hoop_area[1] <= ball_center[1] <= self.hoop_area[3])
        
        if not in_shooting_zone:
            return False
        
        # Validate size and position
        ball_size = max(ball_bbox[2] - ball_bbox[0], ball_bbox[3] - ball_bbox[1])
        size_ratio = ball_size / self.hoop_size if self.hoop_size else 0.0
        
        # More permissive size range for enhanced tracking
        valid_size = 0.25 <= size_ratio <= 0.9
        
        # Position validation
        vertical_distance = ball_center[1] - self.hoop_center[1]
        valid_position = -150 <= vertical_distance <= 150  # Expanded range
        
        return valid_size and valid_position
    
    def _calculate_overlap(self, ball_bbox, hoop_bbox):
        """Calculate overlap percentage between ball and hoop bounding boxes"""
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox
        hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_bbox
        
        # Calculate intersection
        overlap_x = max(0, min(ball_x2, hoop_x2) - max(ball_x1, hoop_x1))
        overlap_y = max(0, min(ball_y2, hoop_y2) - max(ball_y1, hoop_y1))
        intersection_area = overlap_x * overlap_y
        
        # Calculate ball area
        ball_area = (ball_x2 - ball_x1) * (ball_y2 - ball_y1)
        
        if ball_area > 0:
            return (intersection_area / ball_area) * 100
        return 0
    
    def _update_active_sequences(self, basketball_positions: List[PositionData], current_time: float):
        """Update active shot sequences with new position data"""
        for position_data in basketball_positions:
            # Try to associate with existing sequence
            associated_sequence = self._find_associated_sequence(position_data)
            
            if associated_sequence:
                # Add to existing sequence
                associated_sequence.add_position(position_data)
            else:
                # Create new sequence if we have room and ball has overlap
                if (len(self.active_sequences) < self.max_active_sequences and 
                    position_data.overlap_percentage > 0):
                    self._create_new_sequence(position_data)
    
    def _find_associated_sequence(self, position_data: PositionData) -> Optional[EnhancedShotSequence]:
        """Find existing sequence that this position should be associated with"""
        best_sequence = None
        min_distance = float('inf')
        
        for sequence in self.active_sequences.values():
            if sequence.status != "active":
                continue
            
            # Get last position from sequence
            if sequence.positions:
                last_pos = sequence.positions[-1].position
                distance = math.sqrt((position_data.position[0] - last_pos[0])**2 + 
                                   (position_data.position[1] - last_pos[1])**2)
                
                if distance < self.position_match_threshold and distance < min_distance:
                    min_distance = distance
                    best_sequence = sequence
        
        return best_sequence
    
    def _create_new_sequence(self, position_data: PositionData):
        """Create new shot sequence"""
        sequence_id = self.sequence_id_counter
        self.sequence_id_counter += 1
        
        sequence = EnhancedShotSequence(position_data, sequence_id)
        self.active_sequences[sequence_id] = sequence
    
    def _cleanup_sequences(self, current_time: float):
        """Clean up timed out sequences and finalize completed ones"""
        sequences_to_finalize = []
        
        for sequence_id, sequence in list(self.active_sequences.items()):
            # Check for timeout
            time_since_last_update = current_time - sequence.positions[-1].timestamp
            
            if time_since_last_update > self.sequence_timeout:
                sequences_to_finalize.append(sequence_id)
        
        # Finalize sequences
        for sequence_id in sequences_to_finalize:
            self._finalize_sequence(sequence_id)
    
    def _finalize_sequence(self, sequence_id: int):
        """Finalize and log shot sequence"""
        if sequence_id not in self.active_sequences:
            return
        
        sequence = self.active_sequences[sequence_id]
        
        # Only analyze sequences with sufficient data
        if len(sequence.positions) >= 3 and len(sequence.overlap_frames) > 0:
            # Check for minimum shot interval before processing
            sequence_time = sequence.positions[-1].timestamp
            if sequence_time - self.last_shot_time < self.min_shot_interval:
                print(f"Skipping sequence {sequence_id} - too close to previous shot")
                del self.active_sequences[sequence_id]
                return
            
            # Perform enhanced analysis
            if self.hoop_center:
                analysis_result = sequence.analyze_sequence(self.hoop_center)
                
                # Update statistics
                outcome = analysis_result["outcome"]
                if outcome == "made":
                    self.stats['made_shots'] += 1
                elif outcome == "missed":
                    self.stats['missed_shots'] += 1
                else:
                    self.stats['undetermined_shots'] += 1
                
                self.stats['total_shots'] += 1
                
                # Log the shot
                self._log_shot(analysis_result, sequence)
        
        # Remove from active sequences
        del self.active_sequences[sequence_id]
    
    def _log_shot(self, analysis_result: Dict[str, Any], sequence: EnhancedShotSequence):
        """Log shot with comprehensive data"""
        # Get representative frame data
        max_overlap_frame = None
        if sequence.overlap_frames:
            max_overlap_frame = max(sequence.overlap_frames, 
                                  key=lambda x: x.overlap_percentage)
        else:
            max_overlap_frame = sequence.positions[-1]
        
        # Check minimum time interval to prevent duplicate shots
        shot_time = max_overlap_frame.timestamp
        if shot_time - self.last_shot_time < self.min_shot_interval:
            print(f"Skipping duplicate shot (interval: {shot_time - self.last_shot_time:.2f}s)")
            return
        
        self.last_shot_time = shot_time
        
        # Create timestamp from frame time for consistency
        shot_datetime = datetime.fromtimestamp(shot_time)
        
        shot_data = {
            'frame_time': shot_time,
            'timestamp': shot_datetime.isoformat(),
            'outcome': analysis_result["outcome"],
            'confidence': max_overlap_frame.confidence,
            'enhanced_confidence': analysis_result["confidence_score"],
            'confidence_threshold': self.confidence_threshold,
            'max_overlap_percentage': analysis_result["max_overlap_percentage"],
            'overlap_frame_count': analysis_result["overlap_frame_count"],
            'total_frame_count': analysis_result["total_frame_count"],
            'sequence_duration': analysis_result["sequence_duration"],
            'ball_position': max_overlap_frame.position,
            'hoop_center': self.hoop_center,
            'size_ratio': max_overlap_frame.size_ratio,
            'trajectory_analysis': analysis_result["trajectory_analysis"],
            'context_analysis': analysis_result["context_analysis"],
            'detection_method': analysis_result["analysis_method"]
        }
        
        self.shot_log.append(shot_data)
    
    def draw_overlay(self, frame, detections):
        """Draw enhanced detection overlay"""
        overlay = frame.copy()
        
        # Draw hoop detection
        for hoop in detections['basketball_hoop']:
            x1, y1, x2, y2 = hoop['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if self.hoop_area:
                cv2.rectangle(overlay, (self.hoop_area[0], self.hoop_area[1]), 
                            (self.hoop_area[2], self.hoop_area[3]), (0, 255, 0), 1)
        
        # Draw basketball detection
        for ball in detections['basketball']:
            x1, y1, x2, y2 = ball['bbox']
            center = ball['center']
            confidence = ball['confidence']
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            label = f"Ball: {confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Show overlap if in shooting zone
            if self.hoop_bbox and self._is_valid_ball_detection(ball):
                overlap = self._calculate_overlap(ball['bbox'], self.hoop_bbox)
                if overlap > 0:
                    color = (0, 255, 0) if overlap >= 95 else (0, 0, 255)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(overlay, f"{overlap:.0f}%", (x1, y2 + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw sequence information
        self._draw_sequence_info(overlay)
        
        # Draw statistics
        self._draw_enhanced_stats(overlay)
        
        return overlay
    
    def _draw_sequence_info(self, frame):
        """Draw active sequence information"""
        y_offset = 100
        for sequence_id, sequence in self.active_sequences.items():
            if sequence.status == "active":
                text = f"Seq {sequence_id}: {len(sequence.overlap_frames)} overlaps"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
    
    def _draw_enhanced_stats(self, frame):
        """Draw enhanced statistics overlay"""
        height, width = frame.shape[:2]
        x_offset = width - 250
        y_offset = 20
        
        # Statistics
        cv2.putText(frame, f"MADE: {self.stats['made_shots']}", 
                   (x_offset, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MISSED: {self.stats['missed_shots']}", 
                   (x_offset, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        cv2.putText(frame, f"TOTAL: {self.stats['total_shots']}", 
                   (x_offset, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show accuracy if enough shots
        if self.stats['total_shots'] > 0:
            accuracy = (self.stats['made_shots'] + self.stats['missed_shots']) / self.stats['total_shots'] * 100
            cv2.putText(frame, f"ENHANCED MODE", 
                       (x_offset, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def save_session_data(self, filename=None):
        """Save enhanced session data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_shot_session_{timestamp}.json"
        
        session_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'total_duration': time.time() - self.session_start.timestamp(),
                'analysis_method': 'enhanced_trajectory_context'
            },
            'statistics': self.stats,
            'shots': self.shot_log
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✓ Enhanced session data saved to: {filename}")
        return filename