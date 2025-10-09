#!/usr/bin/env python3
"""
Enhanced Trajectory Analysis Module

This module provides advanced trajectory analysis for basketball shot detection,
including velocity analysis, direction consistency checks, and multi-frame context analysis.
"""

import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class PositionData:
    """Data structure for ball position with metadata"""
    position: Tuple[int, int]
    timestamp: float
    confidence: float
    size_ratio: float
    overlap_percentage: float = 0.0

@dataclass
class VelocityVector:
    """Velocity vector with magnitude and direction"""
    dx: float
    dy: float
    magnitude: float
    angle: float  # in radians
    
    @classmethod
    def from_positions(cls, pos1: Tuple[int, int], pos2: Tuple[int, int], dt: float):
        """Create velocity vector from two positions and time difference"""
        if dt <= 0:
            return cls(0, 0, 0, 0)
        
        dx = (pos2[0] - pos1[0]) / dt
        dy = (pos2[1] - pos1[1]) / dt
        magnitude = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        
        return cls(dx, dy, magnitude, angle)

@dataclass
class TrajectoryAnalysis:
    """Results of trajectory analysis"""
    direction_consistency: float  # 0-1, higher = more consistent
    velocity_pattern: Dict[str, Any]
    has_upward_bounce: bool
    shows_clean_downward_motion: bool
    average_velocity: VelocityVector
    velocity_changes: List[float]
    trajectory_smoothness: float

class EnhancedTrajectoryAnalyzer:
    """Enhanced trajectory analyzer with physics-based shot detection"""
    
    def __init__(self, max_history=50, min_trajectory_points=5):
        self.max_history = max_history
        self.min_trajectory_points = min_trajectory_points
        
        # Trajectory analysis parameters
        self.direction_consistency_threshold = 0.6
        self.bounce_detection_threshold = 50  # pixels
        self.min_downward_velocity = 10  # pixels/second
        self.smoothness_threshold = 0.7
        
        # Velocity analysis parameters
        self.velocity_change_threshold = 100  # pixels/second
        self.acceleration_threshold = 500  # pixels/second²
        
    def analyze_trajectory(self, positions: List[PositionData]) -> TrajectoryAnalysis:
        """Perform comprehensive trajectory analysis"""
        if len(positions) < self.min_trajectory_points:
            return self._default_analysis()
        
        # Calculate velocity vectors
        velocity_vectors = self._calculate_velocity_vectors(positions)
        
        # Analyze direction consistency
        direction_consistency = self._calculate_direction_consistency(velocity_vectors)
        
        # Detect upward bounces
        has_upward_bounce = self._detect_upward_bounce(velocity_vectors, positions)
        
        # Check for clean downward motion
        shows_clean_downward_motion = self._check_clean_downward_motion(velocity_vectors)
        
        # Calculate velocity pattern
        velocity_pattern = self._analyze_velocity_pattern(velocity_vectors, positions)
        
        # Calculate trajectory smoothness
        trajectory_smoothness = self._calculate_trajectory_smoothness(positions)
        
        # Calculate average velocity
        average_velocity = self._calculate_average_velocity(velocity_vectors)
        
        # Calculate velocity changes
        velocity_changes = self._calculate_velocity_changes(velocity_vectors)
        
        return TrajectoryAnalysis(
            direction_consistency=direction_consistency,
            velocity_pattern=velocity_pattern,
            has_upward_bounce=has_upward_bounce,
            shows_clean_downward_motion=shows_clean_downward_motion,
            average_velocity=average_velocity,
            velocity_changes=velocity_changes,
            trajectory_smoothness=trajectory_smoothness
        )
    
    def _calculate_velocity_vectors(self, positions: List[PositionData]) -> List[VelocityVector]:
        """Calculate velocity vectors between consecutive positions"""
        vectors = []
        
        for i in range(1, len(positions)):
            pos1 = positions[i-1]
            pos2 = positions[i]
            dt = pos2.timestamp - pos1.timestamp
            
            if dt > 0:
                vector = VelocityVector.from_positions(pos1.position, pos2.position, dt)
                vectors.append(vector)
        
        return vectors
    
    def _calculate_direction_consistency(self, velocity_vectors: List[VelocityVector]) -> float:
        """Calculate how consistent the trajectory direction is (0-1)"""
        if len(velocity_vectors) < 2:
            return 0.0
        
        # Calculate angle differences between consecutive velocity vectors
        angle_differences = []
        for i in range(1, len(velocity_vectors)):
            angle_diff = abs(velocity_vectors[i].angle - velocity_vectors[i-1].angle)
            # Normalize to [0, π]
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            angle_differences.append(angle_diff)
        
        if not angle_differences:
            return 0.0
        
        # Convert to consistency score (lower angle differences = higher consistency)
        avg_angle_difference = sum(angle_differences) / len(angle_differences)
        consistency = max(0.0, 1.0 - (avg_angle_difference / math.pi))
        
        return consistency
    
    def _detect_upward_bounce(self, velocity_vectors: List[VelocityVector], positions: List[PositionData]) -> bool:
        """Detect if ball shows upward bounce pattern indicating rim bounce"""
        if len(velocity_vectors) < 3:
            return False
        
        # Look for pattern: downward motion -> upward motion
        for i in range(1, len(velocity_vectors)):
            prev_vel = velocity_vectors[i-1]
            curr_vel = velocity_vectors[i]
            
            # Check if velocity direction changed from downward to upward
            if prev_vel.dy > 0 and curr_vel.dy < 0:  # dy > 0 is downward in image coordinates
                # Check if the upward velocity is significant (indicating bounce)
                upward_speed = abs(curr_vel.dy)
                if upward_speed > self.bounce_detection_threshold:
                    return True
        
        return False
    
    def _check_clean_downward_motion(self, velocity_vectors: List[VelocityVector]) -> bool:
        """Check if trajectory shows clean downward motion (good for made shots)"""
        if len(velocity_vectors) < 2:
            return False
        
        # Count frames with consistent downward motion
        downward_frames = 0
        total_frames = len(velocity_vectors)
        
        for vector in velocity_vectors:
            if vector.dy > self.min_downward_velocity:  # Downward motion
                downward_frames += 1
        
        # Require majority of frames to show downward motion
        downward_ratio = downward_frames / total_frames
        return downward_ratio >= 0.7
    
    def _analyze_velocity_pattern(self, velocity_vectors: List[VelocityVector], positions: List[PositionData]) -> Dict[str, Any]:
        """Analyze overall velocity pattern"""
        if not velocity_vectors:
            return {"type": "insufficient_data"}
        
        # Calculate velocity statistics
        velocities = [v.magnitude for v in velocity_vectors]
        dy_values = [v.dy for v in velocity_vectors]
        
        avg_speed = sum(velocities) / len(velocities)
        max_speed = max(velocities)
        min_speed = min(velocities)
        
        # Analyze vertical motion pattern
        downward_count = sum(1 for dy in dy_values if dy > 0)
        upward_count = sum(1 for dy in dy_values if dy < 0)
        
        # Determine pattern type
        if downward_count > upward_count * 2:
            pattern_type = "predominantly_downward"
        elif upward_count > downward_count * 2:
            pattern_type = "predominantly_upward"
        else:
            pattern_type = "mixed_motion"
        
        return {
            "type": pattern_type,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "downward_frames": downward_count,
            "upward_frames": upward_count,
            "total_frames": len(velocity_vectors)
        }
    
    def _calculate_trajectory_smoothness(self, positions: List[PositionData]) -> float:
        """Calculate how smooth the trajectory is (0-1, higher = smoother)"""
        if len(positions) < 3:
            return 0.0
        
        # Calculate second derivatives (acceleration) to measure smoothness
        accelerations = []
        
        for i in range(2, len(positions)):
            pos0 = positions[i-2].position
            pos1 = positions[i-1].position
            pos2 = positions[i].position
            
            # Calculate acceleration in x and y
            dt = positions[i].timestamp - positions[i-2].timestamp
            if dt > 0:
                ax = (pos2[0] - 2*pos1[0] + pos0[0]) / (dt**2)
                ay = (pos2[1] - 2*pos1[1] + pos0[1]) / (dt**2)
                acceleration_magnitude = math.sqrt(ax**2 + ay**2)
                accelerations.append(acceleration_magnitude)
        
        if not accelerations:
            return 0.0
        
        # Calculate smoothness (lower acceleration changes = smoother)
        avg_acceleration = sum(accelerations) / len(accelerations)
        smoothness = max(0.0, 1.0 - (avg_acceleration / self.acceleration_threshold))
        
        return min(1.0, smoothness)
    
    def _calculate_average_velocity(self, velocity_vectors: List[VelocityVector]) -> VelocityVector:
        """Calculate average velocity vector"""
        if not velocity_vectors:
            return VelocityVector(0, 0, 0, 0)
        
        avg_dx = sum(v.dx for v in velocity_vectors) / len(velocity_vectors)
        avg_dy = sum(v.dy for v in velocity_vectors) / len(velocity_vectors)
        avg_magnitude = math.sqrt(avg_dx**2 + avg_dy**2)
        avg_angle = math.atan2(avg_dy, avg_dx)
        
        return VelocityVector(avg_dx, avg_dy, avg_magnitude, avg_angle)
    
    def _calculate_velocity_changes(self, velocity_vectors: List[VelocityVector]) -> List[float]:
        """Calculate velocity magnitude changes between consecutive frames"""
        if len(velocity_vectors) < 2:
            return []
        
        changes = []
        for i in range(1, len(velocity_vectors)):
            change = abs(velocity_vectors[i].magnitude - velocity_vectors[i-1].magnitude)
            changes.append(change)
        
        return changes
    
    def _default_analysis(self) -> TrajectoryAnalysis:
        """Return default analysis for insufficient data"""
        return TrajectoryAnalysis(
            direction_consistency=0.0,
            velocity_pattern={"type": "insufficient_data"},
            has_upward_bounce=False,
            shows_clean_downward_motion=False,
            average_velocity=VelocityVector(0, 0, 0, 0),
            velocity_changes=[],
            trajectory_smoothness=0.0
        )

class MultiFrameContextAnalyzer:
    """Analyzer for multi-frame context around shot detection"""
    
    def __init__(self, pre_frame_count=10, post_frame_count=10):
        self.pre_frame_count = pre_frame_count
        self.post_frame_count = post_frame_count
        
        # Context analysis parameters
        self.disappearance_threshold = 5  # frames without detection
        self.reappearance_threshold = 3   # frames to confirm reappearance
        self.position_change_threshold = 100  # pixels for significant movement
        
    def analyze_shot_context(self, 
                           pre_frames: List[PositionData], 
                           shot_frames: List[PositionData], 
                           post_frames: List[PositionData],
                           hoop_center: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze complete shot context including pre and post frames"""
        
        context = {
            "approach_analysis": self._analyze_approach(pre_frames, shot_frames, hoop_center),
            "shot_analysis": self._analyze_shot_sequence(shot_frames, hoop_center),
            "exit_analysis": self._analyze_exit(shot_frames, post_frames, hoop_center),
            "overall_pattern": None
        }
        
        # Determine overall pattern
        context["overall_pattern"] = self._determine_overall_pattern(context)
        
        return context
    
    def _analyze_approach(self, pre_frames: List[PositionData], shot_frames: List[PositionData], 
                         hoop_center: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze ball approach to hoop"""
        if not pre_frames:
            return {"type": "no_approach_data"}
        
        # Check if ball approaches from above
        last_pre_pos = pre_frames[-1].position
        first_shot_pos = shot_frames[0].position if shot_frames else last_pre_pos
        
        # Calculate approach angle
        dx = first_shot_pos[0] - last_pre_pos[0]
        dy = first_shot_pos[1] - last_pre_pos[1]
        approach_angle = math.atan2(dy, dx) if dx != 0 else 0
        
        # Check if approaching from above hoop
        approaching_from_above = last_pre_pos[1] < hoop_center[1]
        
        # Calculate distance to hoop over approach
        distances = []
        for frame in pre_frames[-5:]:  # Last 5 approach frames
            dist = math.sqrt((frame.position[0] - hoop_center[0])**2 + 
                           (frame.position[1] - hoop_center[1])**2)
            distances.append(dist)
        
        # Check if distance is decreasing (approaching hoop)
        decreasing_distance = len(distances) > 1 and distances[-1] < distances[0]
        
        return {
            "type": "normal_approach",
            "approach_angle": approach_angle,
            "approaching_from_above": approaching_from_above,
            "decreasing_distance": decreasing_distance,
            "final_approach_distance": distances[-1] if distances else float('inf')
        }
    
    def _analyze_shot_sequence(self, shot_frames: List[PositionData], 
                              hoop_center: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze the shot sequence frames"""
        if not shot_frames:
            return {"type": "no_shot_data"}
        
        # Calculate trajectory through hoop area
        min_distance = float('inf')
        max_overlap = 0.0
        
        for frame in shot_frames:
            dist = math.sqrt((frame.position[0] - hoop_center[0])**2 + 
                           (frame.position[1] - hoop_center[1])**2)
            min_distance = min(min_distance, dist)
            max_overlap = max(max_overlap, frame.overlap_percentage)
        
        # Check progression through hoop (Y coordinate should generally increase for made shots)
        y_progression = [frame.position[1] for frame in shot_frames]
        generally_downward = len(y_progression) > 1 and y_progression[-1] > y_progression[0]
        
        return {
            "type": "shot_sequence",
            "min_distance_to_hoop": min_distance,
            "max_overlap": max_overlap,
            "frame_count": len(shot_frames),
            "generally_downward": generally_downward,
            "duration": shot_frames[-1].timestamp - shot_frames[0].timestamp if len(shot_frames) > 1 else 0
        }
    
    def _analyze_exit(self, shot_frames: List[PositionData], post_frames: List[PositionData], 
                     hoop_center: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze ball behavior after shot sequence"""
        if not shot_frames:
            return {"type": "no_shot_reference"}
        
        last_shot_pos = shot_frames[-1].position
        
        if not post_frames:
            return {
                "type": "ball_disappeared",
                "last_position": last_shot_pos,
                "disappeared_below_hoop": last_shot_pos[1] > hoop_center[1]
            }
        
        # Check if ball reappears and where
        first_post_pos = post_frames[0].position
        
        # Calculate position change
        position_change = math.sqrt((first_post_pos[0] - last_shot_pos[0])**2 + 
                                  (first_post_pos[1] - last_shot_pos[1])**2)
        
        # Check if ball moved significantly upward (bounce indication)
        moved_upward = first_post_pos[1] < last_shot_pos[1]
        upward_distance = last_shot_pos[1] - first_post_pos[1] if moved_upward else 0
        
        # Check if ball reappears near rim level (bounce indication)
        near_rim_level = abs(first_post_pos[1] - hoop_center[1]) < 50
        
        return {
            "type": "ball_reappeared",
            "position_change": position_change,
            "moved_upward": moved_upward,
            "upward_distance": upward_distance,
            "near_rim_level": near_rim_level,
            "reappearance_frames": len(post_frames)
        }
    
    def _determine_overall_pattern(self, context: Dict[str, Any]) -> str:
        """Determine overall shot pattern from context analysis"""
        approach = context["approach_analysis"]
        shot = context["shot_analysis"]
        exit = context["exit_analysis"]
        
        # Made shot pattern: good approach + clean shot + ball disappears below
        if (approach.get("approaching_from_above", False) and 
            shot.get("max_overlap", 0) >= 95 and
            exit.get("type") == "ball_disappeared" and
            exit.get("disappeared_below_hoop", False)):
            return "clean_made_shot"
        
        # Rim bounce pattern: ball reappears above or near rim
        if (exit.get("type") == "ball_reappeared" and
            (exit.get("moved_upward", False) or exit.get("near_rim_level", False))):
            return "rim_bounce"
        
        # Fast swoosh: high overlap but ball disappears quickly
        if (shot.get("max_overlap", 0) >= 90 and
            shot.get("frame_count", 0) <= 3 and
            exit.get("type") == "ball_disappeared"):
            return "fast_swoosh"
        
        # Miss pattern: low overlap or ball deflects away
        if shot.get("max_overlap", 0) < 80:
            return "clear_miss"
        
        return "undetermined"