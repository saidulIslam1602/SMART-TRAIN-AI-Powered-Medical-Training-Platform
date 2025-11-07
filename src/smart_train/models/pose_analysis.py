"""
Medical Pose Analysis Model for SMART-TRAIN platform.

This module provides advanced pose analysis for medical procedures
with focus on body alignment, technique assessment, and safety validation.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger = get_logger(__name__)


@dataclass
class PoseAnalysisResult:
    """Pose analysis result structure."""
    body_alignment_score: float
    arm_extension_score: float
    posture_stability_score: float
    technique_consistency_score: float
    safety_assessment_score: float
    overall_pose_score: float
    recommendations: List[str]
    pose_landmarks: np.ndarray
    confidence_scores: Dict[str, float]


class MedicalPoseAnalysisModel(BaseModel):
    """
    Medical pose analysis model for technique assessment.
    
    This model analyzes body positioning and movement patterns
    for medical procedures with real-time feedback capabilities.
    """
    
    def __init__(self, model_version: str = "2.0.0"):
        super().__init__("MedicalPoseAnalysis", model_version)
        
        self.audit_manager = AuditTrailManager()
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Medical procedure thresholds
        self.medical_thresholds = {
            'cpr': {
                'arm_extension_min': 0.85,
                'body_alignment_min': 0.8,
                'shoulder_hand_alignment': 0.9,
                'posture_stability_min': 0.75
            },
            'airway_management': {
                'head_tilt_angle_range': (15, 30),
                'chin_lift_distance_range': (2, 4),
                'hand_position_precision': 0.9
            }
        }
        
        # Performance tracking
        self.analysis_history = []
        
        logger.info("Medical Pose Analysis Model initialized", model_version=model_version)
    
    def load_model(self, model_path: Optional[Any] = None) -> None:
        """Load pose analysis model (MediaPipe-based)."""
        self.is_loaded = True
        self.load_timestamp = time.time()
        
        logger.info("Medical Pose Analysis Model loaded")
        
        # Log model loading event
        self.audit_manager.log_event(
            event_type=AuditEventType.MODEL_OPERATION,
            description="Medical Pose Analysis Model loaded",
            severity=AuditSeverity.INFO
        )
    
    def predict(self, input_data: np.ndarray) -> ProcessingResult:
        """
        Analyze pose from video frame or pose landmarks.
        
        Args:
            input_data: Video frame (H, W, 3) or pose landmarks (33, 3)
            
        Returns:
            ProcessingResult with pose analysis
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            start_time = time.time()
            
            # Extract pose landmarks if input is video frame
            if len(input_data.shape) == 3:  # Video frame
                pose_landmarks = self._extract_pose_landmarks(input_data)
            else:  # Already pose landmarks
                pose_landmarks = input_data
            
            if pose_landmarks is None:
                return ProcessingResult(
                    success=False,
                    error_message="Failed to extract pose landmarks",
                    data={}
                )
            
            # Perform pose analysis
            analysis_result = self._analyze_medical_pose(pose_landmarks)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.inference_count += 1
            
            # Store analysis history
            self.analysis_history.append({
                'timestamp': start_time,
                'overall_score': analysis_result.overall_pose_score,
                'processing_time': processing_time
            })
            
            # Create result
            result = ProcessingResult(
                success=True,
                data={
                    'pose_analysis': analysis_result.__dict__,
                    'processing_time_ms': processing_time * 1000,
                    'model_version': self.model_version
                },
                metadata={
                    'model_id': self.model_id,
                    'landmarks_detected': len(pose_landmarks),
                    'processing_timestamp': start_time
                }
            )
            
            # Log analysis event
            self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                description="Medical pose analysis completed",
                severity=AuditSeverity.INFO,
                metadata={
                    'overall_pose_score': analysis_result.overall_pose_score,
                    'processing_time_ms': processing_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            logger.error("Medical pose analysis failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"Pose analysis failed: {e}",
                data={}
            )
    
    def _extract_pose_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks from video frame."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose_estimator.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks)
            
            return None
            
        except Exception as e:
            logger.error("Failed to extract pose landmarks", error=str(e))
            return None
    
    def _analyze_medical_pose(self, pose_landmarks: np.ndarray) -> PoseAnalysisResult:
        """Analyze pose for medical procedure technique."""
        # Calculate individual pose metrics
        body_alignment = self._calculate_body_alignment(pose_landmarks)
        arm_extension = self._calculate_arm_extension(pose_landmarks)
        posture_stability = self._calculate_posture_stability(pose_landmarks)
        technique_consistency = self._calculate_technique_consistency(pose_landmarks)
        safety_assessment = self._calculate_safety_assessment(pose_landmarks)
        
        # Calculate overall pose score
        scores = [body_alignment, arm_extension, posture_stability, 
                 technique_consistency, safety_assessment]
        overall_score = np.mean(scores)
        
        # Generate recommendations
        recommendations = self._generate_pose_recommendations(
            body_alignment, arm_extension, posture_stability, 
            technique_consistency, safety_assessment
        )
        
        # Calculate confidence scores
        confidence_scores = {
            'body_alignment': min(1.0, body_alignment + 0.1),
            'arm_extension': min(1.0, arm_extension + 0.1),
            'posture_stability': min(1.0, posture_stability + 0.1),
            'technique_consistency': min(1.0, technique_consistency + 0.1),
            'safety_assessment': min(1.0, safety_assessment + 0.1)
        }
        
        return PoseAnalysisResult(
            body_alignment_score=body_alignment,
            arm_extension_score=arm_extension,
            posture_stability_score=posture_stability,
            technique_consistency_score=technique_consistency,
            safety_assessment_score=safety_assessment,
            overall_pose_score=overall_score,
            recommendations=recommendations,
            pose_landmarks=pose_landmarks,
            confidence_scores=confidence_scores
        )
    
    def _calculate_body_alignment(self, landmarks: np.ndarray) -> float:
        """Calculate body alignment score for CPR positioning."""
        try:
            # Key landmarks for body alignment
            left_shoulder = landmarks[11]  # Left shoulder
            right_shoulder = landmarks[12]  # Right shoulder
            left_wrist = landmarks[15]  # Left wrist
            right_wrist = landmarks[16]  # Right wrist
            
            # Calculate shoulder center
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Calculate hand center
            hand_center = (left_wrist + right_wrist) / 2
            
            # Calculate alignment score based on vertical alignment
            vertical_alignment = abs(shoulder_center[0] - hand_center[0])
            alignment_score = max(0, 1.0 - vertical_alignment * 5)  # Scale factor
            
            return min(1.0, alignment_score)
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _calculate_arm_extension(self, landmarks: np.ndarray) -> float:
        """Calculate arm extension score for proper CPR technique."""
        try:
            # Key landmarks for arm extension
            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            
            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]
            
            # Calculate arm angles
            left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Ideal arm extension is close to 180 degrees (straight arms)
            left_extension = 1.0 - abs(180 - left_arm_angle) / 180
            right_extension = 1.0 - abs(180 - right_arm_angle) / 180
            
            return (left_extension + right_extension) / 2
            
        except Exception:
            return 0.5
    
    def _calculate_posture_stability(self, landmarks: np.ndarray) -> float:
        """Calculate posture stability score."""
        try:
            # Check hip and shoulder stability
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Calculate hip-shoulder alignment
            hip_center = (left_hip + right_hip) / 2
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Stability based on vertical alignment of torso
            torso_alignment = abs(hip_center[0] - shoulder_center[0])
            stability_score = max(0, 1.0 - torso_alignment * 3)
            
            return min(1.0, stability_score)
            
        except Exception:
            return 0.5
    
    def _calculate_technique_consistency(self, landmarks: np.ndarray) -> float:
        """Calculate technique consistency based on pose history."""
        if len(self.analysis_history) < 3:
            return 1.0  # Not enough history for consistency check
        
        try:
            # Get recent scores
            recent_scores = [entry['overall_score'] for entry in self.analysis_history[-5:]]
            
            # Calculate consistency as inverse of standard deviation
            consistency = 1.0 - min(1.0, np.std(recent_scores) * 2)
            
            return max(0.0, consistency)
            
        except Exception:
            return 0.5
    
    def _calculate_safety_assessment(self, landmarks: np.ndarray) -> float:
        """Calculate safety assessment score."""
        try:
            # Check for potentially unsafe positions
            safety_score = 1.0
            
            # Check head position (should not be too low)
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Head should be above shoulders for safety
            if nose[1] > shoulder_center[1]:  # Y coordinate increases downward
                safety_score -= 0.3
            
            # Check for extreme arm positions
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Arms should not be too far apart
            arm_distance = np.linalg.norm(left_wrist - right_wrist)
            if arm_distance > 0.3:  # Threshold for normalized coordinates
                safety_score -= 0.2
            
            return max(0.0, safety_score)
            
        except Exception:
            return 0.8  # Conservative safety score
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculate angle between three points."""
        try:
            # Vectors
            v1 = point1 - point2
            v2 = point3 - point2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return angle
            
        except Exception:
            return 90.0  # Default angle
    
    def _generate_pose_recommendations(self, body_alignment: float, arm_extension: float,
                                     posture_stability: float, technique_consistency: float,
                                     safety_assessment: float) -> List[str]:
        """Generate pose improvement recommendations."""
        recommendations = []
        
        if body_alignment < 0.7:
            recommendations.append("Improve body alignment - position shoulders directly over hands")
        
        if arm_extension < 0.8:
            recommendations.append("Keep arms straight and locked - avoid bending elbows")
        
        if posture_stability < 0.7:
            recommendations.append("Maintain stable posture - keep hips and shoulders aligned")
        
        if technique_consistency < 0.6:
            recommendations.append("Focus on consistent technique - maintain steady positioning")
        
        if safety_assessment < 0.8:
            recommendations.append("Adjust positioning for safety - ensure proper head and arm placement")
        
        if not recommendations:
            recommendations.append("Excellent pose technique - maintain current positioning")
        
        return recommendations
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate pose analysis input."""
        if not isinstance(input_data, np.ndarray):
            return False
        
        # Check for video frame (H, W, 3) or pose landmarks (33, 3)
        if len(input_data.shape) == 3:
            # Video frame
            return input_data.shape[2] == 3
        elif len(input_data.shape) == 2:
            # Pose landmarks
            return input_data.shape[0] == 33 and input_data.shape[1] == 3
        
        return False
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get pose analysis performance summary."""
        if not self.analysis_history:
            return {"message": "No analysis history available"}
        
        scores = [entry['overall_score'] for entry in self.analysis_history]
        processing_times = [entry['processing_time'] for entry in self.analysis_history]
        
        return {
            "total_analyses": len(self.analysis_history),
            "average_pose_score": np.mean(scores),
            "best_pose_score": np.max(scores),
            "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "average_processing_time_ms": np.mean(processing_times) * 1000,
            "model_version": self.model_version
        }
