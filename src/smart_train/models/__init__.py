"""
AI Models module for SMART-TRAIN platform.

This module contains advanced AI models for medical training analysis:
- CPR Quality Assessment Model
- Action Recognition Model  
- Real-time Feedback Generation
- Medical Procedure Classification
- Performance Scoring Models
"""

from .cpr_quality_model import CPRQualityAssessmentModel
from .action_recognition import MedicalActionRecognitionModel
from .realtime_feedback import RealTimeFeedbackModel
from .pose_analysis import MedicalPoseAnalysisModel
from .quality_scoring import MedicalQualityScoringModel

__all__ = [
    "CPRQualityAssessmentModel",
    "MedicalActionRecognitionModel", 
    "RealTimeFeedbackModel",
    "MedicalPoseAnalysisModel",
    "MedicalQualityScoringModel"
]
