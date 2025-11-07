"""
Medical Quality Scoring Model for SMART-TRAIN platform.

This module provides comprehensive quality scoring for medical procedures
with multi-dimensional assessment and performance tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityDimension:
    """Individual quality dimension assessment."""
    name: str
    score: float
    weight: float
    threshold_met: bool
    improvement_needed: bool
    feedback: str


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""
    overall_score: float
    quality_level: QualityLevel
    dimensions: List[QualityDimension]
    weighted_score: float
    compliance_score: float
    safety_score: float
    improvement_priority: List[str]
    strengths: List[str]
    recommendations: List[str]
    confidence: float


class MedicalQualityScoringModel(BaseModel):
    """
    Medical quality scoring model for comprehensive assessment.
    
    This model provides multi-dimensional quality assessment
    for medical procedures with weighted scoring and prioritized feedback.
    """
    
    def __init__(self, model_version: str = "2.0.0"):
        super().__init__("MedicalQualityScoring", model_version)
        
        self.audit_manager = AuditTrailManager()
        
        # Quality dimensions and weights for CPR
        self.cpr_dimensions = {
            'compression_depth': {
                'weight': 0.25,
                'threshold': 0.8,
                'critical_threshold': 0.5,
                'description': 'Compression depth adequacy'
            },
            'compression_rate': {
                'weight': 0.20,
                'threshold': 0.8,
                'critical_threshold': 0.5,
                'description': 'Compression rate consistency'
            },
            'hand_position': {
                'weight': 0.20,
                'threshold': 0.8,
                'critical_threshold': 0.6,
                'description': 'Hand placement accuracy'
            },
            'release_completeness': {
                'weight': 0.15,
                'threshold': 0.9,
                'critical_threshold': 0.7,
                'description': 'Complete chest recoil'
            },
            'rhythm_consistency': {
                'weight': 0.10,
                'threshold': 0.7,
                'critical_threshold': 0.5,
                'description': 'Rhythm maintenance'
            },
            'body_alignment': {
                'weight': 0.10,
                'threshold': 0.8,
                'critical_threshold': 0.6,
                'description': 'Body positioning'
            }
        }
        
        # Quality level thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.FAIR: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.CRITICAL: 0.0
        }
        
        # Assessment history for trend analysis
        self.assessment_history = []
        
        logger.info("Medical Quality Scoring Model initialized", model_version=model_version)
    
    def load_model(self, model_path: Optional[Any] = None) -> None:
        """Load quality scoring model (rule-based system)."""
        self.is_loaded = True
        self.load_timestamp = time.time()
        
        logger.info("Medical Quality Scoring Model loaded")
        
        # Log model loading event
        self.audit_manager.log_event(
            event_type=AuditEventType.MODEL_OPERATION,
            description="Medical Quality Scoring Model loaded",
            severity=AuditSeverity.INFO
        )
    
    def predict(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """
        Perform comprehensive quality assessment.
        
        Args:
            input_data: Dictionary containing analysis results from other models
            
        Returns:
            ProcessingResult with quality assessment
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            start_time = time.time()
            
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data for quality scoring")
            
            # Perform quality assessment
            quality_assessment = self._assess_quality(input_data)
            
            # Update assessment history
            self.assessment_history.append({
                'timestamp': start_time,
                'overall_score': quality_assessment.overall_score,
                'quality_level': quality_assessment.quality_level.value,
                'compliance_score': quality_assessment.compliance_score
            })
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.inference_count += 1
            
            # Create result
            result = ProcessingResult(
                success=True,
                data={
                    'quality_assessment': quality_assessment.__dict__,
                    'processing_time_ms': processing_time * 1000,
                    'model_version': self.model_version
                },
                metadata={
                    'model_id': self.model_id,
                    'dimensions_assessed': len(quality_assessment.dimensions),
                    'processing_timestamp': start_time
                }
            )
            
            # Log assessment event
            self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                description=f"Quality assessment completed: {quality_assessment.quality_level.value}",
                severity=AuditSeverity.INFO,
                metadata={
                    'overall_score': quality_assessment.overall_score,
                    'quality_level': quality_assessment.quality_level.value,
                    'compliance_score': quality_assessment.compliance_score,
                    'processing_time_ms': processing_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            logger.error("Quality assessment failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"Quality assessment failed: {e}",
                data={}
            )
    
    def _assess_quality(self, input_data: Dict[str, Any]) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        # Extract metrics from input data
        cpr_metrics = input_data.get('cpr_metrics', {})
        pose_analysis = input_data.get('pose_analysis', {})
        
        # Assess individual dimensions
        dimensions = []
        
        # CPR-specific dimensions
        if cpr_metrics:
            dimensions.extend(self._assess_cpr_dimensions(cpr_metrics))
        
        # Pose-specific dimensions
        if pose_analysis:
            dimensions.extend(self._assess_pose_dimensions(pose_analysis))
        
        # Calculate overall scores
        overall_score = np.mean([dim.score for dim in dimensions]) if dimensions else 0.0
        weighted_score = self._calculate_weighted_score(dimensions)
        
        # Determine quality level
        quality_level = self._determine_quality_level(weighted_score)
        
        # Calculate compliance and safety scores
        compliance_score = self._calculate_compliance_score(dimensions)
        safety_score = self._calculate_safety_score(dimensions)
        
        # Generate improvement priorities
        improvement_priority = self._generate_improvement_priorities(dimensions)
        
        # Identify strengths
        strengths = self._identify_strengths(dimensions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimensions, quality_level)
        
        # Calculate confidence
        confidence = self._calculate_confidence(dimensions)
        
        return QualityAssessment(
            overall_score=overall_score,
            quality_level=quality_level,
            dimensions=dimensions,
            weighted_score=weighted_score,
            compliance_score=compliance_score,
            safety_score=safety_score,
            improvement_priority=improvement_priority,
            strengths=strengths,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _assess_cpr_dimensions(self, cpr_metrics: Dict[str, Any]) -> List[QualityDimension]:
        """Assess CPR-specific quality dimensions."""
        dimensions = []
        
        for dim_name, config in self.cpr_dimensions.items():
            if dim_name in cpr_metrics:
                score = cpr_metrics[dim_name]
                
                # Normalize score if needed
                if dim_name == 'compression_depth':
                    # Convert from mm to normalized score
                    depth_mm = score if isinstance(score, (int, float)) else score
                    score = self._normalize_compression_depth(depth_mm)
                elif dim_name == 'compression_rate':
                    # Convert from CPM to normalized score
                    rate_cpm = score if isinstance(score, (int, float)) else score
                    score = self._normalize_compression_rate(rate_cpm)
                
                threshold_met = score >= config['threshold']
                improvement_needed = score < config['threshold']
                
                # Generate dimension-specific feedback
                feedback = self._generate_dimension_feedback(dim_name, score, config)
                
                dimension = QualityDimension(
                    name=dim_name,
                    score=score,
                    weight=config['weight'],
                    threshold_met=threshold_met,
                    improvement_needed=improvement_needed,
                    feedback=feedback
                )
                
                dimensions.append(dimension)
        
        return dimensions
    
    def _assess_pose_dimensions(self, pose_analysis: Dict[str, Any]) -> List[QualityDimension]:
        """Assess pose-specific quality dimensions."""
        dimensions = []
        
        pose_metrics = {
            'body_alignment_score': {'weight': 0.3, 'threshold': 0.8},
            'arm_extension_score': {'weight': 0.25, 'threshold': 0.8},
            'posture_stability_score': {'weight': 0.25, 'threshold': 0.75},
            'technique_consistency_score': {'weight': 0.2, 'threshold': 0.7}
        }
        
        for metric_name, config in pose_metrics.items():
            if metric_name in pose_analysis:
                score = pose_analysis[metric_name]
                threshold_met = score >= config['threshold']
                improvement_needed = score < config['threshold']
                
                feedback = self._generate_pose_feedback(metric_name, score)
                
                dimension = QualityDimension(
                    name=metric_name,
                    score=score,
                    weight=config['weight'],
                    threshold_met=threshold_met,
                    improvement_needed=improvement_needed,
                    feedback=feedback
                )
                
                dimensions.append(dimension)
        
        return dimensions
    
    def _normalize_compression_depth(self, depth_mm: float) -> float:
        """Normalize compression depth to 0-1 score."""
        if 50 <= depth_mm <= 60:
            return 1.0
        elif depth_mm < 50:
            return max(0.0, depth_mm / 50)
        else:
            return max(0.0, 1.0 - (depth_mm - 60) / 20)
    
    def _normalize_compression_rate(self, rate_cpm: float) -> float:
        """Normalize compression rate to 0-1 score."""
        if 100 <= rate_cpm <= 120:
            return 1.0
        elif rate_cpm < 100:
            return max(0.0, rate_cpm / 100)
        else:
            return max(0.0, 1.0 - (rate_cpm - 120) / 40)
    
    def _calculate_weighted_score(self, dimensions: List[QualityDimension]) -> float:
        """Calculate weighted overall score."""
        if not dimensions:
            return 0.0
        
        total_weighted_score = sum(dim.score * dim.weight for dim in dimensions)
        total_weight = sum(dim.weight for dim in dimensions)
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return QualityLevel.CRITICAL
    
    def _calculate_compliance_score(self, dimensions: List[QualityDimension]) -> float:
        """Calculate medical compliance score."""
        if not dimensions:
            return 0.0
        
        # Compliance based on threshold achievement
        compliant_dimensions = [dim for dim in dimensions if dim.threshold_met]
        return len(compliant_dimensions) / len(dimensions)
    
    def _calculate_safety_score(self, dimensions: List[QualityDimension]) -> float:
        """Calculate safety score based on critical thresholds."""
        if not dimensions:
            return 1.0
        
        safety_violations = 0
        for dim in dimensions:
            dim_config = self.cpr_dimensions.get(dim.name, {})
            critical_threshold = dim_config.get('critical_threshold', 0.5)
            
            if dim.score < critical_threshold:
                safety_violations += 1
        
        return max(0.0, 1.0 - (safety_violations / len(dimensions)))
    
    def _generate_improvement_priorities(self, dimensions: List[QualityDimension]) -> List[str]:
        """Generate prioritized list of improvements needed."""
        # Sort dimensions by score (lowest first) and weight (highest first)
        improvement_dims = [dim for dim in dimensions if dim.improvement_needed]
        improvement_dims.sort(key=lambda x: (x.score, -x.weight))
        
        priorities = []
        for dim in improvement_dims[:3]:  # Top 3 priorities
            priorities.append(f"{dim.name.replace('_', ' ').title()}: {dim.feedback}")
        
        return priorities
    
    def _identify_strengths(self, dimensions: List[QualityDimension]) -> List[str]:
        """Identify areas of strength."""
        strong_dimensions = [dim for dim in dimensions if dim.score >= 0.9]
        strong_dimensions.sort(key=lambda x: x.score, reverse=True)
        
        strengths = []
        for dim in strong_dimensions[:3]:  # Top 3 strengths
            strengths.append(f"Excellent {dim.name.replace('_', ' ')}")
        
        return strengths
    
    def _generate_recommendations(self, dimensions: List[QualityDimension], 
                                quality_level: QualityLevel) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if quality_level == QualityLevel.CRITICAL:
            recommendations.append("CRITICAL: Immediate technique correction required")
            recommendations.append("Consider additional training before continuing")
        elif quality_level == QualityLevel.POOR:
            recommendations.append("Focus on fundamental technique improvement")
            recommendations.append("Practice basic skills before advancing")
        elif quality_level == QualityLevel.FAIR:
            recommendations.append("Good foundation - focus on consistency")
            recommendations.append("Address specific technique gaps")
        elif quality_level == QualityLevel.GOOD:
            recommendations.append("Strong performance - fine-tune technique")
            recommendations.append("Focus on maintaining consistency")
        else:  # EXCELLENT
            recommendations.append("Outstanding performance - maintain excellence")
            recommendations.append("Consider advanced training scenarios")
        
        # Add specific dimension recommendations
        for dim in dimensions:
            if dim.improvement_needed and dim.score < 0.6:
                recommendations.append(f"Priority: {dim.feedback}")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _calculate_confidence(self, dimensions: List[QualityDimension]) -> float:
        """Calculate confidence in the assessment."""
        if not dimensions:
            return 0.0
        
        # Confidence based on score consistency and data availability
        scores = [dim.score for dim in dimensions]
        score_std = np.std(scores)
        
        # Higher confidence with more consistent scores and more dimensions
        consistency_factor = max(0.0, 1.0 - score_std)
        data_factor = min(1.0, len(dimensions) / 6)  # Normalize to expected 6 dimensions
        
        return (consistency_factor + data_factor) / 2
    
    def _generate_dimension_feedback(self, dim_name: str, score: float, 
                                   config: Dict[str, Any]) -> str:
        """Generate feedback for specific dimension."""
        feedback_templates = {
            'compression_depth': {
                'excellent': "Perfect compression depth",
                'good': "Good compression depth",
                'fair': "Compression depth needs improvement",
                'poor': "Insufficient compression depth"
            },
            'compression_rate': {
                'excellent': "Optimal compression rate",
                'good': "Good compression rate",
                'fair': "Adjust compression rate",
                'poor': "Compression rate too slow/fast"
            },
            'hand_position': {
                'excellent': "Perfect hand placement",
                'good': "Good hand position",
                'fair': "Adjust hand position",
                'poor': "Incorrect hand placement"
            },
            'release_completeness': {
                'excellent': "Complete chest recoil",
                'good': "Good chest recoil",
                'fair': "Improve chest recoil",
                'poor': "Incomplete chest recoil"
            },
            'rhythm_consistency': {
                'excellent': "Consistent rhythm",
                'good': "Good rhythm",
                'fair': "Improve rhythm consistency",
                'poor': "Inconsistent rhythm"
            },
            'body_alignment': {
                'excellent': "Perfect body alignment",
                'good': "Good body position",
                'fair': "Adjust body alignment",
                'poor': "Poor body positioning"
            }
        }
        
        templates = feedback_templates.get(dim_name, {
            'excellent': "Excellent performance",
            'good': "Good performance",
            'fair': "Needs improvement",
            'poor': "Requires attention"
        })
        
        if score >= 0.9:
            return templates['excellent']
        elif score >= 0.8:
            return templates['good']
        elif score >= 0.6:
            return templates['fair']
        else:
            return templates['poor']
    
    def _generate_pose_feedback(self, metric_name: str, score: float) -> str:
        """Generate feedback for pose metrics."""
        metric_feedback = {
            'body_alignment_score': "body alignment",
            'arm_extension_score': "arm extension",
            'posture_stability_score': "posture stability",
            'technique_consistency_score': "technique consistency"
        }
        
        metric_desc = metric_feedback.get(metric_name, metric_name)
        
        if score >= 0.9:
            return f"Excellent {metric_desc}"
        elif score >= 0.8:
            return f"Good {metric_desc}"
        elif score >= 0.6:
            return f"Improve {metric_desc}"
        else:
            return f"Focus on {metric_desc}"
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate quality scoring input."""
        if not isinstance(input_data, dict):
            return False
        
        # Must have at least one analysis result
        required_keys = ['cpr_metrics', 'pose_analysis']
        if not any(key in input_data for key in required_keys):
            return False
        
        return True
    
    def get_assessment_trends(self) -> Dict[str, Any]:
        """Get assessment trends and statistics."""
        if not self.assessment_history:
            return {"message": "No assessment history available"}
        
        scores = [entry['overall_score'] for entry in self.assessment_history]
        compliance_scores = [entry['compliance_score'] for entry in self.assessment_history]
        
        return {
            "total_assessments": len(self.assessment_history),
            "average_score": np.mean(scores),
            "best_score": np.max(scores),
            "score_improvement": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            "average_compliance": np.mean(compliance_scores),
            "current_trend": "improving" if len(scores) > 2 and scores[-1] > scores[-3] else "stable",
            "model_version": self.model_version
        }
