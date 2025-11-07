"""
Real-time Feedback Model for SMART-TRAIN platform.

This module provides real-time feedback generation for medical training
based on AI analysis results and medical guidelines.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger=get_logger(__name__)


class FeedbackPriority(Enum):
    """Priority levels for feedback messages."""
    CRITICAL="critical"      # Immediate safety concerns
    HIGH="high"             # Major technique issues
    MEDIUM="medium"         # Moderate improvements needed
    LOW="low"               # Minor optimizations
    POSITIVE="positive"     # Encouragement and confirmation


class FeedbackType(Enum):
    """Types of feedback messages."""
    TECHNIQUE="technique"           # Technical skill feedback
    SAFETY="safety"                # Safety-related feedback
    COMPLIANCE="compliance"        # AHA guideline compliance
    PERFORMANCE="performance"      # Overall performance metrics
    ENCOURAGEMENT="encouragement"  # Positive reinforcement


@dataclass
class FeedbackMessage:
    """Individual feedback message structure."""
    message: str
    priority: FeedbackPriority
    feedback_type: FeedbackType
    timestamp: float
    confidence: float  # 0-1 confidence in the feedback
    action_required: bool
    medical_context: Optional[str] = None
    improvement_suggestion: Optional[str] = None


@dataclass
class RealTimeFeedbackResult:
    """Real-time feedback analysis result."""
    primary_feedback: FeedbackMessage
    secondary_feedback: List[FeedbackMessage]
    overall_performance_score: float
    trend_analysis: Dict[str, Any]
    next_focus_area: str
    session_progress: Dict[str, Any]


class FeedbackEngine:
    """
    Core feedback generation engine with medical expertise.
    """

    def __init__(self):
        self.medical_guidelines=self._load_medical_guidelines()
        self.feedback_templates=self._load_feedback_templates()
        self.performance_history=deque(maxlen=100)  # Last 100 assessments

    def _load_medical_guidelines(self) -> Dict[str, Any]:
        """Load medical guidelines and thresholds."""
        return {
            'cpr': {
                'compression_depth': {'min': 50, 'max': 60, 'optimal': 55},
                'compression_rate': {'min': 100, 'max': 120, 'optimal': 110},
                'hand_position_tolerance': 0.8,
                'release_completeness_min': 0.9,
                'rhythm_consistency_min': 0.7,
                'interruption_max_seconds': 10
            },
            'airway_management': {
                'head_tilt_angle': {'min': 15, 'max': 30},
                'chin_lift_distance': {'min': 2, 'max': 4},
                'seal_effectiveness_min': 0.85
            },
            'general': {
                'confidence_threshold': 0.7,
                'improvement_threshold': 0.1,
                'mastery_threshold': 0.9
            }
        }

    def _load_feedback_templates(self) -> Dict[str, Dict[str, str]]:
        """Load feedback message templates."""
        return {
            'cpr_compression_depth': {
                'too_shallow': "Compress deeper - aim for 2-2.4 inches (5-6 cm)",
                'too_deep': "Reduce compression depth to avoid injury",
                'optimal': "Perfect compression depth - maintain this technique",
                'improvement': "Compression depth improving - keep focusing on consistency"
            },
            'cpr_compression_rate': {
                'too_slow': "Increase compression rate - aim for 100-120 per minute",
                'too_fast': "Slow down compressions - quality over speed",
                'optimal': "Excellent compression rate - maintain this rhythm",
                'inconsistent': "Focus on maintaining steady rhythm"
            },
            'cpr_hand_position': {
                'incorrect': "Adjust hand position - center on lower half of breastbone",
                'optimal': "Perfect hand placement - excellent technique",
                'slight_adjustment': "Minor hand position adjustment needed"
            },
            'cpr_release': {
                'incomplete': "Allow complete chest recoil between compressions",
                'optimal': "Excellent chest recoil - maintaining proper technique",
                'improving': "Chest recoil improving - continue this focus"
            },
            'general_encouragement': {
                'good_progress': "Great progress! Your technique is improving",
                'consistent_performance': "Consistent performance - well done",
                'mastery_achieved': "Excellent mastery of this skill",
                'keep_practicing': "Keep practicing - you're on the right track"
            },
            'safety_critical': {
                'stop_immediately': "STOP - Safety concern detected",
                'technique_dangerous': "Adjust technique immediately to prevent injury",
                'seek_instructor': "Please consult with instructor"
            }
        }

    def generate_feedback(self, analysis_results: Dict[str, Any]) -> List[FeedbackMessage]:
        """
        Generate contextual feedback based on analysis results.

        Args:
            analysis_results: AI model analysis results

        Returns:
            List of prioritized feedback messages
        """
        feedback_messages=[]
        current_time=time.time()

        # Extract key metrics
        if 'cpr_metrics' in analysis_results:
            cpr_feedback=self._generate_cpr_feedback(
                analysis_results['cpr_metrics'], current_time
            )
            feedback_messages.extend(cpr_feedback)

        if 'pose_analysis' in analysis_results:
            pose_feedback=self._generate_pose_feedback(
                analysis_results['pose_analysis'], current_time
            )
            feedback_messages.extend(pose_feedback)

        # Add performance trend feedback
        trend_feedback=self._generate_trend_feedback(current_time)
        if trend_feedback:
            feedback_messages.extend(trend_feedback)

        # Sort by priority and confidence
        feedback_messages.sort(
            key=lambda x: (x.priority.value, -x.confidence),
            reverse=True
        )

        return feedback_messages

    def _generate_cpr_feedback(self, cpr_metrics: Dict[str, Any], timestamp: float) -> List[FeedbackMessage]:
        """Generate CPR-specific feedback."""
        feedback=[]
        guidelines=self.medical_guidelines['cpr']

        # Compression depth feedback
        depth=cpr_metrics.get('compression_depth', 0)
        if depth < guidelines['compression_depth']['min']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_compression_depth']['too_shallow'],
                priority=FeedbackPriority.HIGH,
                feedback_type=FeedbackType.TECHNIQUE,
                timestamp=timestamp,
                confidence=0.9,
                action_required=True,
                medical_context="AHA Guidelines: 2-2.4 inches compression depth",
                improvement_suggestion="Focus on pushing harder with locked arms"
            ))
        elif depth > guidelines['compression_depth']['max']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_compression_depth']['too_deep'],
                priority=FeedbackPriority.MEDIUM,
                feedback_type=FeedbackType.SAFETY,
                timestamp=timestamp,
                confidence=0.85,
                action_required=True,
                medical_context="Risk of rib fracture with excessive depth"
            ))
        else:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_compression_depth']['optimal'],
                priority=FeedbackPriority.POSITIVE,
                feedback_type=FeedbackType.ENCOURAGEMENT,
                timestamp=timestamp,
                confidence=0.95,
                action_required=False
            ))

        # Compression rate feedback
        rate=cpr_metrics.get('compression_rate', 0)
        if rate < guidelines['compression_rate']['min']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_compression_rate']['too_slow'],
                priority=FeedbackPriority.HIGH,
                feedback_type=FeedbackType.TECHNIQUE,
                timestamp=timestamp,
                confidence=0.9,
                action_required=True,
                improvement_suggestion="Count aloud: 1-and-2-and-3..."
            ))
        elif rate > guidelines['compression_rate']['max']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_compression_rate']['too_fast'],
                priority=FeedbackPriority.MEDIUM,
                feedback_type=FeedbackType.TECHNIQUE,
                timestamp=timestamp,
                confidence=0.85,
                action_required=True,
                improvement_suggestion="Focus on complete compressions rather than speed"
            ))

        # Hand position feedback
        hand_position=cpr_metrics.get('hand_position_score', 0)
        if hand_position < guidelines['hand_position_tolerance']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_hand_position']['incorrect'],
                priority=FeedbackPriority.HIGH,
                feedback_type=FeedbackType.TECHNIQUE,
                timestamp=timestamp,
                confidence=0.8,
                action_required=True,
                medical_context="Proper hand position ensures effective compressions"
            ))

        # Release completeness feedback
        release=cpr_metrics.get('release_completeness', 0)
        if release < guidelines['release_completeness_min']:
            feedback.append(FeedbackMessage(
                message=self.feedback_templates['cpr_release']['incomplete'],
                priority=FeedbackPriority.MEDIUM,
                feedback_type=FeedbackType.TECHNIQUE,
                timestamp=timestamp,
                confidence=0.85,
                action_required=True,
                improvement_suggestion="Lift hands slightly between compressions"
            ))

        return feedback

    def _generate_pose_feedback(self, pose_analysis: Dict[str, Any], timestamp: float) -> List[FeedbackMessage]:
        """Generate pose-specific feedback."""
        feedback=[]

        # Body alignment feedback
        if 'body_alignment_score' in pose_analysis:
            alignment=pose_analysis['body_alignment_score']
            if alignment < 0.7:
                feedback.append(FeedbackMessage(
                    message="Improve body alignment - keep shoulders over hands",
                    priority=FeedbackPriority.MEDIUM,
                    feedback_type=FeedbackType.TECHNIQUE,
                    timestamp=timestamp,
                    confidence=0.8,
                    action_required=True,
                    improvement_suggestion="Position yourself directly over the patient"
                ))

        # Arm position feedback
        if 'arm_extension_score' in pose_analysis:
            arm_extension=pose_analysis['arm_extension_score']
            if arm_extension < 0.8:
                feedback.append(FeedbackMessage(
                    message="Keep arms straight and locked during compressions",
                    priority=FeedbackPriority.MEDIUM,
                    feedback_type=FeedbackType.TECHNIQUE,
                    timestamp=timestamp,
                    confidence=0.85,
                    action_required=True,
                    improvement_suggestion="Use your body weight, not arm muscles"
                ))

        return feedback

    def _generate_trend_feedback(self, timestamp: float) -> List[FeedbackMessage]:
        """Generate feedback based on performance trends."""
        if len(self.performance_history) < 5:
            return []

        feedback=[]
        recent_scores=[entry['overall_score'] for entry in list(self.performance_history)[-5:]]

        # Check for improvement trend
        if len(recent_scores) >= 3:
            trend=np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

            if trend > 0.02:  # Improving
                feedback.append(FeedbackMessage(
                    message="Great improvement! Your technique is getting better",
                    priority=FeedbackPriority.POSITIVE,
                    feedback_type=FeedbackType.ENCOURAGEMENT,
                    timestamp=timestamp,
                    confidence=0.9,
                    action_required=False
                ))
            elif trend < -0.02:  # Declining
                feedback.append(FeedbackMessage(
                    message="Focus on maintaining technique - take a brief rest if needed",
                    priority=FeedbackPriority.MEDIUM,
                    feedback_type=FeedbackType.PERFORMANCE,
                    timestamp=timestamp,
                    confidence=0.8,
                    action_required=False,
                    improvement_suggestion="Fatigue may be affecting performance"
                ))

        return feedback


class RealTimeFeedbackModel(BaseModel):
    """
    Real-time feedback model for medical training analysis.

    This model provides immediate, contextual feedback based on
    AI analysis results and medical guidelines.
    """

    def __init__(self, model_version: str="2.0.0"):
        super().__init__("RealTimeFeedback", model_version)

        self.feedback_engine=FeedbackEngine()
        self.audit_manager=AuditTrailManager()
        self.session_data={
            'start_time': time.time(),
            'total_feedback_given': 0,
            'critical_issues_identified': 0,
            'positive_reinforcements': 0,
            'performance_trend': []
        }

        logger.info("Real-time Feedback Model initialized", model_version=model_version)

    def load_model(self, model_path: Optional[Any] = None) -> None:
        """
        Load feedback model (no actual model file needed).
        """
        self.is_loaded=True
        self.load_timestamp=time.time()

        logger.info("Real-time Feedback Model loaded")

        # Log model loading event
        self.audit_manager.log_event(
            event_type=AuditEventType.MODEL_OPERATION,
            description="Real-time Feedback Model loaded",
            severity=AuditSeverity.INFO
        )

    def predict(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """
        Generate real-time feedback based on analysis results.

        Args:
            input_data: Dictionary containing analysis results from other models

        Returns:
            ProcessingResult with feedback recommendations
        """
        if not self.is_loaded:
            self.load_model()

        try:
            start_time=time.time()

            # Generate feedback messages
            feedback_messages=self.feedback_engine.generate_feedback(input_data)

            # Update performance history
            if 'overall_score' in input_data:
                self.feedback_engine.performance_history.append({
                    'timestamp': start_time,
                    'overall_score': input_data['overall_score']
                })

            # Analyze trends
            trend_analysis=self._analyze_performance_trends()

            # Determine primary and secondary feedback
            primary_feedback=feedback_messages[0] if feedback_messages else None
            secondary_feedback=feedback_messages[1:5] if len(feedback_messages) > 1 else []

            # Calculate overall performance score
            overall_score=input_data.get('overall_score', 0.0)

            # Determine next focus area
            next_focus=self._determine_next_focus_area(input_data, feedback_messages)

            # Update session statistics
            self._update_session_stats(feedback_messages)

            # Create feedback result
            feedback_result=RealTimeFeedbackResult(
                primary_feedback=primary_feedback,
                secondary_feedback=secondary_feedback,
                overall_performance_score=overall_score,
                trend_analysis=trend_analysis,
                next_focus_area=next_focus,
                session_progress=self._get_session_progress()
            )

            # Calculate processing time
            processing_time=time.time() - start_time
            self.inference_count += 1

            # Create result
            result=ProcessingResult(
                success=True,
                message="Real-time feedback generated successfully",
                data={
                    'feedback_result': feedback_result.__dict__,
                    'feedback_messages': [msg.__dict__ for msg in feedback_messages],
                    'model_version': self.model_version,
                    'model_id': self.model_id,
                    'feedback_count': len(feedback_messages),
                    'session_duration': time.time() - self.session_data['start_time'],
                    'processing_timestamp': start_time
                }
            )

            # Log feedback event
            self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                description=f"Real-time feedback generated: {len(feedback_messages)} messages",
                severity=AuditSeverity.INFO,
                details={
                    'feedback_count': len(feedback_messages),
                    'primary_priority': primary_feedback.priority.value if primary_feedback else None,
                    'processing_time_ms': processing_time * 1000
                }
            )

            return result

        except Exception as e:
            logger.error("Real-time feedback generation failed", error=str(e))
            return ProcessingResult(
                success=False,
                message=f"Feedback generation failed: {e}",
                data={}
            )

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        history=list(self.feedback_engine.performance_history)

        if len(history) < 3:
            return {"status": "insufficient_data", "message": "Need more data for trend analysis"}

        scores=[entry['overall_score'] for entry in history]
        _timestamps=[entry['timestamp'] for entry in history]

        # Calculate trend
        trend_slope=np.polyfit(range(len(scores)), scores, 1)[0]

        # Calculate performance statistics
        current_avg=np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores)
        overall_avg=np.mean(scores)
        improvement=current_avg - overall_avg

        return {
            "trend_direction": "improving" if trend_slope > 0.01 else "declining" if trend_slope < -0.01 else "stable",
            "trend_strength": abs(trend_slope),
            "current_average": current_avg,
            "overall_average": overall_avg,
            "improvement_from_start": improvement,
            "consistency_score": 1.0 - np.std(scores) if len(scores) > 1 else 1.0,
            "data_points": len(history)
        }

    def _determine_next_focus_area(self, input_data: Dict[str, Any], feedback_messages: List[FeedbackMessage]) -> str:
        """Determine the next area for the user to focus on."""
        # Priority-based focus determination
        critical_messages=[msg for msg in feedback_messages if msg.priority== FeedbackPriority.CRITICAL]
        if critical_messages:
            return "Safety - Address critical issues immediately"

        high_priority=[msg for msg in feedback_messages if msg.priority== FeedbackPriority.HIGH]
        if high_priority:
            technique_issues=[msg for msg in high_priority if msg.feedback_type== FeedbackType.TECHNIQUE]
            if technique_issues:
                return f"Technique - {technique_issues[0].message.split(' - ')[0]}"

        # If no high priority issues, focus on consistency
        if 'cpr_metrics' in input_data:
            cpr=input_data['cpr_metrics']
            lowest_score_metric=min(
                [
                    ('compression_depth', cpr.get('compression_depth', 1.0)),
                    ('compression_rate', cpr.get('compression_rate', 1.0)),
                    ('hand_position', cpr.get('hand_position_score', 1.0)),
                    ('release_completeness', cpr.get('release_completeness', 1.0)),
                    ('rhythm_consistency', cpr.get('rhythm_consistency', 1.0))
                ],
                key=lambda x: x[1]
            )
            return f"Consistency - Focus on {lowest_score_metric[0].replace('_', ' ')}"

        return "Overall Performance - Maintain current technique"

    def _update_session_stats(self, feedback_messages: List[FeedbackMessage]) -> None:
        """Update session statistics."""
        self.session_data['total_feedback_given'] += len(feedback_messages)

        for msg in feedback_messages:
            if msg.priority== FeedbackPriority.CRITICAL:
                self.session_data['critical_issues_identified'] += 1
            elif msg.priority== FeedbackPriority.POSITIVE:
                self.session_data['positive_reinforcements'] += 1

    def _get_session_progress(self) -> Dict[str, Any]:
        """Get current session progress."""
        session_duration=time.time() - self.session_data['start_time']

        return {
            'session_duration_minutes': session_duration / 60,
            'total_feedback_given': self.session_data['total_feedback_given'],
            'critical_issues_identified': self.session_data['critical_issues_identified'],
            'positive_reinforcements': self.session_data['positive_reinforcements'],
            'feedback_rate_per_minute': self.session_data['total_feedback_given'] / (session_duration / 60) if session_duration > 0 else 0,
            'safety_score': max(0, 1.0 - (self.session_data['critical_issues_identified'] * 0.2))
        }

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate feedback input data.

        Args:
            input_data: Input analysis results

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            return False

        # Must have at least one analysis result
        required_keys=['cpr_metrics', 'pose_analysis', 'overall_score']
        if not any(key in input_data for key in required_keys):
            return False

        return True

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return {
            'session_stats': self.session_data,
            'performance_trend': self._analyze_performance_trends(),
            'model_info': self.get_model_info(),
            'total_inferences': self.inference_count
        }
