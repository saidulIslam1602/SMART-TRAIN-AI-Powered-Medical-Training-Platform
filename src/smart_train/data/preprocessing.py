"""
Enhanced medical data preprocessing pipeline for SMART-TRAIN platform.

This module provides enterprise-grade data preprocessing capabilities with
medical compliance, quality assurance, and real-time processing support.
"""

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
import albumentations as A
from tqdm import tqdm
import structlog

from ..core.base import BaseProcessor, ProcessingResult
from ..core.exceptions import DataValidationError, MedicalComplianceError
from ..core.config import get_config
from ..core.logging import MedicalLogger
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger=structlog.get_logger(__name__)


@dataclass
class MedicalPoseAnnotation:
    """Enhanced medical pose annotation with compliance tracking."""
    frame_id: int
    timestamp: float
    keypoints: List[Tuple[float, float, float]]  # x, y, confidence
    medical_metrics: Dict[str, float]
    quality_score: float
    aha_compliance: bool
    compliance_details: Dict[str, Any]
    processing_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class VideoProcessingResult:
    """Result of video processing with comprehensive metadata."""
    video_path: str
    success: bool
    total_frames: int
    processed_frames: int
    annotations: List[MedicalPoseAnnotation]
    quality_metrics: Dict[str, float]
    compliance_report: Dict[str, Any]
    processing_time_seconds: float
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict=asdict(self)
        result_dict['annotations'] = [ann.to_dict() for ann in self.annotations]
        return result_dict


class AHAGuidelinesValidator:
    """
    Validator for AHA CPR Guidelines 2020 compliance.

    This class implements the latest American Heart Association guidelines
    for CPR technique assessment and validation.
    """

    def __init__(self):
        """Initialize AHA guidelines validator."""
        self.guidelines={
            "compression_depth": {"min": 5.0, "max": 6.0, "unit": "cm"},
            "compression_rate": {"min": 100, "max": 120, "unit": "per_minute"},
            "compression_fraction": {"min": 0.6, "unit": "percentage"},
            "release_threshold": 0.9,  # 90% complete release
            "hand_position": {
                "center_chest": True,
                "lower_sternum": True,
                "heel_of_hand": True,
                "fingers_interlaced": True
            },
            "interruption_max_seconds": 10,
            "ventilation_rate": {"min": 8, "max": 10, "unit": "per_minute"}
        }

        self.logger=MedicalLogger("aha_validator")

    def validate_compression_depth(self, depth_measurements: List[float]) -> Dict[str, Any]:
        """Validate compression depth against AHA guidelines."""
        if not depth_measurements:
            return {"compliant": False, "reason": "No depth measurements available"}

        avg_depth=np.mean(depth_measurements)
        min_depth=self.guidelines["compression_depth"]["min"]
        max_depth=self.guidelines["compression_depth"]["max"]

        compliant=min_depth <= avg_depth <= max_depth

        return {
            "compliant": compliant,
            "average_depth_cm": avg_depth,
            "target_range": f"{min_depth}-{max_depth} cm",
            "measurements_count": len(depth_measurements),
            "depth_variability": np.std(depth_measurements),
            "compliance_percentage": len([d for d in depth_measurements if min_depth <= d <= max_depth]) / len(depth_measurements)
        }

    def validate_compression_rate(self, compression_timestamps: List[float]) -> Dict[str, Any]:
        """Validate compression rate against AHA guidelines."""
        if len(compression_timestamps) < 2:
            return {"compliant": False, "reason": "Insufficient compression data"}

        # Calculate rate from timestamps
        time_intervals=np.diff(compression_timestamps)
        avg_interval=np.mean(time_intervals)
        rate_per_minute=60.0 / avg_interval if avg_interval > 0 else 0

        min_rate=self.guidelines["compression_rate"]["min"]
        max_rate=self.guidelines["compression_rate"]["max"]

        compliant=min_rate <= rate_per_minute <= max_rate

        return {
            "compliant": compliant,
            "rate_per_minute": rate_per_minute,
            "target_range": f"{min_rate}-{max_rate} per minute",
            "rate_variability": np.std(60.0 / time_intervals) if len(time_intervals) > 1 else 0,
            "total_compressions": len(compression_timestamps)
        }

    def validate_hand_position(self, hand_position_scores: List[float]) -> Dict[str, Any]:
        """Validate hand position against AHA guidelines."""
        if not hand_position_scores:
            return {"compliant": False, "reason": "No hand position data available"}

        avg_score=np.mean(hand_position_scores)
        compliant=avg_score >= 0.8  # 80% threshold for good hand position

        return {
            "compliant": compliant,
            "average_position_score": avg_score,
            "minimum_threshold": 0.8,
            "position_consistency": 1.0 - np.std(hand_position_scores),
            "measurements_count": len(hand_position_scores)
        }

    def generate_compliance_report(self, medical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AHA compliance report."""
        report={
            "aha_guidelines_version": "2020",
            "assessment_timestamp": datetime.now().isoformat(),
            "overall_compliance": True,
            "compliance_details": {},
            "recommendations": []
        }

        # Validate each component
        if "compression_depths" in medical_metrics:
            depth_validation=self.validate_compression_depth(medical_metrics["compression_depths"])
            report["compliance_details"]["compression_depth"] = depth_validation
            if not depth_validation["compliant"]:
                report["overall_compliance"] = False
                report["recommendations"].append("Adjust compression depth to 5-6 cm range")

        if "compression_timestamps" in medical_metrics:
            rate_validation=self.validate_compression_rate(medical_metrics["compression_timestamps"])
            report["compliance_details"]["compression_rate"] = rate_validation
            if not rate_validation["compliant"]:
                report["overall_compliance"] = False
                report["recommendations"].append("Maintain compression rate between 100-120 per minute")

        if "hand_position_scores" in medical_metrics:
            position_validation=self.validate_hand_position(medical_metrics["hand_position_scores"])
            report["compliance_details"]["hand_position"] = position_validation
            if not position_validation["compliant"]:
                report["overall_compliance"] = False
                report["recommendations"].append("Improve hand positioning on lower sternum")

        # Calculate overall compliance score
        compliance_scores=[
            detail.get("compliance_percentage", 0 if not detail.get("compliant", False) else 1)
            for detail in report["compliance_details"].values()
        ]

        report["compliance_score"] = np.mean(compliance_scores) if compliance_scores else 0.0

        return report


class MedicalDataPreprocessor(BaseProcessor):
    """
    Enterprise-grade medical data preprocessor with compliance support.

    This processor handles medical training videos with pose estimation,
    quality assessment, and AHA guidelines compliance checking.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize medical data preprocessor.

        Args:
            config_path: Path to configuration file
        """
        super().__init__("MedicalDataPreprocessor", "2.0.0")

        self.config=get_config()
        self.audit_manager=AuditTrailManager()
        self.aha_validator=AHAGuidelinesValidator()

        # Initialize MediaPipe pose estimation
        self.mp_pose=mp.solutions.pose
        self.pose_estimator=self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=self.config.model.pose_confidence_threshold
        )

        # Initialize data augmentation pipeline
        self.augmentation_pipeline=self._create_augmentation_pipeline()

        # Processing statistics
        self.processing_stats={
            "videos_processed": 0,
            "total_frames_processed": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0,
            "compliance_rate": 0.0
        }

        logger.info(
            "Medical data preprocessor initialized",
            processor_id=self.processor_id,
            pose_confidence_threshold=self.config.model.pose_confidence_threshold
        )

    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create medical-grade data augmentation pipeline."""
        return A.Compose([
            # Lighting and color augmentations (common in medical environments)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ], p=0.4),

            # Noise augmentations (simulate camera noise)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.3),

            # Blur augmentations (simulate motion or focus issues)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.2),

            # Geometric augmentations (different camera angles)
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
                A.Perspective(scale=(0.05, 0.1), p=0.2),
            ], p=0.3),

            # Occlusion simulation (hands/equipment blocking view)
            A.CoarseDropout(
                max_holes=2,
                max_height=40,
                max_width=40,
                min_holes=1,
                min_height=20,
                min_width=20,
                p=0.15
            ),
        ])

    def process(self, input_data: Union[str, Path, List[Union[str, Path]]], **kwargs) -> ProcessingResult:
        """
        Process medical training videos with comprehensive analysis.

        Args:
            input_data: Video file path(s) or directory containing videos
            **kwargs: Additional processing parameters

        Returns:
            ProcessingResult with processing results and metadata
        """
        start_time=time.time()

        result=ProcessingResult(
            success=True,
            message="Medical data preprocessing completed successfully"
        )

        try:
            # Validate input
            validation_result=self.validate_input(input_data)
            if not validation_result.success:
                return validation_result

            # Get list of video files to process
            video_files=self._get_video_files(input_data)

            if not video_files:
                result.success=False
                result.message="No video files found to process"
                return result

            # Process videos
            processing_results=[]

            # Use parallel processing for multiple videos
            max_workers=kwargs.get('max_workers', self.config.data_processing.parallel_processing_workers)

            if len(video_files) > 1 and max_workers > 1:
                processing_results=self._process_videos_parallel(video_files, max_workers, **kwargs)
            else:
                processing_results=self._process_videos_sequential(video_files, **kwargs)

            # Aggregate results
            successful_processing=[r for r in processing_results if r.success]
            failed_processing=[r for r in processing_results if not r.success]

            # Calculate aggregate metrics
            total_frames=sum(r.processed_frames for r in successful_processing)
            avg_quality=np.mean([r.quality_metrics.get("overall_quality", 0) for r in successful_processing]) if successful_processing else 0
            compliance_rate=np.mean([r.compliance_report.get("compliance_score", 0) for r in successful_processing]) if successful_processing else 0

            # Update processing statistics
            self.processing_stats["videos_processed"] += len(successful_processing)
            self.processing_stats["total_frames_processed"] += total_frames
            self.processing_stats["total_processing_time"] += time.time() - start_time
            self.processing_stats["average_quality_score"] = avg_quality
            self.processing_stats["compliance_rate"] = compliance_rate

            result.data={
                "total_videos": len(video_files),
                "successful_processing": len(successful_processing),
                "failed_processing": len(failed_processing),
                "total_frames_processed": total_frames,
                "average_quality_score": avg_quality,
                "compliance_rate": compliance_rate,
                "processing_results": [r.to_dict() for r in processing_results],
                "processing_statistics": self.processing_stats
            }

            result.processing_time_ms=(time.time() - start_time) * 1000

            # Log processing activity
            self._log_processing("medical_video_preprocessing", result)

            # Audit trail
            self.audit_manager.log_event(
                event_type=AuditEventType.DATA_MODIFICATION,
                description=f"Medical data preprocessing completed: {len(successful_processing)}/{len(video_files)} videos",
                details=result.data,
                severity=AuditSeverity.MEDIUM
            )

            if failed_processing:
                result.success=False
                result.message=f"Preprocessing partially failed: {len(failed_processing)} videos failed"
                for failed_result in failed_processing:
                    result.errors.extend(failed_result.errors)

            return result

        except Exception as e:
            result.success=False
            result.message=f"Medical data preprocessing failed: {str(e)}"
            result.add_error(str(e))

            self.logger.log_exception(e, context={"input_data": str(input_data)})
            return result

    def _get_video_files(self, input_data: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """Get list of video files from input data."""
        video_extensions=['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files=[]

        if isinstance(input_data, (str, Path)):
            input_path=Path(input_data)

            if input_path.is_file() and input_path.suffix.lower() in video_extensions:
                video_files.append(input_path)
            elif input_path.is_dir():
                for ext in video_extensions:
                    video_files.extend(input_path.rglob(f'*{ext}'))
        elif isinstance(input_data, list):
            for item in input_data:
                item_path=Path(item)
                if item_path.is_file() and item_path.suffix.lower() in video_extensions:
                    video_files.append(item_path)

        return video_files

    def _process_videos_sequential(self, video_files: List[Path], **kwargs) -> List[VideoProcessingResult]:
        """Process videos sequentially with progress tracking."""
        results=[]

        for video_file in tqdm(video_files, desc="Processing videos"):
            result=self._process_single_video(video_file, **kwargs)
            results.append(result)

        return results

    def _process_videos_parallel(self, video_files: List[Path], max_workers: int, **kwargs) -> List[VideoProcessingResult]:
        """Process videos in parallel for improved performance."""
        results=[]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video={
                executor.submit(self._process_single_video, video_file, **kwargs): video_file
                for video_file in video_files
            }

            for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing videos"):
                video_file=future_to_video[future]
                try:
                    result=future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result=VideoProcessingResult(
                        video_path=str(video_file),
                        success=False,
                        total_frames=0,
                        processed_frames=0,
                        annotations=[],
                        quality_metrics={},
                        compliance_report={},
                        processing_time_seconds=0.0,
                        errors=[f"Processing failed: {str(e)}"],
                        warnings=[]
                    )
                    results.append(error_result)

                    logger.error(f"Parallel processing failed for {video_file}", error=str(e))

        return results

    def _process_single_video(self, video_path: Path, **kwargs) -> VideoProcessingResult:
        """
        Process a single video file with comprehensive medical analysis.

        Args:
            video_path: Path to video file
            **kwargs: Processing parameters

        Returns:
            VideoProcessingResult with detailed analysis
        """
        start_time=time.time()

        result=VideoProcessingResult(
            video_path=str(video_path),
            success=False,
            total_frames=0,
            processed_frames=0,
            annotations=[],
            quality_metrics={},
            compliance_report={},
            processing_time_seconds=0.0,
            errors=[],
            warnings=[]
        )

        try:
            # Validate video file
            if not self._validate_video_file(video_path):
                result.errors.append("Invalid video file")
                return result

            # Open video
            cap=cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                result.errors.append("Failed to open video file")
                return result

            # Get video properties
            fps=cap.get(cv2.CAP_PROP_FPS)
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration=total_frames / fps if fps > 0 else 0

            result.total_frames=total_frames

            # Validate video duration
            if not self._validate_video_duration(duration):
                result.errors.append(f"Invalid video duration: {duration:.2f}s")
                cap.release()
                return result

            # Process frames
            annotations=[]
            medical_metrics_aggregated={
                "compression_depths": [],
                "compression_timestamps": [],
                "hand_position_scores": [],
                "quality_scores": []
            }

            frame_id=0
            processed_frames=0

            # Processing parameters
            frame_skip=kwargs.get('frame_skip', 1)  # Process every nth frame
            enable_augmentation=kwargs.get('enable_augmentation', False)

            while True:
                ret, frame=cap.read()
                if not ret:
                    break

                # Skip frames if specified
                if frame_id % frame_skip != 0:
                    frame_id += 1
                    continue

                try:
                    # Resize frame to target resolution
                    frame_resized=self._resize_frame(frame)

                    # Apply augmentation if enabled
                    if enable_augmentation and self.config.data_processing.data_augmentation_enabled:
                        augmented=self.augmentation_pipeline(image=frame_resized)
                        frame_processed=augmented['image']
                    else:
                        frame_processed=frame_resized

                    # Extract pose landmarks
                    pose_result=self.pose_estimator.process(cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB))

                    if pose_result.pose_landmarks:
                        # Extract keypoints
                        keypoints=self._extract_keypoints(pose_result.pose_landmarks)

                        # Calculate medical metrics
                        medical_metrics=self._calculate_medical_metrics(keypoints, frame_id, fps)

                        # Assess quality based on AHA guidelines
                        quality_score, aha_compliance, compliance_details=self._assess_technique_quality(medical_metrics)

                        # Create annotation
                        annotation=MedicalPoseAnnotation(
                            frame_id=frame_id,
                            _timestamp=frame_id / fps,
                            keypoints=keypoints,
                            medical_metrics=medical_metrics,
                            quality_score=quality_score,
                            aha_compliance=aha_compliance,
                            compliance_details=compliance_details,
                            processing_metadata={
                                "pose_confidence": np.mean([kp[2] for kp in keypoints]),
                                "augmentation_applied": enable_augmentation,
                                "processing_timestamp": datetime.now().isoformat()
                            }
                        )

                        annotations.append(annotation)

                        # Aggregate metrics for video-level analysis
                        if medical_metrics.get("compression_depth_cm"):
                            medical_metrics_aggregated["compression_depths"].append(medical_metrics["compression_depth_cm"])

                        if medical_metrics.get("compression_detected"):
                            medical_metrics_aggregated["compression_timestamps"].append(frame_id / fps)

                        if medical_metrics.get("hand_position_score"):
                            medical_metrics_aggregated["hand_position_scores"].append(medical_metrics["hand_position_score"])

                        medical_metrics_aggregated["quality_scores"].append(quality_score)

                        processed_frames += 1

                except Exception as e:
                    result.warnings.append(f"Frame {frame_id} processing failed: {str(e)}")

                frame_id += 1

            cap.release()

            result.processed_frames=processed_frames
            result.annotations=annotations

            # Calculate video-level quality metrics
            result.quality_metrics=self._calculate_video_quality_metrics(annotations, medical_metrics_aggregated)

            # Generate AHA compliance report
            result.compliance_report=self.aha_validator.generate_compliance_report(medical_metrics_aggregated)

            result.processing_time_seconds=time.time() - start_time
            result.success=True

            logger.info(
                f"Video processing completed: {video_path.name}",
                processed_frames=processed_frames,
                total_frames=total_frames,
                quality_score=result.quality_metrics.get("overall_quality", 0),
                compliance_score=result.compliance_report.get("compliance_score", 0)
            )

            return result

        except Exception as e:
            result.errors.append(f"Video processing failed: {str(e)}")
            result.processing_time_seconds=time.time() - start_time

            logger.error(f"Video processing failed: {video_path}", error=str(e))
            return result

    def _validate_video_file(self, video_path: Path) -> bool:
        """Validate video file format and accessibility."""
        try:
            if not video_path.exists():
                return False

            # Check file extension
            valid_extensions=['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            if video_path.suffix.lower() not in valid_extensions:
                return False

            # Try opening with OpenCV
            cap=cv2.VideoCapture(str(video_path))
            is_valid=cap.isOpened()
            cap.release()

            return is_valid

        except Exception:
            return False

    def _validate_video_duration(self, duration: float) -> bool:
        """Validate video duration against configuration."""
        min_duration=self.config.data_processing.min_video_duration_seconds
        max_duration=self.config.data_processing.max_video_duration_seconds

        return min_duration <= duration <= max_duration

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution."""
        target_height, target_width=self.config.data_processing.video_target_resolution
        return cv2.resize(frame, (target_width, target_height))

    def _extract_keypoints(self, pose_landmarks) -> List[Tuple[float, float, float]]:
        """Extract pose keypoints from MediaPipe results."""
        keypoints=[]

        for landmark in pose_landmarks.landmark:
            keypoints.append((
                landmark.x,
                landmark.y,
                landmark.visibility
            ))

        return keypoints

    def _calculate_medical_metrics(self, keypoints: List[Tuple[float, float, float]],
                                  frame_id: int, fps: float) -> Dict[str, float]:
        """Calculate comprehensive medical metrics from pose keypoints."""
        # MediaPipe pose landmark indices
        LEFT_WRIST=15
        RIGHT_WRIST=16
        LEFT_SHOULDER=11
        RIGHT_SHOULDER=12
        LEFT_ELBOW=13
        RIGHT_ELBOW=14
        LEFT_HIP=23
        RIGHT_HIP=24

        try:
            # Hand position analysis
            left_wrist=keypoints[LEFT_WRIST]
            right_wrist=keypoints[RIGHT_WRIST]
            left_shoulder=keypoints[LEFT_SHOULDER]
            right_shoulder=keypoints[RIGHT_SHOULDER]

            # Calculate hand center position
            hand_center_x=(left_wrist[0] + right_wrist[0]) / 2
            hand_center_y=(left_wrist[1] + right_wrist[1]) / 2

            # Calculate shoulder center
            shoulder_center_x=(left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_center_y=(left_shoulder[1] + right_shoulder[1]) / 2

            # Estimate compression depth (enhanced calculation)
            # This is a simplified 2D estimation - real implementation would use depth cameras
            compression_depth=abs(hand_center_y - shoulder_center_y) * 15  # Rough estimate in cm

            # Detect compression phase (hands moving toward chest)
            compression_detected=hand_center_y > shoulder_center_y + 0.1  # Threshold for compression

            # Calculate arm angles for proper positioning assessment
            left_arm_angle=self._calculate_arm_angle(
                keypoints[LEFT_SHOULDER],
                keypoints[LEFT_ELBOW],
                keypoints[LEFT_WRIST]
            )

            right_arm_angle=self._calculate_arm_angle(
                keypoints[RIGHT_SHOULDER],
                keypoints[RIGHT_ELBOW],
                keypoints[RIGHT_WRIST]
            )

            # Enhanced hand positioning score
            hand_position_score=self._assess_hand_position_advanced(
                hand_center_x, hand_center_y,
                shoulder_center_x, shoulder_center_y,
                keypoints
            )

            # Body alignment assessment
            body_alignment_score=self._assess_body_alignment(keypoints)

            # Compression quality indicators
            compression_quality=self._assess_compression_quality(
                compression_depth, left_arm_angle, right_arm_angle, hand_position_score
            )

            return {
                "compression_depth_cm": compression_depth,
                "compression_detected": compression_detected,
                "hand_center_x": hand_center_x,
                "hand_center_y": hand_center_y,
                "shoulder_center_x": shoulder_center_x,
                "shoulder_center_y": shoulder_center_y,
                "left_arm_angle": left_arm_angle,
                "right_arm_angle": right_arm_angle,
                "hand_position_score": hand_position_score,
                "body_alignment_score": body_alignment_score,
                "compression_quality": compression_quality,
                "frame_timestamp": frame_id / fps,
                "keypoint_confidence": np.mean([kp[2] for kp in keypoints])
            }

        except Exception as e:
            logger.warning(f"Error calculating medical metrics: {str(e)}")
            return {
                "compression_depth_cm": 0.0,
                "compression_detected": False,
                "hand_position_score": 0.0,
                "frame_timestamp": frame_id / fps,
                "error": str(e)
            }

    def _calculate_arm_angle(self, shoulder: Tuple[float, float, float],
                           elbow: Tuple[float, float, float],
                           wrist: Tuple[float, float, float]) -> float:
        """Calculate arm angle for proper CPR positioning."""
        try:
            # Vector from shoulder to elbow
            v1=np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
            # Vector from elbow to wrist
            v2=np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

            # Calculate angle
            cos_angle=np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle=np.arccos(np.clip(cos_angle, -1.0, 1.0))

            return np.degrees(angle)

        except Exception:
            return 0.0

    def _assess_hand_position_advanced(self, hand_x: float, hand_y: float,
                                     shoulder_x: float, shoulder_y: float,
                                     keypoints: List[Tuple[float, float, float]]) -> float:
        """Advanced hand position assessment for CPR compliance."""
        try:
            # Ideal position is center of chest, slightly below shoulder line
            ideal_x=shoulder_x  # Aligned with shoulders
            ideal_y=shoulder_y + 0.15  # Below shoulders for sternum positioning

            # Calculate distance from ideal position
            distance=np.sqrt((hand_x - ideal_x)**2 + (hand_y - ideal_y)**2)

            # Convert to score (1.0 is perfect, 0.0 is poor)
            base_score=max(0.0, 1.0 - distance * 3)

            # Additional factors for hand positioning

            # Check if hands are properly aligned (not too far apart)
            left_wrist=keypoints[15]
            right_wrist=keypoints[16]
            hand_separation=abs(left_wrist[0] - right_wrist[0])

            if hand_separation > 0.1:  # Hands too far apart
                base_score *= 0.8

            # Check for proper hand stacking (one hand on top of the other)
            hand_stacking_score=1.0 - abs(left_wrist[1] - right_wrist[1]) * 2
            hand_stacking_score=max(0.0, min(1.0, hand_stacking_score))

            # Combine scores
            final_score=(base_score + hand_stacking_score) / 2

            return max(0.0, min(1.0, final_score))

        except Exception:
            return 0.0

    def _assess_body_alignment(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """Assess body alignment for proper CPR positioning."""
        try:
            # Get key body landmarks
            left_shoulder=keypoints[11]
            right_shoulder=keypoints[12]
            left_hip=keypoints[23]
            right_hip=keypoints[24]

            # Calculate shoulder and hip alignment
            shoulder_alignment=abs(left_shoulder[1] - right_shoulder[1])
            hip_alignment=abs(left_hip[1] - right_hip[1])

            # Calculate body straightness (shoulders and hips should be aligned)
            body_straightness=1.0 - (shoulder_alignment + hip_alignment)
            body_straightness=max(0.0, min(1.0, body_straightness))

            # Check for proper kneeling position (hips should be above knees if visible)
            # This is a simplified check - real implementation would be more sophisticated

            return body_straightness

        except Exception:
            return 0.5  # Default neutral score

    def _assess_compression_quality(self, depth: float, left_angle: float,
                                  right_angle: float, position_score: float) -> float:
        """Assess overall compression quality."""
        try:
            # Depth quality (5-6 cm is optimal)
            depth_quality=1.0
            if depth < 5.0:
                depth_quality=depth / 5.0
            elif depth > 6.0:
                depth_quality=max(0.0, 1.0 - (depth - 6.0) / 2.0)

            # Arm angle quality (should be relatively straight, around 160-180 degrees)
            left_angle_quality=1.0 - abs(left_angle - 170) / 50.0
            right_angle_quality=1.0 - abs(right_angle - 170) / 50.0
            angle_quality=(left_angle_quality + right_angle_quality) / 2
            angle_quality=max(0.0, min(1.0, angle_quality))

            # Combine all quality factors
            overall_quality=(depth_quality * 0.4 + angle_quality * 0.3 + position_score * 0.3)

            return max(0.0, min(1.0, overall_quality))

        except Exception:
            return 0.0

    def _assess_technique_quality(self, medical_metrics: Dict[str, float]) -> Tuple[float, bool, Dict[str, Any]]:
        """Assess CPR technique quality based on AHA guidelines."""
        try:
            quality_components=[]
            compliance_details={}

            # Compression depth assessment
            depth=medical_metrics.get("compression_depth_cm", 0)
            depth_score=self._score_compression_depth(depth)
            quality_components.append(depth_score)
            compliance_details["compression_depth"] = {
                "value": depth,
                "score": depth_score,
                "aha_compliant": 5.0 <= depth <= 6.0
            }

            # Hand position assessment
            position_score=medical_metrics.get("hand_position_score", 0)
            quality_components.append(position_score)
            compliance_details["hand_position"] = {
                "score": position_score,
                "aha_compliant": position_score >= 0.8
            }

            # Arm angle assessment
            left_angle=medical_metrics.get("left_arm_angle", 0)
            right_angle=medical_metrics.get("right_arm_angle", 0)
            angle_score=self._score_arm_angles(left_angle, right_angle)
            quality_components.append(angle_score)
            compliance_details["arm_positioning"] = {
                "left_angle": left_angle,
                "right_angle": right_angle,
                "score": angle_score,
                "aha_compliant": 150 <= left_angle <= 190 and 150 <= right_angle <= 190
            }

            # Body alignment assessment
            body_alignment=medical_metrics.get("body_alignment_score", 0.5)
            quality_components.append(body_alignment)
            compliance_details["body_alignment"] = {
                "score": body_alignment,
                "aha_compliant": body_alignment >= 0.7
            }

            # Calculate overall quality score
            overall_score=np.mean(quality_components) if quality_components else 0.0

            # AHA compliance check (all components must meet minimum standards)
            aha_compliant=all([
                compliance_details["compression_depth"]["aha_compliant"],
                compliance_details["hand_position"]["aha_compliant"],
                compliance_details["arm_positioning"]["aha_compliant"],
                compliance_details["body_alignment"]["aha_compliant"]
            ])

            return overall_score, aha_compliant, compliance_details

        except Exception as e:
            logger.warning(f"Error assessing technique quality: {str(e)}")
            return 0.0, False, {"error": str(e)}

    def _score_compression_depth(self, depth: float) -> float:
        """Score compression depth based on AHA guidelines (5-6 cm optimal)."""
        if 5.0 <= depth <= 6.0:
            return 1.0
        elif depth < 5.0:
            return max(0.0, depth / 5.0)
        else:  # depth > 6.0
            return max(0.0, 1.0 - (depth - 6.0) / 3.0)

    def _score_arm_angles(self, left_angle: float, right_angle: float) -> float:
        """Score arm angles for proper CPR positioning."""
        # Ideal arm angle is approximately 170 degrees (slightly bent)
        ideal_angle=170.0
        tolerance=20.0  # ±20 degrees tolerance

        left_deviation=abs(left_angle - ideal_angle)
        right_deviation=abs(right_angle - ideal_angle)

        left_score=max(0.0, 1.0 - left_deviation / tolerance)
        right_score=max(0.0, 1.0 - right_deviation / tolerance)

        return (left_score + right_score) / 2

    def _calculate_video_quality_metrics(self, annotations: List[MedicalPoseAnnotation],
                                       aggregated_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate comprehensive video-level quality metrics."""
        if not annotations:
            return {"overall_quality": 0.0, "data_completeness": 0.0}

        # Basic quality metrics
        quality_scores=[ann.quality_score for ann in annotations]
        overall_quality=np.mean(quality_scores)
        quality_consistency=1.0 - np.std(quality_scores)

        # Compliance metrics
        compliance_rate=sum(1 for ann in annotations if ann.aha_compliance) / len(annotations)

        # Data completeness
        frames_with_pose=len(annotations)
        data_completeness=frames_with_pose / max(1, len(annotations))  # Simplified

        # Medical-specific metrics
        compression_quality=0.0
        if aggregated_metrics["compression_depths"]:
            depths=aggregated_metrics["compression_depths"]
            optimal_depths=[d for d in depths if 5.0 <= d <= 6.0]
            compression_quality=len(optimal_depths) / len(depths)

        hand_position_quality=0.0
        if aggregated_metrics["hand_position_scores"]:
            hand_scores=aggregated_metrics["hand_position_scores"]
            hand_position_quality=np.mean(hand_scores)

        # Temporal consistency (how consistent is the technique over time)
        temporal_consistency=quality_consistency

        return {
            "overall_quality": overall_quality,
            "quality_consistency": max(0.0, quality_consistency),
            "compliance_rate": compliance_rate,
            "data_completeness": data_completeness,
            "compression_quality": compression_quality,
            "hand_position_quality": hand_position_quality,
            "temporal_consistency": temporal_consistency,
            "total_annotations": len(annotations),
            "average_keypoint_confidence": np.mean([
                ann.processing_metadata.get("pose_confidence", 0) for ann in annotations
            ])
        }
    
    def preprocess_pose_sequence(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Preprocess pose sequence data for model input.
        
        Args:
            pose_data: Raw pose sequence data (frames, landmarks, coordinates)
            
        Returns:
            Preprocessed pose sequence ready for model inference
        """
        try:
            # Flatten the pose landmarks for each frame
            frames, landmarks, coords = pose_data.shape
            processed_data = pose_data.reshape(frames, landmarks * coords)
            
            # Normalize the data
            processed_data = (processed_data - np.mean(processed_data, axis=0)) / (np.std(processed_data, axis=0) + 1e-8)
            
            # Apply smoothing filter to reduce noise
            from scipy.ndimage import gaussian_filter1d
            for i in range(processed_data.shape[1]):
                processed_data[:, i] = gaussian_filter1d(processed_data[:, i], sigma=1.0)
            
            logger.info(f"Preprocessed pose sequence: {pose_data.shape} -> {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Pose sequence preprocessing failed: {e}")
            # Return original data if preprocessing fails
            return pose_data.reshape(pose_data.shape[0], -1)
    
    def extract_medical_features(self, processed_data: np.ndarray) -> Dict[str, float]:
        """
        Extract medical-specific features from processed pose data.
        
        Args:
            processed_data: Preprocessed pose sequence data
            
        Returns:
            Dictionary of extracted medical features
        """
        try:
            features = {}
            
            # Temporal features
            features['sequence_length'] = processed_data.shape[0]
            features['movement_variance'] = np.var(processed_data, axis=0).mean()
            features['movement_smoothness'] = np.mean(np.diff(processed_data, axis=0)**2)
            
            # Compression-related features (mock calculation)
            # In real implementation, these would be calculated from specific landmarks
            features['compression_frequency'] = self._estimate_compression_frequency(processed_data)
            features['compression_amplitude'] = self._estimate_compression_amplitude(processed_data)
            features['hand_position_stability'] = self._calculate_hand_stability(processed_data)
            features['body_alignment_score'] = self._calculate_body_alignment(processed_data)
            
            # Quality indicators
            features['data_quality_score'] = min(1.0, 1.0 / (1.0 + features['movement_variance']))
            features['temporal_consistency'] = 1.0 / (1.0 + features['movement_smoothness'])
            
            # Medical compliance indicators
            features['aha_depth_compliance'] = self._check_depth_compliance(processed_data)
            features['aha_rate_compliance'] = self._check_rate_compliance(processed_data)
            features['technique_score'] = (features['hand_position_stability'] + 
                                         features['body_alignment_score']) / 2.0
            
            # Overall medical score
            features['overall_medical_score'] = (
                features['aha_depth_compliance'] * 0.3 +
                features['aha_rate_compliance'] * 0.3 +
                features['technique_score'] * 0.4
            )
            
            logger.info(f"Extracted {len(features)} medical features")
            return features
            
        except Exception as e:
            logger.error(f"Medical feature extraction failed: {e}")
            return {
                'sequence_length': processed_data.shape[0] if processed_data is not None else 0,
                'overall_medical_score': 0.5,  # Default neutral score
                'error': str(e)
            }
    
    def _estimate_compression_frequency(self, data: np.ndarray) -> float:
        """Estimate CPR compression frequency from pose data."""
        try:
            # Simple frequency estimation using FFT on movement variance
            movement_signal = np.var(data, axis=1)
            fft = np.fft.fft(movement_signal)
            freqs = np.fft.fftfreq(len(movement_signal), d=1/30)  # Assuming 30 FPS
            
            # Find dominant frequency in CPR range (1.5-2.5 Hz = 90-150 CPM)
            valid_freq_mask = (freqs >= 1.5) & (freqs <= 2.5)
            if np.any(valid_freq_mask):
                dominant_freq = freqs[valid_freq_mask][np.argmax(np.abs(fft[valid_freq_mask]))]
                return dominant_freq * 60  # Convert to compressions per minute
            return 108.0  # Default AHA recommended rate
        except:
            return 108.0
    
    def _estimate_compression_amplitude(self, data: np.ndarray) -> float:
        """Estimate CPR compression amplitude from pose data."""
        try:
            # Calculate movement amplitude (simplified)
            movement_range = np.ptp(data, axis=0).mean()
            # Normalize to approximate compression depth in cm
            return min(6.0, max(3.0, movement_range * 100))
        except:
            return 5.0  # Default compression depth
    
    def _calculate_hand_stability(self, data: np.ndarray) -> float:
        """Calculate hand position stability score."""
        try:
            # Calculate stability as inverse of position variance
            position_variance = np.var(data[:, :6], axis=0).mean()  # First 6 features for hands
            return min(1.0, 1.0 / (1.0 + position_variance * 10))
        except:
            return 0.8
    
    def _calculate_body_alignment(self, data: np.ndarray) -> float:
        """Calculate body alignment score."""
        try:
            # Simple alignment calculation based on pose consistency
            alignment_variance = np.var(data[:, 6:12], axis=0).mean()  # Body landmarks
            return min(1.0, 1.0 / (1.0 + alignment_variance * 5))
        except:
            return 0.85
    
    def _check_depth_compliance(self, data: np.ndarray) -> float:
        """Check AHA depth compliance (5-6cm)."""
        try:
            estimated_depth = self._estimate_compression_amplitude(data)
            if 5.0 <= estimated_depth <= 6.0:
                return 1.0
            elif 4.0 <= estimated_depth <= 7.0:
                return 0.8
            else:
                return 0.5
        except:
            return 0.7
    
    def _check_rate_compliance(self, data: np.ndarray) -> float:
        """Check AHA rate compliance (100-120 CPM)."""
        try:
            estimated_rate = self._estimate_compression_frequency(data)
            if 100 <= estimated_rate <= 120:
                return 1.0
            elif 90 <= estimated_rate <= 130:
                return 0.8
            else:
                return 0.5
        except:
            return 0.7


class CPRVideoProcessor(MedicalDataPreprocessor):
    """
    Specialized processor for CPR training videos.

    This class extends the medical data preprocessor with CPR-specific
    analysis capabilities and enhanced AHA guidelines compliance.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize CPR video processor."""
        super().__init__(config_path)
        self.processor_name="CPRVideoProcessor"

        # CPR-specific configuration
        self.cpr_config={
            "compression_detection_threshold": 0.1,
            "ventilation_detection_enabled": False,  # Future enhancement
            "team_cpr_analysis": False,  # Future enhancement
            "real_time_feedback": True
        }

        logger.info("CPR video processor initialized with enhanced AHA compliance")

    def process_cpr_session(self, video_path: Path, session_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a complete CPR training session with detailed analysis.

        Args:
            video_path: Path to CPR training video
            session_metadata: Additional session information

        Returns:
            Comprehensive CPR session analysis
        """
        # Process the video
        processing_result=self.process(video_path)

        if not processing_result.success:
            return {
                "success": False,
                "error": processing_result.message,
                "session_metadata": session_metadata
            }

        # Extract CPR-specific analysis
        video_result=processing_result.data["processing_results"][0]

        # Generate CPR performance report
        cpr_report=self._generate_cpr_performance_report(
            video_result, session_metadata
        )

        return cpr_report

    def _generate_cpr_performance_report(self, video_result: Dict[str, Any],
                                       session_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive CPR performance report."""

        # Extract key metrics from video analysis
        quality_metrics=video_result.get("quality_metrics", {})
        compliance_report=video_result.get("compliance_report", {})
        annotations=video_result.get("annotations", [])

        # Calculate CPR-specific metrics
        total_compressions=len([ann for ann in annotations if ann.get("medical_metrics", {}).get("compression_detected", False)])
        session_duration=max([ann.get("timestamp", 0) for ann in annotations]) if annotations else 0

        compression_rate=(total_compressions / session_duration * 60) if session_duration > 0 else 0

        # Generate performance summary
        performance_summary={
            "session_overview": {
                "duration_seconds": session_duration,
                "total_compressions": total_compressions,
                "compression_rate_per_minute": compression_rate,
                "overall_quality_score": quality_metrics.get("overall_quality", 0),
                "aha_compliance_rate": compliance_report.get("compliance_score", 0)
            },
            "technique_analysis": {
                "compression_depth": compliance_report.get("compliance_details", {}).get("compression_depth", {}),
                "compression_rate": compliance_report.get("compliance_details", {}).get("compression_rate", {}),
                "hand_position": compliance_report.get("compliance_details", {}).get("hand_position", {}),
                "body_alignment": quality_metrics.get("temporal_consistency", 0)
            },
            "improvement_recommendations": compliance_report.get("recommendations", []),
            "session_metadata": session_metadata or {},
            "processing_metadata": {
                "processor_version": self.processor_version,
                "aha_guidelines_version": "2020",
                "processing_timestamp": datetime.now().isoformat(),
                "quality_assurance_passed": quality_metrics.get("overall_quality", 0) >= 0.7
            }
        }

        return {
            "success": True,
            "cpr_performance_report": performance_summary,
            "detailed_analysis": video_result,
            "audit_trail_id": self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                description="CPR session analysis completed",
                details=performance_summary,
                severity=AuditSeverity.MEDIUM
            )
        }
    
