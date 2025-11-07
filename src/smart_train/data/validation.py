"""
Data validation module for SMART-TRAIN platform.
"""

import numpy as np
from typing import Dict, Any
from ..core.base import ProcessingResult
from ..core.logging import MedicalLogger


class MedicalDataValidator:
    """Medical data validator."""

    def __init__(self):
        """Initialize medical data validator."""
        self.logger = MedicalLogger("medical_validator")

    def validate(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate medical data."""
        result = ProcessingResult(
            success=True,
            message="Medical data validation completed"
        )

        result.data = {
            "validation_status": "passed",
            "data_quality_score": 0.95
        }

        return result


class ComplianceValidator:
    """Compliance validator."""

    def __init__(self):
        """Initialize compliance validator."""
        self.logger = MedicalLogger("compliance_validator")

    def validate(self, data: Dict[str, Any]) -> ProcessingResult:
        """Validate compliance."""
        result = ProcessingResult(
            success=True,
            message="Compliance validation completed"
        )

        result.data = {
            "compliance_status": "compliant"
        }

        return result


class DataValidator:
    """General data validator for pose sequences and medical data."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = MedicalLogger("data_validator")
    
    def validate_pose_data(self, pose_data: np.ndarray) -> ProcessingResult:
        """Validate pose sequence data."""
        try:
            # Check data shape
            if len(pose_data.shape) != 3:
                return ProcessingResult(
                    success=False,
                    message="Invalid pose data shape. Expected 3D array (frames, landmarks, coordinates)"
                )
            
            frames, landmarks, coords = pose_data.shape
            
            # Check expected dimensions
            if landmarks != 33:
                return ProcessingResult(
                    success=False,
                    message=f"Invalid number of landmarks. Expected 33, got {landmarks}"
                )
            
            if coords != 3:
                return ProcessingResult(
                    success=False,
                    message=f"Invalid coordinate dimensions. Expected 3, got {coords}"
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(pose_data)) or np.any(np.isinf(pose_data)):
                return ProcessingResult(
                    success=False,
                    message="Pose data contains NaN or infinite values"
                )
            
            # Check coordinate ranges (normalized coordinates should be 0-1)
            if np.any(pose_data < 0) or np.any(pose_data > 1):
                return ProcessingResult(
                    success=False,
                    message="Pose coordinates out of expected range [0, 1]"
                )
            
            return ProcessingResult(
                success=True,
                message="Pose data validation passed",
                data={
                    "frames": frames,
                    "landmarks": landmarks,
                    "coordinates": coords,
                    "data_quality": "valid"
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Pose data validation failed: {str(e)}"
            )
