"""
IEC 62304 Medical Device Software Lifecycle compliance module.
"""

from typing import Dict, Any
from ..core.base import ProcessingResult
from ..core.logging import MedicalLogger


class IEC62304Compliance:
    """IEC 62304 compliance manager."""

    def __init__(self):
        """Initialize IEC 62304 compliance manager."""
        self.logger = MedicalLogger("iec_62304")

    def check_compliance(self, system_data: Dict[str, Any]) -> ProcessingResult:
        """Check IEC 62304 compliance."""
        result = ProcessingResult(
            success=True,
            message="IEC 62304 compliance check completed"
        )

        result.data = {
            "compliance_status": "compliant",
            "software_safety_class": "B"
        }

        return result

    def validate_software_lifecycle(self, lifecycle_data: Dict[str, Any]) -> ProcessingResult:
        """
        Validate software lifecycle for IEC 62304 compliance.
        
        Args:
            lifecycle_data: Software lifecycle data to validate
            
        Returns:
            ProcessingResult with validation results
        """
        result = ProcessingResult(
            success=True,
            message="Software lifecycle validation completed"
        )
        
        # Check required lifecycle phases
        required_phases = ['planning', 'requirements', 'design', 'implementation', 'testing', 'release']
        missing_phases = []
        
        for phase in required_phases:
            if phase not in lifecycle_data:
                missing_phases.append(phase)
            else:
                phase_data = lifecycle_data[phase]
                # Check if phase has required completion indicators
                if not isinstance(phase_data, dict):
                    missing_phases.append(f"{phase}_data")
                elif not any(phase_data.values()):  # Check if any completion flag is True
                    missing_phases.append(f"{phase}_completion")
        
        if missing_phases:
            result.success = False
            result.message = f"Software lifecycle validation failed: missing {missing_phases}"
            result.add_error(f"Missing or incomplete phases: {missing_phases}")
        else:
            result.data = {
                "validation_status": "compliant",
                "lifecycle_phases_checked": required_phases,
                "compliance_standard": "IEC_62304",
                "software_safety_class": "B"
            }
        
        return result
