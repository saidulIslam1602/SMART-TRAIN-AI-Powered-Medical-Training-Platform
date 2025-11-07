"""
Custom exceptions for SMART-TRAIN platform.

This module defines the exception hierarchy used throughout the platform,
with specific focus on medical compliance and data validation errors.
"""

from typing import Optional, Dict, Any
import traceback
from datetime import datetime


class SmartTrainException(Exception):
    """
    Base exception class for all SMART-TRAIN related errors.
    
    This exception provides enhanced error tracking capabilities
    required for medical device compliance and audit trails.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize SmartTrainException.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for tracking
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "ST_GENERIC_ERROR"
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        self.traceback_str = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and audit trails."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_str,
            "cause": str(self.cause) if self.cause else None
        }


class MedicalComplianceError(SmartTrainException):
    """
    Exception raised when medical compliance requirements are violated.
    
    This exception is used for ISO 13485, IEC 62304, and other medical
    device standard violations that require immediate attention.
    """
    
    def __init__(
        self, 
        message: str, 
        compliance_standard: str,
        violation_type: str,
        severity: str = "HIGH",
        **kwargs
    ):
        """
        Initialize MedicalComplianceError.
        
        Args:
            message: Description of compliance violation
            compliance_standard: Which standard was violated (e.g., "ISO 13485")
            violation_type: Type of violation (e.g., "data_integrity")
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        error_code = f"MC_{compliance_standard.replace(' ', '_')}_{violation_type.upper()}"
        context = {
            "compliance_standard": compliance_standard,
            "violation_type": violation_type,
            "severity": severity
        }
        context.update(kwargs.get("context", {}))
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            cause=kwargs.get("cause")
        )
        
        self.compliance_standard = compliance_standard
        self.violation_type = violation_type
        self.severity = severity


class DataValidationError(SmartTrainException):
    """
    Exception raised when data validation fails.
    
    This exception is used for medical data validation failures,
    including pose estimation errors, video quality issues, etc.
    """
    
    def __init__(
        self, 
        message: str, 
        validation_type: str,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize DataValidationError.
        
        Args:
            message: Description of validation failure
            validation_type: Type of validation that failed
            field_name: Name of the field that failed validation
            expected_value: Expected value
            actual_value: Actual value that failed validation
        """
        error_code = f"DV_{validation_type.upper()}"
        context = {
            "validation_type": validation_type,
            "field_name": field_name,
            "expected_value": expected_value,
            "actual_value": actual_value
        }
        context.update(kwargs.get("context", {}))
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            cause=kwargs.get("cause")
        )
        
        self.validation_type = validation_type
        self.field_name = field_name
        self.expected_value = expected_value
        self.actual_value = actual_value


class ModelInferenceError(SmartTrainException):
    """Exception raised when AI model inference fails."""
    
    def __init__(self, message: str, model_name: str, **kwargs):
        error_code = f"MI_{model_name.upper()}_INFERENCE_FAILED"
        context = {"model_name": model_name}
        context.update(kwargs.get("context", {}))
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            cause=kwargs.get("cause")
        )


class ConfigurationError(SmartTrainException):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        error_code = "CF_CONFIGURATION_ERROR"
        context = {"config_key": config_key}
        context.update(kwargs.get("context", {}))
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            cause=kwargs.get("cause")
        )


class APIError(SmartTrainException):
    """Exception raised for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        error_code = f"API_{status_code}_ERROR"
        context = {
            "status_code": status_code,
            "endpoint": endpoint
        }
        context.update(kwargs.get("context", {}))
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            cause=kwargs.get("cause")
        )
        
        self.status_code = status_code
        self.endpoint = endpoint
