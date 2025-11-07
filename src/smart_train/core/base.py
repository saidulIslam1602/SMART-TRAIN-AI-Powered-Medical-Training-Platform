"""
Base classes and utilities for SMART-TRAIN platform.

This module provides base classes that implement common functionality
and design patterns used throughout the platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from datetime import datetime
import uuid
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

from .exceptions import SmartTrainException, DataValidationError, MedicalComplianceError
from .logging import MedicalLogger

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class ProcessingResult:
    """
    Standard result object for processing operations.
    
    This class provides a consistent interface for returning results
    from various processing operations with metadata and audit information.
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None
    timestamp: Optional[datetime] = None
    operation_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.operation_id is None:
            self.operation_id = str(uuid.uuid4())
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        if self.timestamp:
            result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict


class BaseModel(ABC):
    """
    Base class for all AI models in SMART-TRAIN platform.
    
    This class provides common functionality for model loading, inference,
    and medical compliance tracking.
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_id = f"{model_name}_v{model_version}"
        self.logger = MedicalLogger(f"model.{model_name}")
        self.is_loaded = False
        self.load_timestamp: Optional[datetime] = None
        self.inference_count = 0
        
        # Medical compliance tracking
        self.medical_device_compliant = True
        self.fda_cleared = False
        self.iso_13485_compliant = True
        
        logger.info("Model initialized", model_id=self.model_id)
    
    @abstractmethod
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load the model from storage.
        
        Args:
            model_path: Path to model files
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> ProcessingResult:
        """
        Perform model inference.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            ProcessingResult with prediction results
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before inference.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Base validation - override in subclasses
        return input_data is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "load_timestamp": self.load_timestamp.isoformat() if self.load_timestamp else None,
            "inference_count": self.inference_count,
            "medical_device_compliant": self.medical_device_compliant,
            "fda_cleared": self.fda_cleared,
            "iso_13485_compliant": self.iso_13485_compliant
        }
    
    def _log_inference(self, input_hash: str, result: ProcessingResult) -> None:
        """Log inference for audit trail."""
        self.inference_count += 1
        
        self.logger.log_model_inference(
            model_name=self.model_id,
            input_data_hash=input_hash,
            output_summary={
                "success": result.success,
                "processing_time_ms": result.processing_time_ms,
                "operation_id": result.operation_id
            },
            performance_metrics={
                "inference_count": self.inference_count
            }
        )
    
    def _calculate_input_hash(self, input_data: Any) -> str:
        """Calculate hash of input data for audit trail."""
        try:
            # Convert input to string representation
            if hasattr(input_data, 'tobytes'):
                # NumPy array or similar
                data_str = input_data.tobytes()
            else:
                # Convert to JSON string
                data_str = json.dumps(input_data, sort_keys=True, default=str).encode()
            
            return hashlib.sha256(data_str).hexdigest()[:16]
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(input_data).encode()).hexdigest()[:16]


class BaseProcessor(ABC):
    """
    Base class for data processors in SMART-TRAIN platform.
    
    This class provides common functionality for data processing operations
    with medical compliance and audit trail support.
    """
    
    def __init__(self, processor_name: str, processor_version: str = "1.0.0"):
        """
        Initialize base processor.
        
        Args:
            processor_name: Name of the processor
            processor_version: Version of the processor
        """
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.processor_id = f"{processor_name}_v{processor_version}"
        self.logger = MedicalLogger(f"processor.{processor_name}")
        
        # Processing statistics
        self.processing_count = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0
        
        logger.info("Processor initialized", processor_id=self.processor_id)
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """
        Process input data.
        
        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult with processing results
        """
        pass
    
    def validate_input(self, input_data: Any) -> ProcessingResult:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            ProcessingResult indicating validation success/failure
        """
        result = ProcessingResult(
            success=True,
            message="Input validation passed"
        )
        
        if input_data is None:
            result.add_error("Input data is None")
        
        return result
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        avg_processing_time = (
            self.total_processing_time_ms / self.processing_count
            if self.processing_count > 0 else 0.0
        )
        
        return {
            "processor_name": self.processor_name,
            "processor_version": self.processor_version,
            "processor_id": self.processor_id,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "total_processing_time_ms": self.total_processing_time_ms,
            "average_processing_time_ms": avg_processing_time,
            "error_rate": self.error_count / max(self.processing_count, 1)
        }
    
    def _log_processing(self, operation: str, result: ProcessingResult) -> None:
        """Log processing operation for audit trail."""
        self.processing_count += 1
        
        if not result.success:
            self.error_count += 1
        
        if result.processing_time_ms:
            self.total_processing_time_ms += result.processing_time_ms
        
        self.logger.log_data_processing(
            operation=operation,
            data_source=self.processor_id,
            records_processed=1,
            quality_metrics={
                "success": result.success,
                "processing_time_ms": result.processing_time_ms,
                "error_count": len(result.errors) if result.errors else 0,
                "warning_count": len(result.warnings) if result.warnings else 0
            }
        )


class MedicalDataValidator:
    """
    Validator for medical data compliance and quality.
    
    This class provides validation methods for ensuring medical data
    meets compliance requirements and quality standards.
    """
    
    def __init__(self):
        """Initialize medical data validator."""
        self.logger = MedicalLogger("medical_validator")
    
    def validate_medical_compliance(
        self, 
        data: Dict[str, Any], 
        compliance_standards: List[str]
    ) -> ProcessingResult:
        """
        Validate data against medical compliance standards.
        
        Args:
            data: Data to validate
            compliance_standards: List of compliance standards to check
            
        Returns:
            ProcessingResult with validation results
        """
        result = ProcessingResult(
            success=True,
            message="Medical compliance validation passed"
        )
        
        for standard in compliance_standards:
            if standard == "ISO_13485":
                self._validate_iso_13485(data, result)
            elif standard == "IEC_62304":
                self._validate_iec_62304(data, result)
            elif standard == "HIPAA":
                self._validate_hipaa(data, result)
            elif standard == "GDPR":
                self._validate_gdpr(data, result)
        
        # Log compliance check
        self.logger.log_compliance_check(
            check_type="medical_compliance",
            result=result.success,
            details={
                "standards_checked": compliance_standards,
                "errors": result.errors,
                "warnings": result.warnings
            }
        )
        
        return result
    
    def _validate_iso_13485(self, data: Dict[str, Any], result: ProcessingResult) -> None:
        """Validate ISO 13485 compliance."""
        # Check for required audit trail information
        required_fields = ["timestamp", "operation_id", "data_source"]
        
        for field in required_fields:
            if field not in data:
                result.add_error(f"ISO 13485: Missing required field '{field}'")
        
        # Check for data integrity
        if "checksum" not in data and "hash" not in data:
            result.add_warning("ISO 13485: No data integrity check found")
    
    def _validate_iec_62304(self, data: Dict[str, Any], result: ProcessingResult) -> None:
        """Validate IEC 62304 compliance."""
        # Check for software lifecycle documentation
        if "software_version" not in data:
            result.add_error("IEC 62304: Missing software version information")
        
        if "validation_status" not in data:
            result.add_error("IEC 62304: Missing validation status")
    
    def _validate_hipaa(self, data: Dict[str, Any], result: ProcessingResult) -> None:
        """Validate HIPAA compliance."""
        # Check for PHI (Protected Health Information)
        phi_patterns = [
            "patient_id", "medical_record_number", "ssn", 
            "date_of_birth", "phone_number", "email"
        ]
        
        for pattern in phi_patterns:
            if pattern in str(data).lower():
                result.add_error(f"HIPAA: Potential PHI detected - {pattern}")
        
        # Check for encryption/anonymization
        if "anonymized" not in data or not data.get("anonymized", False):
            result.add_warning("HIPAA: Data anonymization status unclear")
    
    def _validate_gdpr(self, data: Dict[str, Any], result: ProcessingResult) -> None:
        """Validate GDPR compliance."""
        # Check for consent information
        if "consent_given" not in data:
            result.add_error("GDPR: Missing consent information")
        
        # Check for data subject rights
        if "data_retention_period" not in data:
            result.add_warning("GDPR: Data retention period not specified")


class ConfigurableComponent:
    """
    Mixin class for components that need configuration support.
    
    This class provides configuration management capabilities
    that can be mixed into other classes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configurable component.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config or {}
        self._validate_config()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration (override in subclasses)."""
        pass
