"""
Structured logging configuration for SMART-TRAIN platform.

This module provides medical-grade logging capabilities with audit trails,
compliance tracking, and structured log formats required for medical devices.
"""

import logging
import structlog
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import os

from .exceptions import SmartTrainException


class MedicalAuditProcessor:
    """
    Processor for medical audit trail logging.
    
    This processor ensures all logs meet medical device compliance
    requirements for audit trails and traceability.
    """
    
    def __call__(self, logger, method_name, event_dict):
        """Process log entry for medical compliance."""
        # Add audit trail information
        event_dict["audit_timestamp"] = datetime.utcnow().isoformat()
        event_dict["audit_level"] = event_dict.get("level", "info").upper()
        event_dict["compliance_required"] = True
        
        # Add medical context if available
        if "medical_context" not in event_dict:
            event_dict["medical_context"] = {
                "patient_data_involved": False,
                "training_session": False,
                "model_inference": False
            }
        
        return event_dict


class ComplianceFilter(logging.Filter):
    """
    Logging filter for medical compliance requirements.
    
    This filter ensures sensitive medical data is properly handled
    and compliance violations are flagged.
    """
    
    def filter(self, record):
        """Filter log records for compliance."""
        # Add compliance metadata
        record.compliance_checked = True
        record.medical_grade = True
        
        # Check for sensitive data patterns
        message = str(record.getMessage()).lower()
        sensitive_patterns = [
            "patient", "medical_record", "phi", "pii", 
            "ssn", "medical_id", "diagnosis"
        ]
        
        record.contains_sensitive_data = any(
            pattern in message for pattern in sensitive_patterns
        )
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_audit_trail: bool = True,
    enable_console: bool = True
) -> None:
    """
    Setup structured logging for SMART-TRAIN platform.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_audit_trail: Enable medical audit trail logging
        enable_console: Enable console logging
    """
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add medical audit processor if enabled
    if enable_audit_trail:
        processors.append(MedicalAuditProcessor())
    
    # Configure output format
    if enable_console:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Add compliance filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(ComplianceFilter())
    
    # Setup file handlers if log directory is specified
    if log_dir:
        # Application logs
        app_handler = logging.FileHandler(log_dir / "smart_train.log")
        app_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        root_logger.addHandler(app_handler)
        
        # Audit trail logs (separate file for compliance)
        if enable_audit_trail:
            audit_handler = logging.FileHandler(log_dir / "audit_trail.log")
            audit_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
                )
            )
            audit_handler.addFilter(ComplianceFilter())
            
            # Create audit logger
            audit_logger = logging.getLogger("smart_train.audit")
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog BoundLogger instance
    """
    return structlog.get_logger(name)


class MedicalLogger:
    """
    Specialized logger for medical operations.
    
    This logger provides additional methods for logging medical-specific
    events with proper compliance and audit trail support.
    """
    
    def __init__(self, name: str):
        """Initialize medical logger."""
        self.logger = get_logger(name)
        self.audit_logger = logging.getLogger("smart_train.audit")
    
    def log_medical_event(
        self, 
        event_type: str, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """
        Log a medical event with proper audit trail.
        
        Args:
            event_type: Type of medical event
            message: Event description
            context: Additional context data
            level: Log level
        """
        log_data = {
            "event_type": event_type,
            "message": message,
            "medical_context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
            "compliance_event": True
        }
        
        # Log to main logger
        getattr(self.logger, level.lower())(message, **log_data)
        
        # Log to audit trail (avoid 'message' conflict)
        audit_log_data = {k: v for k, v in log_data.items() if k != 'message'}
        self.audit_logger.info(
            f"MEDICAL_EVENT: {event_type} - {message}",
            extra=audit_log_data
        )
    
    def log_model_inference(
        self, 
        model_name: str, 
        input_data_hash: str,
        output_summary: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log AI model inference for audit trail."""
        self.log_medical_event(
            event_type="MODEL_INFERENCE",
            message=f"Model {model_name} inference completed",
            context={
                "model_name": model_name,
                "input_data_hash": input_data_hash,
                "output_summary": output_summary,
                "performance_metrics": performance_metrics or {}
            }
        )
    
    def log_data_processing(
        self, 
        operation: str, 
        data_source: str,
        records_processed: int,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data processing operations."""
        self.log_medical_event(
            event_type="DATA_PROCESSING",
            message=f"Data processing: {operation} on {data_source}",
            context={
                "operation": operation,
                "data_source": data_source,
                "records_processed": records_processed,
                "quality_metrics": quality_metrics or {}
            }
        )
    
    def log_compliance_check(
        self, 
        check_type: str, 
        result: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log compliance check results."""
        level = "info" if result else "warning"
        self.log_medical_event(
            event_type="COMPLIANCE_CHECK",
            message=f"Compliance check {check_type}: {'PASSED' if result else 'FAILED'}",
            context={
                "check_type": check_type,
                "result": result,
                "details": details or {}
            },
            level=level
        )
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log exceptions with medical compliance context."""
        if isinstance(exception, SmartTrainException):
            # Enhanced logging for SmartTrain exceptions
            self.log_medical_event(
                event_type="EXCEPTION",
                message=f"Exception: {exception.message}",
                context={
                    "exception_type": exception.__class__.__name__,
                    "error_code": exception.error_code,
                    "exception_context": exception.context,
                    "additional_context": context or {}
                },
                level="error"
            )
        else:
            # Standard exception logging
            self.log_medical_event(
                event_type="EXCEPTION",
                message=f"Exception: {str(exception)}",
                context={
                    "exception_type": exception.__class__.__name__,
                    "additional_context": context or {}
                },
                level="error"
            )
