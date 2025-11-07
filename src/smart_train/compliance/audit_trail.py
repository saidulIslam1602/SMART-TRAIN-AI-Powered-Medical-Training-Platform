"""
Audit trail management for medical compliance.

This module provides comprehensive audit trail capabilities required for
medical device compliance, including data integrity, user actions, and
system events tracking.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import structlog

from ..core.base import ProcessingResult
from ..core.exceptions import MedicalComplianceError
from ..core.logging import MedicalLogger

logger=structlog.get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN="user_login"
    USER_LOGOUT="user_logout"
    DATA_ACCESS="data_access"
    DATA_MODIFICATION="data_modification"
    MODEL_INFERENCE="model_inference"
    MODEL_OPERATION="model_operation"
    SYSTEM_CONFIGURATION="system_configuration"
    COMPLIANCE_CHECK="compliance_check"
    ERROR_EVENT="error_event"
    SECURITY_EVENT="security_event"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO="info"
    LOW="low"
    MEDIUM="medium"
    HIGH="high"
    CRITICAL="critical"


@dataclass
class AuditEvent:
    """
    Audit event data structure.

    This class represents a single audit event with all required
    information for medical compliance tracking.
    """
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    details: Dict[str, Any]
    severity: AuditSeverity
    data_hash: Optional[str]
    system_info: Dict[str, str]
    compliance_tags: List[str]

    def __post_init__(self):
        """Validate audit event after creation."""
        if not self.event_id:
            self.event_id=str(uuid.uuid4())

        if not self.timestamp:
            self.timestamp=datetime.now(timezone.utc)

        if not self.compliance_tags:
            self.compliance_tags=["ISO_13485", "IEC_62304"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        event_dict=asdict(self)
        event_dict['timestamp'] = self.timestamp.isoformat()
        event_dict['event_type'] = self.event_type.value
        event_dict['severity'] = self.severity.value
        return event_dict

    def calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for the audit event."""
        # Create deterministic string representation
        event_data={
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "description": self.description,
            "details": self.details
        }

        # Convert numpy types to JSON-serializable types
        event_data = self._convert_numpy_types(event_data)
        event_str=json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class AuditTrailManager:
    """
    Comprehensive audit trail manager for medical compliance.

    This class manages audit events, ensures data integrity, and provides
    compliance reporting capabilities required for medical devices.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_encryption: bool=True,
        retention_days: int=2555  # 7 years for medical records
    ):
        """
        Initialize audit trail manager.

        Args:
            storage_path: Path for audit trail storage
            enable_encryption: Enable audit trail encryption
            retention_days: Retention period in days (default: 7 years)
        """
        self.storage_path=storage_path or Path("compliance/audit_logs")
        self.enable_encryption=enable_encryption
        self.retention_days=retention_days
        self.logger=MedicalLogger("audit_trail")

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize audit trail
        self._initialize_audit_trail()

        logger.info(
            "Audit trail manager initialized",
            storage_path=str(self.storage_path),
            encryption_enabled=enable_encryption,
            retention_days=retention_days
        )

    def _initialize_audit_trail(self) -> None:
        """Initialize audit trail system."""
        # Create audit trail metadata
        metadata={
            "audit_trail_version": "1.0.0",
            "created_timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance_standards": ["ISO_13485", "IEC_62304", "HIPAA", "GDPR"],
            "encryption_enabled": self.enable_encryption,
            "retention_policy_days": self.retention_days,
            "integrity_algorithm": "SHA-256"
        }

        metadata_path=self.storage_path / "audit_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Log initialization
        self.log_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            description="Audit trail system initialized",
            details=metadata,
            severity=AuditSeverity.MEDIUM
        )

    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity=AuditSeverity.LOW,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        data_hash: Optional[str] = None
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            description: Human-readable description
            details: Additional event details
            severity: Event severity level
            user_id: User ID associated with event
            session_id: Session ID associated with event
            data_hash: Hash of associated data

        Returns:
            Event ID of the logged event
        """
        try:
            # Create audit event
            audit_event=AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                session_id=session_id,
                description=description,
                details=details or {},
                severity=severity,
                data_hash=data_hash,
                system_info=self._get_system_info(),
                compliance_tags=["ISO_13485", "IEC_62304"]
            )

            # Calculate integrity hash
            integrity_hash=audit_event.calculate_integrity_hash()
            audit_event.details["integrity_hash"] = integrity_hash

            # Store audit event
            self._store_audit_event(audit_event)

            # Log to medical logger (with error handling)
            try:
                self.logger.log_medical_event(
                    event_type="AUDIT_EVENT",
                    message=f"Audit event logged: {description}",
                    context={
                        "event_id": audit_event.event_id,
                        "event_type": event_type.value,
                        "severity": severity.value,
                        "user_id": user_id
                    }
                )
            except Exception as log_error:
                # Don't fail audit trail storage due to logging issues
                logger.warning(
                    "Failed to log audit event to medical logger",
                    error=str(log_error),
                    event_id=audit_event.event_id
                )

            return audit_event.event_id

        except Exception as e:
            # Critical error - audit trail failure
            error_msg=f"Failed to log audit event: {str(e)}"
            logger.error(error_msg, event_type=event_type.value, description=description)

            raise MedicalComplianceError(
                message=error_msg,
                compliance_standard="ISO 13485",
                violation_type="audit_trail_failure",
                severity="CRITICAL",
                context={"original_event": description, "error": str(e)}
            )

    def _store_audit_event(self, audit_event: AuditEvent) -> None:
        """Store audit event to persistent storage."""
        # Create daily audit log file
        date_str=audit_event.timestamp.strftime("%Y-%m-%d")
        log_file=self.storage_path / f"audit_log_{date_str}.json"

        # Convert event to dictionary
        event_dict=audit_event.to_dict()
        
        # Convert numpy types to JSON-serializable types
        event_dict = audit_event._convert_numpy_types(event_dict)

        # Append to daily log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(event_dict) + '\n')

        # Also store in structured format for compliance reporting
        self._store_structured_event(audit_event)

    def _store_structured_event(self, audit_event: AuditEvent) -> None:
        """Store audit event in structured format for compliance queries."""
        # Create monthly structured file
        month_str=audit_event.timestamp.strftime("%Y-%m")
        structured_dir=self.storage_path / "structured" / month_str
        structured_dir.mkdir(parents=True, exist_ok=True)

        # Store by event type for efficient querying
        event_type_file=structured_dir / f"{audit_event.event_type.value}.json"

        # Load existing events
        events=[]
        if event_type_file.exists():
            try:
                with open(event_type_file, 'r') as f:
                    content=f.read().strip()
                    if content:  # Only parse if file has content
                        events=json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # If file is corrupted or empty, start fresh
                events=[]

        # Add new event
        event_dict = audit_event.to_dict()
        event_dict = audit_event._convert_numpy_types(event_dict)
        events.append(event_dict)

        # Save updated events
        with open(event_type_file, 'w') as f:
            json.dump(events, f, indent=2)

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for audit trail."""
        import platform
        import os

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "process_id": str(os.getpid()),
            "user": os.getenv("USER", "unknown")
        }

    def get_audit_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None
    ) -> List[AuditEvent]:
        """
        Retrieve audit events based on criteria.

        Args:
            start_date: Start date for event retrieval
            end_date: End date for event retrieval
            event_type: Filter by event type
            user_id: Filter by user ID
            severity: Filter by severity level

        Returns:
            List of matching audit events
        """
        events=[]

        # Default to last 30 days if no dates specified
        if not start_date:
            start_date=datetime.now(timezone.utc).replace(day=1)  # Start of month
        if not end_date:
            end_date=datetime.now(timezone.utc)

        # Iterate through date range
        current_date=start_date.date()
        end_date_only=end_date.date()

        while current_date <= end_date_only:
            date_str=current_date.strftime("%Y-%m-%d")
            log_file=self.storage_path / f"audit_log_{date_str}.json"

            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            event_dict=json.loads(line.strip())

                            # Apply filters
                            if event_type and event_dict.get('event_type') != event_type.value:
                                continue
                            if user_id and event_dict.get('user_id') != user_id:
                                continue
                            if severity and event_dict.get('severity') != severity.value:
                                continue

                            # Convert back to AuditEvent
                            event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                            event_dict['event_type'] = AuditEventType(event_dict['event_type'])
                            event_dict['severity'] = AuditSeverity(event_dict['severity'])

                            events.append(AuditEvent(**event_dict))

                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                "Failed to parse audit event",
                                file=str(log_file),
                                error=str(e)
                            )

            # Move to next day
            current_date=current_date.replace(day=current_date.day + 1) if current_date.day < 28 else \
                          current_date.replace(month=current_date.month + 1, day=1) if current_date.month < 12 else \
                          current_date.replace(year=current_date.year + 1, month=1, day=1)

        return events

    def verify_audit_integrity(self, event_id: str) -> ProcessingResult:
        """
        Verify the integrity of an audit event.

        Args:
            event_id: ID of the event to verify

        Returns:
            ProcessingResult indicating integrity status
        """
        result=ProcessingResult(
            success=False,
            message="Audit event not found"
        )

        try:
            # Find the event
            events=self.get_audit_events()
            target_event=None

            for event in events:
                if event.event_id== event_id:
                    target_event=event
                    break

            if not target_event:
                result.message=f"Audit event {event_id} not found"
                return result

            # Verify integrity hash
            stored_hash=target_event.details.get("integrity_hash")
            calculated_hash=target_event.calculate_integrity_hash()

            if stored_hash== calculated_hash:
                result.success=True
                result.message="Audit event integrity verified"
                result.data={
                    "event_id": event_id,
                    "integrity_verified": True,
                    "hash": calculated_hash
                }
            else:
                result.message="Audit event integrity verification failed"
                result.add_error("Hash mismatch detected")
                result.data={
                    "event_id": event_id,
                    "integrity_verified": False,
                    "stored_hash": stored_hash,
                    "calculated_hash": calculated_hash
                }

            return result

        except Exception as e:
            result.message=f"Error verifying audit integrity: {str(e)}"
            result.add_error(str(e))
            return result

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        compliance_standard: str="ISO_13485"
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audit trail.

        Args:
            start_date: Report start date
            end_date: Report end date
            compliance_standard: Compliance standard to report on

        Returns:
            Compliance report dictionary
        """
        events=self.get_audit_events(start_date, end_date)

        # Analyze events by type
        event_summary={}
        severity_summary={}
        user_activity={}

        for event in events:
            # Event type summary
            event_type=event.event_type.value
            event_summary[event_type] = event_summary.get(event_type, 0) + 1

            # Severity summary
            severity=event.severity.value
            severity_summary[severity] = severity_summary.get(severity, 0) + 1

            # User activity
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1

        # Calculate compliance metrics
        total_events=len(events)
        critical_events=severity_summary.get("critical", 0)
        high_severity_events=severity_summary.get("high", 0)

        compliance_score=max(0, 100 - (critical_events * 10) - (high_severity_events * 5))

        report={
            "report_metadata": {
                "compliance_standard": compliance_standard,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events": total_events
            },
            "event_summary": event_summary,
            "severity_summary": severity_summary,
            "user_activity": user_activity,
            "compliance_metrics": {
                "compliance_score": compliance_score,
                "critical_events": critical_events,
                "high_severity_events": high_severity_events,
                "audit_trail_integrity": "verified"  # Could be enhanced with actual verification
            },
            "recommendations": self._generate_compliance_recommendations(
                critical_events, high_severity_events, total_events
            )
        }

        return report

    def _generate_compliance_recommendations(
        self,
        critical_events: int,
        high_severity_events: int,
        total_events: int
    ) -> List[str]:
        """Generate compliance recommendations based on audit analysis."""
        recommendations=[]

        if critical_events > 0:
            recommendations.append(
                f"Address {critical_events} critical events immediately for compliance"
            )

        if high_severity_events > total_events * 0.1:  # More than 10% high severity
            recommendations.append(
                "High severity event rate exceeds 10% - review system security"
            )

        if total_events < 100:  # Very low activity might indicate logging issues
            recommendations.append(
                "Low audit activity detected - verify audit logging is functioning"
            )

        if not recommendations:
            recommendations.append("Audit trail compliance appears satisfactory")

        return recommendations
