"""
Medical compliance framework for SMART-TRAIN platform.

This module provides comprehensive medical device compliance support including:
- ISO 13485 (Medical Device Quality Management)
- IEC 62304 (Medical Device Software Lifecycle)
- FDA 510(k) preparation
- HIPAA compliance
- GDPR compliance
- Audit trail management
"""

from .iso_13485 import ISO13485Compliance
from .iec_62304 import IEC62304Compliance
from .audit_trail import AuditTrailManager
from .risk_management import RiskManager
from .quality_management import QualityManager

__all__ = [
    "ISO13485Compliance",
    "IEC62304Compliance", 
    "AuditTrailManager",
    "RiskManager",
    "QualityManager"
]
