"""
Core module for SMART-TRAIN platform.

This module contains the fundamental components used throughout the platform:
- Configuration management
- Exception handling
- Logging infrastructure
- Base classes and utilities
"""

from .config import SmartTrainConfig
from .exceptions import SmartTrainException, MedicalComplianceError, DataValidationError
from .logging import get_logger, setup_logging
from .base import BaseModel, BaseProcessor

__all__ = [
    "SmartTrainConfig",
    "SmartTrainException",
    "MedicalComplianceError",
    "DataValidationError",
    "get_logger",
    "setup_logging",
    "BaseModel",
    "BaseProcessor"
]
