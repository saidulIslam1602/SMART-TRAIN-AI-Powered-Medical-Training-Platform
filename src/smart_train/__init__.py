"""
SMART-TRAIN: AI-Powered Medical Training Platform
Simulation-based Medical AI for Real-time Training Analysis and Improvement Network

This package provides AI-powered analysis and feedback for medical training,
specifically focused on CPR and emergency response training.
"""

__version__ = "1.0.0"
__author__ = "SMART-TRAIN Development Team"
__email__ = "contact@smart-train.ai"
__license__ = "MIT"

# Medical compliance information
__medical_standards__ = [
    "ISO 13485:2016",  # Medical Device Quality Management
    "IEC 62304:2006",  # Medical Device Software Lifecycle
    "AHA CPR Guidelines 2020"  # American Heart Association CPR Guidelines
]

__compliance__ = [
    "GDPR",  # General Data Protection Regulation
    "HIPAA",  # Health Insurance Portability and Accountability Act
    "FDA 510(k) Ready"  # FDA Medical Device Approval Ready
]

# Core imports for easy access
from .core.exceptions import SmartTrainException
from .core.config import SmartTrainConfig
from .core.logging import get_logger

__all__ = [
    "SmartTrainException",
    "SmartTrainConfig",
    "get_logger",
    "__version__",
    "__medical_standards__",
    "__compliance__"
]
