"""
Data processing module for SMART-TRAIN platform.

This module provides comprehensive data processing capabilities including:
- Real dataset collection and management
- Medical-grade data preprocessing
- Synthetic data generation
- Data validation and quality assurance
- Compliance-aware data handling
"""

from .collection import RealDatasetCollector, MedicalDatasetManager
from .preprocessing import MedicalDataPreprocessor, CPRVideoProcessor
from .validation import MedicalDataValidator, ComplianceValidator
from .synthetic import SyntheticDataGenerator, CPRScenarioGenerator

__all__ = [
    "RealDatasetCollector",
    "MedicalDatasetManager",
    "MedicalDataPreprocessor",
    "CPRVideoProcessor",
    "MedicalDataValidator",
    "ComplianceValidator",
    "SyntheticDataGenerator",
    "CPRScenarioGenerator"
]
