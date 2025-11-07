"""
Model Training module for SMART-TRAIN platform.

This module provides comprehensive model training capabilities including:
- MLflow experiment tracking
- Medical-grade model validation
- Automated hyperparameter tuning
- Model versioning and deployment
- Compliance-aware training pipelines
"""

from .trainer import ModelTrainer, CPRModelTrainer
from .experiment_manager import MLflowExperimentManager
from .validation import MedicalModelValidator
from .hyperparameter_tuning import HyperparameterOptimizer

__all__ = [
    "ModelTrainer",
    "CPRModelTrainer",
    "MLflowExperimentManager",
    "MedicalModelValidator",
    "HyperparameterOptimizer"
]
