"""
Configuration management for SMART-TRAIN platform.

This module provides centralized configuration management with support for
environment variables, validation, and medical compliance requirements.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from marshmallow import Schema, fields, validate, ValidationError
import structlog

from .exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "smart_train"
    username: str = "smart_train_user"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class AzureConfig:
    """Azure cloud configuration settings."""
    subscription_id: str = ""
    resource_group: str = ""
    workspace_name: str = ""
    location: str = "eastus"
    storage_account: str = ""
    container_name: str = ""
    key_vault_name: str = ""


@dataclass
class ModelConfig:
    """AI model configuration settings."""
    pose_estimation_model: str = "mediapipe"
    pose_confidence_threshold: float = 0.5
    action_recognition_model: str = "transformer_net"
    quality_assessment_model: str = "medical_quality_net"
    inference_batch_size: int = 1
    max_inference_time_ms: int = 100
    model_cache_size: int = 3


@dataclass
class MedicalComplianceConfig:
    """Medical compliance configuration settings."""
    iso_13485_enabled: bool = True
    iec_62304_enabled: bool = True
    hipaa_compliance: bool = True
    gdpr_compliance: bool = True
    audit_trail_enabled: bool = True
    data_anonymization_required: bool = True
    quality_management_system: bool = True
    risk_management_enabled: bool = True


@dataclass
class DataProcessingConfig:
    """Data processing configuration settings."""
    video_target_resolution: List[int] = field(default_factory=lambda: [1280, 720])
    video_target_fps: int = 30
    max_video_duration_seconds: int = 300
    min_video_duration_seconds: int = 10
    pose_keypoint_format: str = "coco_17"
    data_augmentation_enabled: bool = True
    parallel_processing_workers: int = 4
    quality_validation_threshold: float = 0.8


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "127.0.0.1"  # Use localhost by default for security
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests_per_minute: int = 100
    max_request_size_mb: int = 100
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    log_dir: Optional[str] = None
    enable_audit_trail: bool = True
    enable_console: bool = True
    enable_file_logging: bool = True
    max_log_file_size_mb: int = 100
    log_retention_days: int = 90


class SmartTrainConfig:
    """
    Main configuration class for SMART-TRAIN platform.

    This class provides centralized configuration management with validation,
    environment variable support, and medical compliance settings.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_data: Dict[str, Any] = {}

        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.azure = AzureConfig()
        self.model = ModelConfig()
        self.medical_compliance = MedicalComplianceConfig()
        self.data_processing = DataProcessingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()

        # Load configuration
        self._load_configuration()
        self._validate_configuration()

        logger.info("Configuration loaded successfully", config_path=str(self.config_path))

    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from file if provided
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self._config_data = yaml.safe_load(f) or {}
                logger.info("Configuration file loaded", path=str(self.config_path))
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load configuration file: {e}",
                    config_key="config_file",
                    context={"config_path": str(self.config_path)}
                )

        # Override with environment variables
        self._load_environment_variables()

        # Apply configuration to dataclasses
        self._apply_configuration()

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # Database
            "SMART_TRAIN_DB_HOST": ("database", "host"),
            "SMART_TRAIN_DB_PORT": ("database", "port"),
            "SMART_TRAIN_DB_NAME": ("database", "database"),
            "SMART_TRAIN_DB_USER": ("database", "username"),
            "SMART_TRAIN_DB_PASSWORD": ("database", "password"),

            # Azure
            "AZURE_SUBSCRIPTION_ID": ("azure", "subscription_id"),
            "AZURE_RESOURCE_GROUP": ("azure", "resource_group"),
            "AZURE_ML_WORKSPACE_NAME": ("azure", "workspace_name"),
            "AZURE_LOCATION": ("azure", "location"),
            "AZURE_STORAGE_ACCOUNT_NAME": ("azure", "storage_account"),
            "AZURE_STORAGE_CONTAINER_NAME": ("azure", "container_name"),

            # API
            "SMART_TRAIN_API_HOST": ("api", "host"),
            "SMART_TRAIN_API_PORT": ("api", "port"),
            "SMART_TRAIN_JWT_SECRET": ("api", "jwt_secret_key"),

            # Logging
            "SMART_TRAIN_LOG_LEVEL": ("logging", "level"),
            "SMART_TRAIN_LOG_DIR": ("logging", "log_dir"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_data:
                    self._config_data[section] = {}

                # Type conversion for numeric values
                if key in ["port", "pool_size", "max_overflow", "inference_batch_size"]:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(
                            "Invalid integer value for environment variable",
                            env_var=env_var,
                            value=value
                        )
                        continue
                elif key in ["pose_confidence_threshold", "quality_validation_threshold"]:
                    try:
                        value = float(value)
                    except ValueError:
                        logger.warning(
                            "Invalid float value for environment variable",
                            env_var=env_var,
                            value=value
                        )
                        continue
                elif key in ["debug", "cors_enabled", "hipaa_compliance", "gdpr_compliance"]:
                    value = value.lower() in ("true", "1", "yes", "on")

                self._config_data[section][key] = value

    def _apply_configuration(self) -> None:
        """Apply loaded configuration to dataclass instances."""
        # Database configuration
        if "database" in self._config_data:
            db_config = self._config_data["database"]
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)

        # Azure configuration
        if "azure" in self._config_data:
            azure_config = self._config_data["azure"]
            for key, value in azure_config.items():
                if hasattr(self.azure, key):
                    setattr(self.azure, key, value)

        # Model configuration
        if "model" in self._config_data:
            model_config = self._config_data["model"]
            for key, value in model_config.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)

        # Medical compliance configuration
        if "medical_compliance" in self._config_data:
            compliance_config = self._config_data["medical_compliance"]
            for key, value in compliance_config.items():
                if hasattr(self.medical_compliance, key):
                    setattr(self.medical_compliance, key, value)

        # Data processing configuration
        if "data_processing" in self._config_data:
            processing_config = self._config_data["data_processing"]
            for key, value in processing_config.items():
                if hasattr(self.data_processing, key):
                    setattr(self.data_processing, key, value)

        # API configuration
        if "api" in self._config_data:
            api_config = self._config_data["api"]
            for key, value in api_config.items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)

        # Logging configuration
        if "logging" in self._config_data:
            logging_config = self._config_data["logging"]
            for key, value in logging_config.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)

    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        errors = []

        # Validate required Azure settings for production
        if not self.azure.subscription_id and os.getenv("ENVIRONMENT") == "production":
            errors.append("Azure subscription_id is required for production")

        # Validate API settings
        if not self.api.jwt_secret_key and not os.getenv("SMART_TRAIN_JWT_SECRET"):
            errors.append("JWT secret key is required for API security")

        # Validate model settings
        if not (0.0 <= self.model.pose_confidence_threshold <= 1.0):
            errors.append("Pose confidence threshold must be between 0.0 and 1.0")

        # Validate data processing settings
        if self.data_processing.max_video_duration_seconds <= self.data_processing.min_video_duration_seconds:
            errors.append("Max video duration must be greater than min video duration")

        # Validate medical compliance requirements
        if self.medical_compliance.hipaa_compliance and not self.medical_compliance.audit_trail_enabled:
            errors.append("HIPAA compliance requires audit trail to be enabled")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                context={"validation_errors": errors}
            )

    def get_database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.database.username}:{self.database.password}@"
            f"{self.database.host}:{self.database.port}/{self.database.database}"
            f"?sslmode={self.database.ssl_mode}"
        )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return not self.is_production()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        config_dict = {}

        # Add non-sensitive configuration
        for section_name in ["model", "medical_compliance", "data_processing", "api", "logging"]:
            section = getattr(self, section_name)
            config_dict[section_name] = {
                key: value for key, value in section.__dict__.items()
                if not key.endswith("password") and not key.endswith("secret")
            }

        # Add database config (excluding password)
        config_dict["database"] = {
            key: value for key, value in self.database.__dict__.items()
            if key != "password"
        }

        # Add Azure config (excluding sensitive data)
        config_dict["azure"] = {
            key: value for key, value in self.azure.__dict__.items()
            if not any(sensitive in key.lower() for sensitive in ["key", "secret", "password"])
        }

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SmartTrainConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SmartTrainConfig instance
        """
        # Create a temporary config instance
        config = cls.__new__(cls)
        config.config_path = None
        config._config_data = config_dict.copy()
        
        # Initialize configuration sections
        config.database = DatabaseConfig()
        config.azure = AzureConfig()
        config.model = ModelConfig()
        config.medical_compliance = MedicalComplianceConfig()
        config.data_processing = DataProcessingConfig()
        config.api = APIConfig()
        config.logging = LoggingConfig()
        
        # Apply configuration
        config._apply_configuration()
        
        return config


# Global configuration instance
_config_instance: Optional[SmartTrainConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> SmartTrainConfig:
    """
    Get global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        SmartTrainConfig instance
    """
    global _config_instance

    if _config_instance is None:
        # Look for default config files
        if config_path is None:
            default_paths = [
                Path("config/smart_train.yaml"),
                Path("smart_train.yaml"),
                Path.home() / ".smart_train" / "config.yaml"
            ]

            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        _config_instance = SmartTrainConfig(config_path)

    return _config_instance


def reload_config(config_path: Optional[Union[str, Path]] = None) -> SmartTrainConfig:
    """
    Reload configuration (useful for testing or config changes).

    Args:
        config_path: Path to configuration file

    Returns:
        New SmartTrainConfig instance
    """
    global _config_instance
    _config_instance = SmartTrainConfig(config_path)
    return _config_instance
