"""Unit tests for core modules."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from smart_train.core.config import SmartTrainConfig, get_config
from smart_train.core.logging import get_logger, setup_logging
from smart_train.core.exceptions import (
    SmartTrainException, 
    ModelTrainingError, 
    DataProcessingError,
    MedicalComplianceError
)
from smart_train.core.base import BaseProcessor, ProcessingResult


class TestSmartTrainConfig:
    """Test SmartTrainConfig class."""
    
    def test_config_initialization(self):
        """Test config can be initialized."""
        config = SmartTrainConfig()
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'api')
        
    def test_config_from_dict(self):
        """Test config can be created from dictionary."""
        config_dict = {
            'model': {'pose_confidence_threshold': 0.8},
            'api': {'host': '127.0.0.1', 'port': 8080}
        }
        config = SmartTrainConfig.from_dict(config_dict)
        assert config.model.pose_confidence_threshold == 0.8
        assert config.api.host == '127.0.0.1'
        assert config.api.port == 8080
        
    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestLogging:
    """Test logging functionality."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
        
    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_level='DEBUG', log_dir=temp_dir)
            logger = get_logger("test_setup")
            logger.info("Test message")
            # Should not raise any exceptions


class TestExceptions:
    """Test custom exceptions."""
    
    def test_smart_train_exception(self):
        """Test SmartTrainException."""
        with pytest.raises(SmartTrainException):
            raise SmartTrainException("Test error")
            
    def test_model_training_error(self):
        """Test ModelTrainingError."""
        with pytest.raises(ModelTrainingError):
            raise ModelTrainingError("Training failed")
            
    def test_data_processing_error(self):
        """Test DataProcessingError."""
        with pytest.raises(DataProcessingError):
            raise DataProcessingError("Processing failed")
            
    def test_medical_compliance_error(self):
        """Test MedicalComplianceError."""
        error = MedicalComplianceError(
            message="Compliance violation",
            compliance_standard="ISO 13485",
            violation_type="audit_failure",
            severity="HIGH"
        )
        assert error.compliance_standard == "ISO 13485"
        assert error.violation_type == "audit_failure"
        assert error.severity == "HIGH"


class TestBaseProcessor:
    """Test BaseProcessor functionality."""
    
    def test_processing_result_creation(self):
        """Test ProcessingResult creation."""
        result = ProcessingResult(
            success=True,
            message="Processing completed",
            data={'key': 'value'},
            processing_time=1.5
        )
        assert result.success is True
        assert result.message == "Processing completed"
        assert result.data == {'key': 'value'}
        assert result.processing_time == 1.5
        
    def test_processing_result_to_dict(self):
        """Test ProcessingResult to_dict method."""
        result = ProcessingResult(
            success=True,
            message="Test",
            data={'test': True}
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['message'] == "Test"
        assert result_dict['data'] == {'test': True}


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def process(self, input_data):
        """Mock process method."""
        return ProcessingResult(
            success=True,
            message="Mock processing completed",
            data={'processed': True}
        )


class TestBaseProcessorImplementation:
    """Test BaseProcessor implementation."""
    
    def test_mock_processor(self):
        """Test mock processor implementation."""
        processor = MockProcessor()
        result = processor.process({'test': 'data'})
        assert result.success is True
        assert result.data['processed'] is True
