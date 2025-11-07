"""Basic tests for SMART-TRAIN platform."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_smart_train():
    """Test that the main package can be imported."""
    import smart_train
    assert smart_train.__version__ == "1.0.0"


def test_import_core_modules():
    """Test that core modules can be imported."""
    from smart_train.core.config import SmartTrainConfig
    from smart_train.core.logging import get_logger
    from smart_train.core.exceptions import SmartTrainException
    
    assert SmartTrainConfig is not None
    assert get_logger is not None
    assert SmartTrainException is not None


def test_import_models():
    """Test that model modules can be imported."""
    from smart_train.models.cpr_quality_model import CPRQualityAssessmentModel
    from smart_train.models.realtime_feedback import RealTimeFeedbackModel
    
    assert CPRQualityAssessmentModel is not None
    assert RealTimeFeedbackModel is not None


def test_basic_functionality():
    """Test basic functionality works."""
    from smart_train.core.logging import get_logger
    
    logger = get_logger("test")
    assert logger is not None
    
    # Test logging doesn't crash
    logger.info("Test log message")
    
    assert True  # If we get here, basic functionality works
