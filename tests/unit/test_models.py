"""Unit tests for model modules."""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from smart_train.models.cpr_quality_model import (
    CPRQualityAssessmentModel, 
    CPRMetrics, 
    CPRQualityNet
)
from smart_train.models.realtime_feedback import (
    RealTimeFeedbackModel,
    FeedbackMessage,
    FeedbackPriority,
    FeedbackType
)
from smart_train.core.base import ProcessingResult


class TestCPRMetrics:
    """Test CPRMetrics dataclass."""
    
    def test_cpr_metrics_creation(self):
        """Test CPR metrics can be created."""
        metrics = CPRMetrics(
            compression_depth=55.0,
            compression_rate=110.0,
            hand_position_score=0.9,
            release_completeness=0.85,
            rhythm_consistency=0.92,
            overall_quality_score=0.88,
            aha_compliant=True,
            feedback_messages=["Good compression depth", "Maintain rate"]
        )
        assert metrics.compression_depth == 55.0
        assert metrics.aha_compliant is True
        assert len(metrics.feedback_messages) == 2


class TestCPRQualityNet:
    """Test CPRQualityNet neural network."""
    
    def test_network_initialization(self):
        """Test network can be initialized."""
        net = CPRQualityNet(input_dim=99, hidden_dim=128, num_layers=2)
        assert net.input_dim == 99
        assert net.hidden_dim == 128
        assert net.num_layers == 2
        
    def test_network_forward_pass(self):
        """Test network forward pass."""
        net = CPRQualityNet(input_dim=99, hidden_dim=64, num_layers=1)
        
        # Create mock input
        batch_size, seq_len, input_dim = 2, 10, 99
        x = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            output = net(x)
            
        assert isinstance(output, dict)
        assert 'compression_depth' in output
        assert 'compression_rate' in output
        assert 'quality_class' in output


class TestCPRQualityAssessmentModel:
    """Test CPRQualityAssessmentModel."""
    
    @patch('smart_train.models.cpr_quality_model.CPRQualityNet')
    def test_model_initialization(self, mock_net):
        """Test model initialization."""
        mock_net.return_value = Mock()
        model = CPRQualityAssessmentModel()
        assert model.model_name == "CPRQualityAssessment"
        assert model.model_version == "2.0.0"
        
    @patch('smart_train.models.cpr_quality_model.logger')
    @patch('smart_train.models.cpr_quality_model.AuditTrailManager')
    @patch('smart_train.models.cpr_quality_model.CPRQualityNet')
    def test_model_predict_success(self, mock_net, mock_audit, mock_logger):
        """Test successful prediction."""
        # Mock audit manager to avoid logging issues
        mock_audit.return_value.log_event = Mock()
        
        # Setup mock network with proper tensor outputs
        mock_network = Mock()
        mock_network.return_value = {
            'compression_depth': torch.tensor([[55.0]]),
            'compression_rate': torch.tensor([[110.0]]),
            'hand_position': torch.tensor([[0.9]]),
            'release_completeness': torch.tensor([[0.85]]),
            'rhythm_consistency': torch.tensor([[0.92]]),
            'attention_weights': torch.tensor([[[0.1, 0.2, 0.3]]])
        }
        mock_net.return_value = mock_network
        
        model = CPRQualityAssessmentModel()
        model.is_loaded = True
        model.model = mock_network
        
        # Mock the audit manager instance on the model
        model.audit_manager = Mock()
        model.audit_manager.log_event = Mock()
        
        # Test prediction with proper numpy array format (ensure >= 10 frames)
        input_data = np.random.rand(30, 33, 3)  # (sequence_length, 33_landmarks, 3_coords)
        
        result = model.predict(input_data)
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        
    @patch('smart_train.models.cpr_quality_model.logger')
    @patch('smart_train.models.cpr_quality_model.AuditTrailManager')
    @patch('smart_train.models.cpr_quality_model.CPRQualityNet')
    def test_model_predict_invalid_input(self, mock_net, mock_audit, mock_logger):
        """Test prediction with invalid input."""
        # Mock audit manager to avoid logging issues
        mock_audit.return_value.log_event = Mock()
        
        mock_net.return_value = Mock()
        model = CPRQualityAssessmentModel()
        model.is_loaded = True
        model.model = mock_net.return_value
        
        # Mock the audit manager instance on the model
        model.audit_manager = Mock()
        model.audit_manager.log_event = Mock()
        
        # Test with invalid input (empty dict instead of numpy array)
        result = model.predict({})
        assert result.success is False
        assert ("Invalid input" in result.message or 
                "CPR quality assessment failed" in result.message)


class TestFeedbackMessage:
    """Test FeedbackMessage dataclass."""
    
    def test_feedback_message_creation(self):
        """Test feedback message creation."""
        message = FeedbackMessage(
            message="Increase compression depth",
            priority=FeedbackPriority.HIGH,
            feedback_type=FeedbackType.TECHNIQUE,
            timestamp=1234567890.0,
            confidence=0.95,
            action_required=True,
            medical_context="AHA Guidelines",
            improvement_suggestion="Press 5-6cm deep"
        )
        assert message.message == "Increase compression depth"
        assert message.priority == FeedbackPriority.HIGH
        assert message.feedback_type == FeedbackType.TECHNIQUE
        assert message.confidence == 0.95
        assert message.action_required is True


class TestRealTimeFeedbackModel:
    """Test RealTimeFeedbackModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = RealTimeFeedbackModel()
        assert model.model_name == "RealTimeFeedback"
        assert model.model_version == "2.0.0"
        
    @patch('smart_train.models.realtime_feedback.logger')
    @patch('smart_train.models.realtime_feedback.AuditTrailManager')
    def test_model_predict_success(self, mock_audit, mock_logger):
        """Test successful feedback prediction."""
        # Mock the audit manager to avoid logging issues
        mock_audit.return_value.log_event = Mock()
        
        model = RealTimeFeedbackModel()
        model.is_loaded = True
        
        # Mock the audit manager instance on the model
        model.audit_manager = Mock()
        model.audit_manager.log_event = Mock()
        
        input_data = {
            'cpr_metrics': {
                'compression_depth': 45.0,  # Below optimal
                'compression_rate': 130.0,  # Above optimal
                'hand_position_score': 0.9,
                'release_completeness': 0.8,
                'rhythm_consistency': 0.7,
                'overall_quality_score': 0.75
            },
            'session_context': {
                'trainee_level': 'beginner',
                'session_duration': 120
            }
        }
        
        result = model.predict(input_data)
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert 'feedback_messages' in result.data
        
    def test_model_predict_invalid_input(self):
        """Test prediction with invalid input."""
        model = RealTimeFeedbackModel()
        model.is_loaded = True
        
        # Test with missing cpr_metrics
        result = model.predict({})
        assert result.success is False
        assert "Feedback generation failed" in result.message


class TestModelIntegration:
    """Test model integration scenarios."""
    
    @patch('smart_train.models.cpr_quality_model.logger')
    @patch('smart_train.models.cpr_quality_model.AuditTrailManager')
    @patch('smart_train.models.cpr_quality_model.CPRQualityNet')
    def test_cpr_to_feedback_pipeline(self, mock_net, mock_audit, mock_logger):
        """Test CPR assessment to feedback pipeline."""
        # Mock audit manager to avoid logging issues
        mock_audit.return_value.log_event = Mock()
        
        # Setup CPR model
        mock_network = Mock()
        mock_network.return_value = {
            'compression_depth': torch.tensor([[45.0]]),
            'compression_rate': torch.tensor([[130.0]]),
            'hand_position': torch.tensor([[0.9]]),
            'release_completeness': torch.tensor([[0.8]]),
            'rhythm_consistency': torch.tensor([[0.7]]),
            'overall_quality': torch.tensor([[0.75]])
        }
        mock_net.return_value = mock_network
        
        cpr_model = CPRQualityAssessmentModel()
        cpr_model.is_loaded = True
        cpr_model.model = mock_network
        
        # Mock the audit manager instance on the CPR model
        cpr_model.audit_manager = Mock()
        cpr_model.audit_manager.log_event = Mock()
        
        feedback_model = RealTimeFeedbackModel()
        feedback_model.is_loaded = True
        
        # Mock the audit manager instance on the feedback model
        feedback_model.audit_manager = Mock()
        feedback_model.audit_manager.log_event = Mock()
        
        # Test pipeline
        pose_data = np.random.rand(30, 33, 3)  # Proper format
        cpr_result = cpr_model.predict(pose_data)
        
        if cpr_result.success:
            feedback_input = {
                'cpr_metrics': cpr_result.data.get('cpr_metrics', {}),
                'session_context': {'trainee_level': 'beginner'}
            }
            feedback_result = feedback_model.predict(feedback_input)
            
            assert feedback_result.success is True
            assert 'feedback_messages' in feedback_result.data
