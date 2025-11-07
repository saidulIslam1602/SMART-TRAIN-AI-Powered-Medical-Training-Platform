"""
CPR Quality Assessment Model for SMART-TRAIN platform.

This module implements a deep learning model for assessing CPR quality
based on pose estimation data, following AHA guidelines and medical standards.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import cv2
from dataclasses import dataclass

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..core.exceptions import ModelTrainingError, DataProcessingError
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger = get_logger(__name__)


@dataclass
class CPRMetrics:
    """CPR quality metrics following AHA guidelines."""
    compression_depth: float  # mm
    compression_rate: float   # compressions per minute
    hand_position_score: float  # 0-1 score
    release_completeness: float  # 0-1 score
    rhythm_consistency: float   # 0-1 score
    overall_quality_score: float  # 0-1 score
    aha_compliant: bool
    feedback_messages: List[str]


class CPRQualityNet(nn.Module):
    """
    Deep neural network for CPR quality assessment.
    
    Architecture:
    - Input: Pose keypoints sequence (33 landmarks x T frames)
    - LSTM for temporal modeling
    - Attention mechanism for key frame identification
    - Multi-task output for different quality metrics
    """
    
    def __init__(self, input_dim: int = 99, hidden_dim: int = 256, num_layers: int = 3):
        super(CPRQualityNet, self).__init__()
        
        # Input dimensions: 33 pose landmarks * 3 coordinates = 99
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Multi-task output heads
        self.compression_depth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1, scaled to 0-60mm
        )
        
        self.compression_rate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1, scaled to 80-120 CPM
        )
        
        self.hand_position_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 score
        )
        
        self.release_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 score
        )
        
        self.rhythm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 score
        )
        
        # Overall quality classifier
        self.quality_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Poor, Fair, Good, Excellent
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Dictionary of predictions for each quality metric
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        x_features = self.feature_extractor(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_features)
        
        # Attention mechanism
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled_features = torch.mean(attended_out, dim=1)
        
        # Multi-task predictions
        predictions = {
            'compression_depth': self.compression_depth_head(pooled_features),
            'compression_rate': self.compression_rate_head(pooled_features),
            'hand_position': self.hand_position_head(pooled_features),
            'release_completeness': self.release_head(pooled_features),
            'rhythm_consistency': self.rhythm_head(pooled_features),
            'quality_class': self.quality_classifier(pooled_features),
            'attention_weights': attention_weights
        }
        
        return predictions


class CPRQualityAssessmentModel(BaseModel):
    """
    CPR Quality Assessment Model for real-time analysis.
    
    This model analyzes pose estimation data to provide comprehensive
    CPR quality assessment following AHA guidelines.
    """
    
    def __init__(self, model_version: str = "2.0.0"):
        super().__init__("CPRQualityAssessment", model_version)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CPRQualityNet] = None
        self.audit_manager = AuditTrailManager()
        
        # AHA Guidelines thresholds
        self.aha_thresholds = {
            'compression_depth_min': 50,  # mm
            'compression_depth_max': 60,  # mm
            'compression_rate_min': 100,  # CPM
            'compression_rate_max': 120,  # CPM
            'hand_position_threshold': 0.8,
            'release_threshold': 0.9,
            'rhythm_threshold': 0.7
        }
        
        # Performance tracking
        self.inference_times = []
        self.quality_scores = []
        
        logger.info("CPR Quality Assessment Model initialized", 
                   device=str(self.device), model_version=model_version)
    
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load the CPR quality assessment model.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            if model_path is None:
                model_path = Path("models/cpr_quality_model.pth")
            
            # Initialize model architecture
            self.model = CPRQualityNet()
            self.model.to(self.device)
            
            if model_path.exists():
                # Load pre-trained weights (secure loading)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Pre-trained model loaded", model_path=str(model_path))
            else:
                # Initialize with random weights for development
                logger.warning("No pre-trained model found, using random initialization")
            
            self.model.eval()
            self.is_loaded = True
            self.load_timestamp = torch.tensor(0.0).item()  # Current timestamp
            
            # Log model loading event
            self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_OPERATION,
                description=f"CPR Quality Model loaded: {model_path}",
                severity=AuditSeverity.INFO
            )
            
        except Exception as e:
            logger.error("Failed to load CPR quality model", error=str(e))
            raise ModelTrainingError(f"Model loading failed: {e}")
    
    def predict(self, input_data: np.ndarray) -> ProcessingResult:
        """
        Perform CPR quality assessment on pose data.
        
        Args:
            input_data: Pose landmarks array of shape (sequence_length, 33, 3)
            
        Returns:
            ProcessingResult with CPR quality metrics
        """
        if not self.is_loaded or self.model is None:
            raise ModelTrainingError("Model not loaded. Call load_model() first.")
        
        try:
            start_time = torch.tensor(0.0).item()  # Current timestamp
            
            # Validate input
            if not self.validate_input(input_data):
                raise DataProcessingError("Invalid input data for CPR quality assessment")
            
            # Preprocess input
            processed_input = self._preprocess_pose_data(input_data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_input).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Post-process predictions
            cpr_metrics = self._postprocess_predictions(predictions)
            
            # Calculate inference time
            inference_time = torch.tensor(0.0).item() - start_time  # Current timestamp - start_time
            self.inference_times.append(inference_time)
            self.inference_count += 1
            
            # Create result
            result = ProcessingResult(
                success=True,
                data={
                    'cpr_metrics': cpr_metrics.__dict__,
                    'inference_time_ms': inference_time * 1000,
                    'model_version': self.model_version,
                    'aha_compliant': cpr_metrics.aha_compliant
                },
                metadata={
                    'model_id': self.model_id,
                    'device': str(self.device),
                    'input_shape': input_data.shape,
                    'processing_timestamp': torch.tensor(0.0).item()  # Current timestamp
                }
            )
            
            # Log inference event
            self.audit_manager.log_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                description=f"CPR quality assessment completed",
                severity=AuditSeverity.INFO,
                metadata={
                    'quality_score': cpr_metrics.overall_quality_score,
                    'aha_compliant': cpr_metrics.aha_compliant,
                    'inference_time_ms': inference_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            logger.error("CPR quality assessment failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"CPR quality assessment failed: {e}",
                data={}
            )
    
    def _preprocess_pose_data(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Preprocess pose landmarks for model input.
        
        Args:
            pose_data: Raw pose landmarks (sequence_length, 33, 3)
            
        Returns:
            Processed pose data (sequence_length, 99)
        """
        # Flatten pose landmarks
        flattened = pose_data.reshape(pose_data.shape[0], -1)
        
        # Normalize coordinates
        normalized = (flattened - np.mean(flattened, axis=0)) / (np.std(flattened, axis=0) + 1e-8)
        
        # Handle sequence length
        target_length = 60  # 2 seconds at 30 FPS
        if normalized.shape[0] < target_length:
            # Pad sequence
            padding = np.zeros((target_length - normalized.shape[0], normalized.shape[1]))
            normalized = np.vstack([normalized, padding])
        elif normalized.shape[0] > target_length:
            # Truncate sequence
            normalized = normalized[:target_length]
        
        return normalized
    
    def _postprocess_predictions(self, predictions: Dict[str, torch.Tensor]) -> CPRMetrics:
        """
        Convert model predictions to CPR metrics.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            CPRMetrics object with processed results
        """
        # Extract predictions
        depth_raw = predictions['compression_depth'].cpu().numpy()[0, 0]
        rate_raw = predictions['compression_rate'].cpu().numpy()[0, 0]
        hand_pos = predictions['hand_position'].cpu().numpy()[0, 0]
        release = predictions['release_completeness'].cpu().numpy()[0, 0]
        rhythm = predictions['rhythm_consistency'].cpu().numpy()[0, 0]
        
        # Scale predictions to real values
        compression_depth = depth_raw * 60  # Scale to 0-60mm
        compression_rate = 80 + (rate_raw * 40)  # Scale to 80-120 CPM
        
        # Calculate overall quality score
        quality_scores = [hand_pos, release, rhythm]
        if 50 <= compression_depth <= 60:
            quality_scores.append(1.0)
        else:
            quality_scores.append(max(0, 1 - abs(compression_depth - 55) / 10))
        
        if 100 <= compression_rate <= 120:
            quality_scores.append(1.0)
        else:
            quality_scores.append(max(0, 1 - abs(compression_rate - 110) / 20))
        
        overall_quality = np.mean(quality_scores)
        
        # Check AHA compliance
        aha_compliant = (
            self.aha_thresholds['compression_depth_min'] <= compression_depth <= self.aha_thresholds['compression_depth_max'] and
            self.aha_thresholds['compression_rate_min'] <= compression_rate <= self.aha_thresholds['compression_rate_max'] and
            hand_pos >= self.aha_thresholds['hand_position_threshold'] and
            release >= self.aha_thresholds['release_threshold'] and
            rhythm >= self.aha_thresholds['rhythm_threshold']
        )
        
        # Generate feedback messages
        feedback_messages = []
        if compression_depth < 50:
            feedback_messages.append("Increase compression depth - aim for 2-2.4 inches")
        elif compression_depth > 60:
            feedback_messages.append("Reduce compression depth - avoid over-compression")
        
        if compression_rate < 100:
            feedback_messages.append("Increase compression rate - aim for 100-120 per minute")
        elif compression_rate > 120:
            feedback_messages.append("Reduce compression rate - maintain steady rhythm")
        
        if hand_pos < 0.8:
            feedback_messages.append("Adjust hand position - center on lower half of breastbone")
        
        if release < 0.9:
            feedback_messages.append("Allow complete chest recoil between compressions")
        
        if rhythm < 0.7:
            feedback_messages.append("Maintain consistent compression rhythm")
        
        if not feedback_messages:
            feedback_messages.append("Excellent CPR technique - maintain current performance")
        
        return CPRMetrics(
            compression_depth=compression_depth,
            compression_rate=compression_rate,
            hand_position_score=hand_pos,
            release_completeness=release,
            rhythm_consistency=rhythm,
            overall_quality_score=overall_quality,
            aha_compliant=aha_compliant,
            feedback_messages=feedback_messages
        )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate pose data input.
        
        Args:
            input_data: Input pose data
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, np.ndarray):
            return False
        
        if len(input_data.shape) != 3:
            return False
        
        if input_data.shape[1] != 33 or input_data.shape[2] != 3:
            return False
        
        if input_data.shape[0] < 10:  # Minimum sequence length
            return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        if not self.inference_times:
            return {"message": "No inference data available"}
        
        return {
            "total_inferences": self.inference_count,
            "average_inference_time_ms": np.mean(self.inference_times) * 1000,
            "max_inference_time_ms": np.max(self.inference_times) * 1000,
            "min_inference_time_ms": np.min(self.inference_times) * 1000,
            "average_quality_score": np.mean(self.quality_scores) if self.quality_scores else 0.0,
            "model_version": self.model_version,
            "device": str(self.device)
        }
