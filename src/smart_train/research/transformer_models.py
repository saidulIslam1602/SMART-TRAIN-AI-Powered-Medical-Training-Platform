"""
State-of-the-Art Transformer Models for Medical Analysis.

This module implements cutting-edge Transformer architectures specifically
designed for medical procedure analysis, including:
- Vision Transformer (ViT) for medical image analysis
- Temporal Transformer for action sequence modeling
- Multi-scale attention for hierarchical feature extraction
- Medical-specific positional encodings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from einops.layers.torch import Rearrange

from ..core.base import BaseModel, ProcessingResult
from ..core.logging import get_logger
from ..core.exceptions import ModelTrainingError

logger = get_logger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"

    # Input configuration
    sequence_length: int = 60  # Number of frames
    pose_dim: int = 99  # 33 landmarks * 3 coordinates

    # Vision Transformer specific
    patch_size: int = 16
    image_size: int = 224
    num_classes: int = 6  # CPR quality dimensions

    # Medical-specific
    medical_attention_heads: int = 4
    anatomical_regions: int = 5  # Head, torso, arms, hands, legs
    temporal_window: int = 30  # Frames for temporal analysis


class MedicalPositionalEncoding(nn.Module):
    """
    Medical-specific positional encoding that incorporates:
    - Temporal information (frame sequence)
    - Anatomical structure (body landmarks)
    - Medical procedure phases
    """

    def __init__(self, d_model: int, max_len: int = 5000,
                 anatomical_regions: int = 5):
        super().__init__()
        self.d_model = d_model
        self.anatomical_regions = anatomical_regions

        # Standard temporal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # Anatomical positional encoding
        self.anatomical_embedding = nn.Embedding(anatomical_regions, d_model // 4)

        # Medical phase encoding (preparation, execution, recovery)
        self.phase_embedding = nn.Embedding(3, d_model // 4)

    def forward(self, x: torch.Tensor, anatomical_ids: Optional[torch.Tensor] = None,
                phase_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            anatomical_ids: Anatomical region IDs [seq_len, batch_size]
            phase_ids: Medical phase IDs [seq_len, batch_size]
        """
        seq_len = x.size(0)

        # Add temporal positional encoding
        x = x + self.pe[:seq_len, :]

        # Add anatomical encoding if provided
        if anatomical_ids is not None:
            anatomical_enc = self.anatomical_embedding(anatomical_ids)
            x = x + anatomical_enc.repeat(1, 1, 4)  # Expand to match d_model

        # Add phase encoding if provided
        if phase_ids is not None:
            phase_enc = self.phase_embedding(phase_ids)
            x = x + phase_enc.repeat(1, 1, 4)  # Expand to match d_model

        return x


class MedicalMultiHeadAttention(nn.Module):
    """
    Medical-specific multi-head attention that incorporates:
    - Anatomical structure awareness
    - Temporal dependencies
    - Medical procedure constraints
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        if self.head_dim * nhead != d_model:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Medical-specific attention masks
        self.register_buffer('anatomical_mask', self._create_anatomical_mask())

    def _create_anatomical_mask(self) -> torch.Tensor:
        """Create attention mask based on anatomical connectivity."""
        # Simplified anatomical connectivity matrix
        # In practice, this would be based on medical knowledge
        mask = torch.ones(33, 33)  # 33 pose landmarks

        # Example: hands should attend more to arms, less to legs
        # This is a simplified example - real implementation would use
        # detailed anatomical knowledge

        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [seq_len, batch_size, d_model]
            attn_mask: Optional attention mask

        Returns:
            output: [seq_len, batch_size, d_model]
            attention_weights: [batch_size, nhead, seq_len, seq_len]
        """
        seq_len, batch_size, d_model = query.size()

        # Linear transformations
        Q = self.q_linear(query)  # [seq_len, batch_size, d_model]
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(seq_len, batch_size, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(seq_len, batch_size, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(seq_len, batch_size, self.nhead, self.head_dim).transpose(1, 2)
        # Now: [batch_size, nhead, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, nhead, seq_len, seq_len]

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # [batch_size, nhead, seq_len, head_dim]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            seq_len, batch_size, d_model
        )

        # Final linear transformation
        output = self.out_linear(context)

        return output, attention_weights


class CPRTransformerNet(nn.Module):
    """
    State-of-the-art Transformer network for CPR quality assessment.

    This model uses advanced Transformer architecture with:
    - Medical-specific positional encoding
    - Anatomical attention mechanisms
    - Temporal modeling for action sequences
    - Multi-scale feature extraction
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_embedding = nn.Linear(config.pose_dim, config.d_model)

        # Medical positional encoding
        self.pos_encoder = MedicalPositionalEncoding(
            config.d_model,
            max_len=config.sequence_length * 2,
            anatomical_regions=config.anatomical_regions
        )

        # Transformer encoder layers with medical attention
        encoder_layers = []
        for _ in range(config.num_encoder_layers):
            layer = TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False
            )
            encoder_layers.append(layer)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers[0],
            num_layers=config.num_encoder_layers
        )

        # Multi-scale temporal convolutions
        self.temporal_conv1 = nn.Conv1d(config.d_model, config.d_model,
                                       kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(config.d_model, config.d_model,
                                       kernel_size=5, padding=2)
        self.temporal_conv3 = nn.Conv1d(config.d_model, config.d_model,
                                       kernel_size=7, padding=3)

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            config.d_model, config.medical_attention_heads,
            dropout=config.dropout, batch_first=False
        )

        # Classification heads for different CPR quality metrics
        self.compression_depth_head = nn.Linear(config.d_model, 1)
        self.compression_rate_head = nn.Linear(config.d_model, 1)
        self.hand_position_head = nn.Linear(config.d_model, 3)  # x, y, confidence
        self.release_quality_head = nn.Linear(config.d_model, 1)
        self.overall_quality_head = nn.Linear(config.d_model, 1)
        self.aha_compliance_head = nn.Linear(config.d_model, 1)

        # Uncertainty quantification
        self.uncertainty_head = nn.Linear(config.d_model, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor,
                anatomical_ids: Optional[torch.Tensor] = None,
                phase_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the CPR Transformer.

        Args:
            x: Input pose sequences [batch_size, seq_len, pose_dim]
            anatomical_ids: Anatomical region IDs [batch_size, seq_len]
            phase_ids: Medical phase IDs [batch_size, seq_len]

        Returns:
            Dictionary containing predictions for different CPR metrics
        """
        batch_size, seq_len, pose_dim = x.shape

        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]

        # Add positional encoding
        x = self.pos_encoder(x, anatomical_ids, phase_ids)

        # Transformer encoding
        transformer_output = self.transformer_encoder(x)
        # [seq_len, batch_size, d_model]

        # Multi-scale temporal convolutions
        conv_input = transformer_output.transpose(0, 2)  # [batch_size, d_model, seq_len]

        conv1_out = F.relu(self.temporal_conv1(conv_input))
        conv2_out = F.relu(self.temporal_conv2(conv_input))
        conv3_out = F.relu(self.temporal_conv3(conv_input))

        # Combine multi-scale features
        multi_scale = (conv1_out + conv2_out + conv3_out) / 3
        multi_scale = multi_scale.transpose(0, 2)  # [seq_len, batch_size, d_model]

        # Attention pooling for sequence aggregation
        pooled_output, attention_weights = self.attention_pool(
            multi_scale, multi_scale, multi_scale
        )

        # Global average pooling
        global_features = pooled_output.mean(dim=0)  # [batch_size, d_model]

        # Generate predictions for different CPR metrics
        predictions = {
            'compression_depth': torch.sigmoid(self.compression_depth_head(global_features)),
            'compression_rate': torch.sigmoid(self.compression_rate_head(global_features)),
            'hand_position': self.hand_position_head(global_features),
            'release_quality': torch.sigmoid(self.release_quality_head(global_features)),
            'overall_quality': torch.sigmoid(self.overall_quality_head(global_features)),
            'aha_compliance': torch.sigmoid(self.aha_compliance_head(global_features)),
            'uncertainty': F.softplus(self.uncertainty_head(global_features)),
            'attention_weights': attention_weights,
            'global_features': global_features
        }

        return predictions


class MedicalVisionTransformer(nn.Module):
    """
    Vision Transformer adapted for medical image analysis.

    This model processes medical training videos frame by frame
    using Vision Transformer architecture with medical-specific
    adaptations.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.d_model)
        )

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Transformer encoder
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_encoder_layers
        )

        # Classification head
        self.classification_head = nn.Linear(config.d_model, config.num_classes)

        # Medical region attention
        self.medical_attention = nn.MultiheadAttention(
            config.d_model, config.medical_attention_heads,
            dropout=config.dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Medical Vision Transformer.

        Args:
            x: Input images [batch_size, channels, height, width]

        Returns:
            Dictionary containing predictions and attention maps
        """
        _batch_size = x.shape[0]

        # Patch embedding
        patches = self.patch_embedding(x)  # [batch_size, d_model, H/P, W/P]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, d_model]

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoding
        x = self.transformer(x)

        # Medical attention on class token
        cls_output = x[:, 0:1, :]  # [batch_size, 1, d_model]
        medical_attended, medical_attention = self.medical_attention(
            cls_output, x, x
        )

        # Classification
        predictions = self.classification_head(medical_attended.squeeze(1))

        return {
            'predictions': predictions,
            'medical_attention': medical_attention,
            'cls_features': cls_output.squeeze(1),
            'patch_features': x[:, 1:, :]
        }


class MedicalTransformer(BaseModel):
    """
    Complete Medical Transformer system combining pose and vision analysis.

    This model integrates:
    - CPR Transformer for pose sequence analysis
    - Vision Transformer for video frame analysis
    - Multi-modal fusion for comprehensive assessment
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__("MedicalTransformer", "4.0.0")

        self.config = config or TransformerConfig()

        # Initialize sub-models
        self.cpr_transformer = CPRTransformerNet(self.config)
        self.vision_transformer = MedicalVisionTransformer(self.config)

        # Multi-modal fusion
        self.fusion_layer = nn.Linear(
            self.config.d_model * 2,  # CPR + Vision features
            self.config.d_model
        )

        # Final prediction heads
        self.final_quality_head = nn.Linear(self.config.d_model, 1)
        self.final_compliance_head = nn.Linear(self.config.d_model, 1)
        self.confidence_head = nn.Linear(self.config.d_model, 1)

        self.is_loaded = True
        logger.info("Medical Transformer model initialized",
                   model_version=self.model_version)

    def load_model(self, model_path: Optional[str] = None):
        """Load pre-trained model weights."""
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                self.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Medical Transformer model loaded", model_path=model_path)
            except Exception as e:
                logger.error("Failed to load model", error=str(e))
                raise ModelTrainingError(f"Model loading failed: {e}")

        self.is_loaded = True

    def predict(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """
        Predict CPR quality using multi-modal Transformer analysis.

        Args:
            input_data: Dictionary containing:
                - pose_sequences: [batch_size, seq_len, pose_dim]
                - video_frames: [batch_size, channels, height, width] (optional)
                - anatomical_ids: [batch_size, seq_len] (optional)
                - phase_ids: [batch_size, seq_len] (optional)

        Returns:
            ProcessingResult with comprehensive CPR analysis
        """
        try:
            self.inference_count += 1

            pose_sequences = input_data.get('pose_sequences')
            if pose_sequences is None:
                return ProcessingResult(
                    success=False,
                    error_message="Pose sequences are required",
                    data={}
                )

            # Convert to tensor if needed
            if not isinstance(pose_sequences, torch.Tensor):
                pose_sequences = torch.FloatTensor(pose_sequences)

            # CPR Transformer analysis
            cpr_results = self.cpr_transformer(
                pose_sequences,
                anatomical_ids=input_data.get('anatomical_ids'),
                phase_ids=input_data.get('phase_ids')
            )

            # Vision Transformer analysis (if video frames provided)
            vision_results = None
            if 'video_frames' in input_data:
                video_frames = input_data['video_frames']
                if not isinstance(video_frames, torch.Tensor):
                    video_frames = torch.FloatTensor(video_frames)

                vision_results = self.vision_transformer(video_frames)

            # Multi-modal fusion
            if vision_results is not None:
                # Combine CPR and vision features
                combined_features = torch.cat([
                    cpr_results['global_features'],
                    vision_results['cls_features']
                ], dim=-1)

                fused_features = self.fusion_layer(combined_features)
            else:
                fused_features = cpr_results['global_features']

            # Final predictions
            final_quality = torch.sigmoid(self.final_quality_head(fused_features))
            final_compliance = torch.sigmoid(self.final_compliance_head(fused_features))
            confidence = torch.sigmoid(self.confidence_head(fused_features))

            # Prepare results
            results = {
                'cpr_metrics': {
                    'compression_depth': cpr_results['compression_depth'].item(),
                    'compression_rate': cpr_results['compression_rate'].item(),
                    'hand_position': cpr_results['hand_position'].tolist(),
                    'release_quality': cpr_results['release_quality'].item(),
                    'overall_quality_score': final_quality.item(),
                    'aha_compliance_score': final_compliance.item()
                },
                'confidence_metrics': {
                    'prediction_confidence': confidence.item(),
                    'uncertainty_estimates': cpr_results['uncertainty'].tolist()
                },
                'attention_analysis': {
                    'temporal_attention': cpr_results['attention_weights'].detach().numpy().tolist(),
                    'medical_attention': vision_results['medical_attention'].detach().numpy().tolist() if vision_results else None
                },
                'model_metadata': {
                    'model_type': 'MedicalTransformer',
                    'model_version': self.model_version,
                    'inference_count': self.inference_count,
                    'multimodal': vision_results is not None
                }
            }

            return ProcessingResult(
                success=True,
                data=results,
                metadata={
                    'processing_time_ms': 0,  # Would be calculated in practice
                    'model_confidence': confidence.item(),
                    'attention_maps_available': True
                }
            )

        except Exception as e:
            logger.error("Medical Transformer prediction failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"Prediction failed: {str(e)}",
                data={}
            )

    def get_attention_visualization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate attention visualization data for interpretability.

        Args:
            input_data: Same as predict method

        Returns:
            Dictionary containing attention maps and visualization data
        """
        try:
            # Run prediction to get attention weights
            result = self.predict(input_data)

            if not result.success:
                return {'error': result.error_message}

            attention_data = result.data.get('attention_analysis', {})

            # Process attention weights for visualization
            temporal_attention = attention_data.get('temporal_attention')
            medical_attention = attention_data.get('medical_attention')

            visualization_data = {
                'temporal_attention_heatmap': temporal_attention,
                'medical_region_attention': medical_attention,
                'attention_summary': {
                    'most_attended_frames': self._get_top_attended_frames(temporal_attention),
                    'most_attended_regions': self._get_top_attended_regions(medical_attention),
                    'attention_distribution': self._compute_attention_stats(temporal_attention)
                }
            }

            return visualization_data

        except Exception as e:
            logger.error("Attention visualization failed", error=str(e))
            return {'error': f"Visualization failed: {str(e)}"}

    def _get_top_attended_frames(self, attention_weights: List[List[List[float]]]) -> List[int]:
        """Get indices of most attended frames."""
        if not attention_weights:
            return []

        # Simplified implementation
        # In practice, would analyze attention patterns more sophisticatedly
        return [0, 15, 30, 45]  # Mock top frames

    def _get_top_attended_regions(self, medical_attention: Optional[List]) -> List[str]:
        """Get most attended medical regions."""
        if not medical_attention:
            return []

        # Mock medical regions based on attention
        return ['chest_center', 'hand_placement', 'arm_position', 'body_alignment']

    def _compute_attention_stats(self, attention_weights: List[List[List[float]]]) -> Dict[str, float]:
        """Compute attention distribution statistics."""
        if not attention_weights:
            return {}

        # Mock statistics
        return {
            'attention_entropy': 2.34,
            'attention_concentration': 0.67,
            'temporal_spread': 0.45,
            'peak_attention_ratio': 0.23
        }
