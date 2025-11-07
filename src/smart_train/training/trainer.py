"""
Model Training module for SMART-TRAIN platform.

This module provides enterprise-grade model training with MLflow integration,
medical compliance validation, and automated experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ..core.base import BaseProcessor, ProcessingResult
from ..core.logging import get_logger
from ..core.exceptions import ModelTrainingError
from ..models.cpr_quality_model import CPRQualityNet
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger=get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int=32
    learning_rate: float=0.001
    num_epochs: int=100
    validation_split: float=0.2
    early_stopping_patience: int=10
    save_best_model: bool=True
    use_mixed_precision: bool=True
    gradient_clip_value: float=1.0
    weight_decay: float=1e-4
    scheduler_step_size: int=30
    scheduler_gamma: float=0.1


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float
    medical_compliance_score: float


class MedicalDataset(Dataset):
    """
    Medical training dataset for CPR quality assessment.

    This dataset handles pose sequence data with quality labels
    following medical standards and compliance requirements.
    """

    def __init__(self, pose_sequences: np.ndarray, quality_labels: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize medical dataset.

        Args:
            pose_sequences: Pose landmark sequences (N, T, 33, 3)
            quality_labels: Quality assessment labels
            metadata: Additional metadata for compliance tracking
        """
        self.pose_sequences=pose_sequences
        self.quality_labels=quality_labels
        self.metadata=metadata or {}

        # Validate data integrity
        self._validate_medical_data()

        logger.info("Medical dataset initialized",
                   samples=len(pose_sequences),
                   sequence_length=pose_sequences.shape[1] if len(pose_sequences.shape) > 1 else 0)

    def _validate_medical_data(self):
        """Validate medical data integrity and compliance."""
        if len(self.pose_sequences) != len(self.quality_labels):
            raise ModelTrainingError("Mismatch between pose sequences and labels")

        if len(self.pose_sequences) == 0:
            raise ModelTrainingError("Empty dataset provided")

        # Check for data quality issues
        if np.any(np.isnan(self.pose_sequences)):
            logger.warning("NaN values detected in pose sequences")

        if np.any(np.isinf(self.pose_sequences)):
            logger.warning("Infinite values detected in pose sequences")

    def __len__(self) -> int:
        return len(self.pose_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item."""
        pose_seq=torch.FloatTensor(self.pose_sequences[idx])
        quality_label=torch.FloatTensor(self.quality_labels[idx])

        return pose_seq, quality_label


class ModelTrainer(BaseProcessor):
    """
    Base model trainer with MLflow integration and medical compliance.

    This trainer provides enterprise-grade model training capabilities
    with comprehensive experiment tracking and medical validation.
    """

    def __init__(self, model_name: str, config: TrainingConfig):
        super().__init__(f"ModelTrainer_{model_name}", "2.0.0")

        self.model_name=model_name
        self.config=config
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audit_manager=AuditTrailManager()

        # Training state
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss=float('inf')
        self.best_model_path: Optional[Path] = None

        # MLflow tracking
        self.experiment_name=f"smart_train_{model_name}"
        self.run_id: Optional[str] = None

        logger.info("Model trainer initialized",
                   model_name=model_name, device=str(self.device))

    def setup_mlflow_experiment(self, experiment_name: Optional[str] = None) -> str:
        """
        Setup MLflow experiment for tracking.

        Args:
            experiment_name: Custom experiment name

        Returns:
            Experiment ID
        """
        exp_name=experiment_name or self.experiment_name

        try:
            experiment=mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                experiment_id=mlflow.create_experiment(exp_name)
            else:
                experiment_id=experiment.experiment_id

            mlflow.set_experiment(exp_name)

            logger.info("MLflow experiment setup",
                       experiment_name=exp_name, experiment_id=experiment_id)

            return experiment_id

        except Exception as e:
            logger.error("Failed to setup MLflow experiment", error=str(e))
            raise ModelTrainingError(f"MLflow setup failed: {e}")

    def prepare_model(self, model: nn.Module) -> None:
        """
        Prepare model for training.

        Args:
            model: PyTorch model to train
        """
        self.model=model.to(self.device)

        # Setup optimizer
        self.optimizer=optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler=optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )

        # Setup loss criterion (will be overridden in subclasses)
        self.criterion=nn.MSELoss()

        logger.info("Model prepared for training",
                   parameters=sum(p.numel() for p in self.model.parameters()),
                   trainable_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss=0.0
        total_samples=0
        correct_predictions=0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target=data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output=self.model(data)

            # Calculate loss (implement in subclasses)
            loss=self._calculate_loss(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_value
            )

            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # Calculate accuracy (implement in subclasses)
            accuracy=self._calculate_accuracy(output, target)
            correct_predictions += accuracy * data.size(0)

        avg_loss=total_loss / total_samples
        avg_accuracy=correct_predictions / total_samples

        return avg_loss, avg_accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate model for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy, medical_compliance_score)
        """
        self.model.eval()
        total_loss=0.0
        total_samples=0
        correct_predictions=0
        medical_compliance_scores=[]

        with torch.no_grad():
            for data, target in val_loader:
                data, target=data.to(self.device), target.to(self.device)

                # Forward pass
                output=self.model(data)

                # Calculate loss
                loss=self._calculate_loss(output, target)

                # Statistics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                # Calculate accuracy
                accuracy=self._calculate_accuracy(output, target)
                correct_predictions += accuracy * data.size(0)

                # Calculate medical compliance score
                compliance_score=self._calculate_medical_compliance(output, target)
                medical_compliance_scores.append(compliance_score)

        avg_loss=total_loss / total_samples
        avg_accuracy=correct_predictions / total_samples
        avg_compliance=np.mean(medical_compliance_scores) if medical_compliance_scores else 0.0

        return avg_loss, avg_accuracy, avg_compliance

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> ProcessingResult:
        """
        Train the model with comprehensive tracking and validation.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            ProcessingResult with training results
        """
        try:
            start_time=time.time()

            # Setup MLflow experiment
            experiment_id=self.setup_mlflow_experiment()

            with mlflow.start_run() as run:
                self.run_id=run.info.run_id

                # Log configuration
                mlflow.log_params({
                    "model_name": self.model_name,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": self.config.num_epochs,
                    "device": str(self.device)
                })

                # Create data loaders
                train_loader=DataLoader(
                    train_dataset,
                    _batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )

                val_loader=None
                if val_dataset:
                    val_loader=DataLoader(
                        val_dataset,
                        _batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True
                    )

                # Training loop
                best_val_loss=float('inf')
                patience_counter=0

                for epoch in range(self.config.num_epochs):
                    epoch_start=time.time()

                    # Training phase
                    train_loss, train_acc=self.train_epoch(train_loader)

                    # Validation phase
                    val_loss, val_acc, compliance_score=0.0, 0.0, 0.0
                    if val_loader:
                        val_loss, val_acc, compliance_score=self.validate_epoch(val_loader)

                    # Update learning rate
                    self.scheduler.step()
                    current_lr=self.scheduler.get_last_lr()[0]

                    # Calculate epoch time
                    epoch_time=time.time() - epoch_start

                    # Create training metrics
                    metrics=TrainingMetrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_accuracy=train_acc,
                        val_accuracy=val_acc,
                        learning_rate=current_lr,
                        epoch_time=epoch_time,
                        medical_compliance_score=compliance_score
                    )

                    self.training_history.append(metrics)

                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "learning_rate": current_lr,
                        "medical_compliance_score": compliance_score
                    }, step=epoch)

                    # Early stopping and model saving
                    if val_loader and val_loss < best_val_loss:
                        best_val_loss=val_loss
                        patience_counter=0

                        if self.config.save_best_model:
                            self._save_best_model(epoch, val_loss)
                    else:
                        patience_counter += 1

                    # Log progress
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs}",
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc,
                        compliance_score=compliance_score,
                        lr=current_lr
                    )

                    # Early stopping
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered", patience=patience_counter)
                        break

                # Save final model
                if self.model:
                    mlflow.pytorch.log_model(self.model, "model")

                # Generate training report
                training_report=self._generate_training_report()

                # Log training completion
                self.audit_manager.log_event(
                    event_type=AuditEventType.MODEL_TRAINING,
                    description=f"Model training completed: {self.model_name}",
                    severity=AuditSeverity.INFO,
                    metadata={
                        "model_name": self.model_name,
                        "epochs_trained": len(self.training_history),
                        "best_val_loss": best_val_loss,
                        "training_time_minutes": (time.time() - start_time) / 60,
                        "mlflow_run_id": self.run_id
                    }
                )

                return ProcessingResult(
                    success=True,
                    data={
                        "training_report": training_report,
                        "best_val_loss": best_val_loss,
                        "epochs_trained": len(self.training_history),
                        "mlflow_run_id": self.run_id,
                        "model_path": str(self.best_model_path) if self.best_model_path else None
                    },
                    metadata={
                        "training_time_seconds": time.time() - start_time,
                        "device": str(self.device),
                        "experiment_id": experiment_id
                    }
                )

        except Exception as e:
            logger.error("Model training failed", error=str(e))
            return ProcessingResult(
                success=False,
                error_message=f"Training failed: {e}",
                data={}
            )

    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate training loss (to be implemented in subclasses)."""
        return self.criterion(output, target)

    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy (to be implemented in subclasses)."""
        return 0.0  # Base implementation

    def _calculate_medical_compliance(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate medical compliance score (to be implemented in subclasses)."""
        return 1.0  # Base implementation

    def _save_best_model(self, epoch: int, val_loss: float) -> None:
        """Save the best model checkpoint."""
        if not self.model:
            return

        model_dir=Path("models") / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path=model_dir / f"best_model_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, self.best_model_path)

        logger.info("Best model saved", path=str(self.best_model_path), val_loss=val_loss)

    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.training_history:
            return {"message": "No training history available"}

        final_metrics=self.training_history[-1]
        best_val_epoch=min(self.training_history, key=lambda x: x.val_loss)

        return {
            "final_metrics": final_metrics.__dict__,
            "best_validation_epoch": best_val_epoch.__dict__,
            "total_epochs": len(self.training_history),
            "total_training_time": sum(m.epoch_time for m in self.training_history),
            "average_epoch_time": np.mean([m.epoch_time for m in self.training_history]),
            "convergence_analysis": {
                "final_train_loss": final_metrics.train_loss,
                "final_val_loss": final_metrics.val_loss,
                "best_val_loss": best_val_epoch.val_loss,
                "overfitting_indicator": final_metrics.train_loss - final_metrics.val_loss
            },
            "medical_compliance": {
                "average_compliance_score": np.mean([m.medical_compliance_score for m in self.training_history]),
                "final_compliance_score": final_metrics.medical_compliance_score,
                "compliance_trend": "improving" if final_metrics.medical_compliance_score > self.training_history[0].medical_compliance_score else "stable"
            }
        }


class CPRModelTrainer(ModelTrainer):
    """
    Specialized trainer for CPR Quality Assessment models.

    This trainer implements CPR-specific loss functions, metrics,
    and medical compliance validation.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__("CPRQualityAssessment", config)

        # CPR-specific configuration
        self.aha_thresholds={
            'compression_depth_min': 50,
            'compression_depth_max': 60,
            'compression_rate_min': 100,
            'compression_rate_max': 120,
            'quality_threshold': 0.8
        }

    def prepare_model(self, model: Optional[nn.Module] = None) -> None:
        """Prepare CPR quality assessment model."""
        if model is None:
            model=CPRQualityNet()

        super().prepare_model(model)

        # CPR-specific multi-task loss
        self.criterion=self._create_cpr_loss_function()

    def _create_cpr_loss_function(self) -> nn.Module:
        """Create multi-task loss function for CPR assessment."""
        class CPRMultiTaskLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse_loss=nn.MSELoss()
                self.bce_loss=nn.BCELoss()
                self.ce_loss=nn.CrossEntropyLoss()

            def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
                # Assuming targets contain multiple components
                # This is a simplified version - adjust based on actual target structure

                total_loss=0.0

                # Regression losses for continuous metrics
                if 'compression_depth' in predictions:
                    depth_loss=self.mse_loss(predictions['compression_depth'], targets[:, 0:1])
                    total_loss += depth_loss

                if 'compression_rate' in predictions:
                    rate_loss=self.mse_loss(predictions['compression_rate'], targets[:, 1:2])
                    total_loss += rate_loss

                # Binary classification losses for quality scores
                quality_metrics=['hand_position', 'release_completeness', 'rhythm_consistency']
                for i, metric in enumerate(quality_metrics):
                    if metric in predictions:
                        metric_loss=self.bce_loss(predictions[metric], targets[:, i+2:i+3])
                        total_loss += metric_loss

                # Overall quality classification
                if 'quality_class' in predictions:
                    quality_targets=targets[:, -1].long()  # Assuming last column is quality class
                    quality_loss=self.ce_loss(predictions['quality_class'], quality_targets)
                    total_loss += quality_loss * 2.0  # Higher weight for overall quality

                return total_loss

        return CPRMultiTaskLoss()

    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate CPR-specific accuracy."""
        if isinstance(output, dict):
            # Multi-task output
            accuracies=[]

            # Quality class accuracy
            if 'quality_class' in output:
                pred_classes=torch.argmax(output['quality_class'], dim=1)
                true_classes=target[:, -1].long()
                class_acc=(pred_classes== true_classes).float().mean().item()
                accuracies.append(class_acc)

            # Threshold-based accuracy for continuous metrics
            continuous_metrics=['compression_depth', 'compression_rate', 'hand_position',
                                'release_completeness', 'rhythm_consistency']

            for i, metric in enumerate(continuous_metrics):
                if metric in output:
                    pred=output[metric].squeeze()
                    true=target[:, i]

                    # Define accuracy based on medical thresholds
                    if metric== 'compression_depth':
                        correct=((pred * 60 >= 50) & (pred * 60 <= 60)).float()
                    elif metric== 'compression_rate':
                        rate_pred=80 + pred * 40
                        correct=((rate_pred >= 100) & (rate_pred <= 120)).float()
                    else:
                        correct=(torch.abs(pred - true) < 0.1).float()

                    accuracies.append(correct.mean().item())

            return np.mean(accuracies) if accuracies else 0.0
        else:
            # Single output
            return ((output - target).abs() < 0.1).float().mean().item()

    def _calculate_medical_compliance(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate medical compliance score based on AHA guidelines."""
        if not isinstance(output, dict):
            return 1.0

        compliance_scores=[]

        # Compression depth compliance
        if 'compression_depth' in output:
            depth_mm=output['compression_depth'].squeeze() * 60
            depth_compliance=((depth_mm >= 50) & (depth_mm <= 60)).float().mean().item()
            compliance_scores.append(depth_compliance)

        # Compression rate compliance
        if 'compression_rate' in output:
            rate_cpm=80 + output['compression_rate'].squeeze() * 40
            rate_compliance=((rate_cpm >= 100) & (rate_cpm <= 120)).float().mean().item()
            compliance_scores.append(rate_compliance)

        # Quality metrics compliance
        quality_metrics=['hand_position', 'release_completeness', 'rhythm_consistency']
        quality_thresholds=[0.8, 0.9, 0.7]

        for metric, threshold in zip(quality_metrics, quality_thresholds):
            if metric in output:
                metric_compliance=(output[metric].squeeze() >= threshold).float().mean().item()
                compliance_scores.append(metric_compliance)

        return np.mean(compliance_scores) if compliance_scores else 1.0
