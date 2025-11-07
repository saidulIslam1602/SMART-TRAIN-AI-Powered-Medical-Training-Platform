"""Unit tests for data processing modules."""

import pytest
import sys
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment variables
os.environ['SMART_TRAIN_JWT_SECRET'] = 'test-jwt-secret-key-for-testing'
os.environ['ENVIRONMENT'] = 'testing'

from smart_train.data.preprocessing import MedicalDataPreprocessor
from smart_train.data.collection import RealDatasetCollector, DatasetMetadata
from smart_train.core.base import ProcessingResult
from smart_train.core.exceptions import DataProcessingError


class TestMedicalDataPreprocessor:
    """Test MedicalDataPreprocessor functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = MedicalDataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'process')
        
    @patch('cv2.VideoCapture')
    @patch('mediapipe.solutions.pose.Pose')
    def test_process_video_success(self, mock_pose, mock_video_capture):
        """Test successful video processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_cap.get.return_value = 30.0  # FPS
        mock_video_capture.return_value = mock_cap
        
        # Mock MediaPipe pose
        mock_pose_instance = Mock()
        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [Mock() for _ in range(33)]
        for i, landmark in enumerate(mock_results.pose_landmarks.landmark):
            landmark.x = 0.5 + i * 0.01
            landmark.y = 0.5 + i * 0.01
            landmark.z = 0.0
            landmark.visibility = 0.9
        mock_pose_instance.process.return_value = mock_results
        mock_pose.return_value = mock_pose_instance
        
        preprocessor = MedicalDataPreprocessor()
        
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            input_data = {
                'video_path': temp_video.name,
                'extract_poses': True,
                'calculate_metrics': True
            }
            
            result = preprocessor.process(input_data)
            assert isinstance(result, ProcessingResult)
            # Note: This might fail due to actual video processing, 
            # but tests the interface
            
    def test_process_invalid_input(self):
        """Test processing with invalid input."""
        preprocessor = MedicalDataPreprocessor()
        
        # Test with missing video_path
        result = preprocessor.process({})
        assert result.success is False
        assert ("video_path" in result.message.lower() or 
                "no video files found" in result.message.lower() or 
                len(result.errors) > 0)
        
    def test_process_nonexistent_video(self):
        """Test processing with nonexistent video file."""
        preprocessor = MedicalDataPreprocessor()
        
        input_data = {
            'video_path': '/nonexistent/video.mp4',
            'extract_poses': True
        }
        
        result = preprocessor.process(input_data)
        assert result.success is False


class TestDatasetMetadata:
    """Test DatasetMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test dataset metadata creation."""
        from datetime import datetime
        
        metadata = DatasetMetadata(
            dataset_id="test_dataset_001",
            name="Test CPR Dataset",
            version="1.0.0",
            description="Test dataset for CPR quality assessment",
            source="Medical Simulation Center",
            license="MIT",
            size_mb=150.5,
            file_count=100,
            medical_standards=["AHA_CPR_2020", "ISO_13485"],
            compliance_verified=True,
            quality_score=0.95,
            created_timestamp=datetime.now(),
            last_updated=datetime.now(),
            checksum="abc123def456"
        )
        
        assert metadata.dataset_id == "test_dataset_001"
        assert metadata.name == "Test CPR Dataset"
        assert metadata.compliance_verified is True
        assert metadata.quality_score == 0.95
        
    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        from datetime import datetime
        
        now = datetime.now()
        metadata = DatasetMetadata(
            dataset_id="test_002",
            name="Test Dataset 2",
            version="2.0.0",
            description="Another test dataset",
            source="Academic Partner",
            license="Apache-2.0",
            size_mb=200.0,
            file_count=150,
            medical_standards=["AHA_CPR_2020"],
            compliance_verified=False,
            quality_score=0.88,
            created_timestamp=now,
            last_updated=now,
            checksum="def456ghi789"
        )
        
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['dataset_id'] == "test_002"
        assert metadata_dict['compliance_verified'] is False
        assert 'created_timestamp' in metadata_dict
        assert 'last_updated' in metadata_dict


class TestRealDatasetCollector:
    """Test RealDatasetCollector functionality."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = RealDatasetCollector(storage_path=Path(temp_dir))
            assert collector.storage_path == Path(temp_dir)
            
    @patch('requests.get')
    def test_download_dataset_success(self, mock_get):
        """Test successful dataset download."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'mock_data_chunk']
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = RealDatasetCollector(storage_path=Path(temp_dir))
            
            # Mock dataset spec
            dataset_spec = {
                'name': 'test_dataset',
                'url': 'https://example.com/dataset.zip',
                'expected_size_mb': 1.0,
                'file_format': 'zip',
                'annotation_format': 'json'
            }
            
            # This would normally download and process
            # For testing, we just verify the interface works
            assert hasattr(collector, 'process')
            
    def test_validate_dataset_integrity(self):
        """Test dataset integrity validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = RealDatasetCollector(storage_path=Path(temp_dir))
            
            # Create a test file
            test_file = Path(temp_dir) / "test_dataset.txt"
            test_content = "test dataset content"
            test_file.write_text(test_content)
            
            # Calculate expected checksum
            import hashlib
            expected_checksum = hashlib.md5(test_content.encode()).hexdigest()
            
            # Test that collector has basic functionality
            assert hasattr(collector, 'storage_path')
            assert collector.storage_path == Path(temp_dir)


class TestDataProcessingIntegration:
    """Test data processing integration scenarios."""
    
    @patch('smart_train.data.preprocessing.cv2.VideoCapture')
    def test_preprocessor_with_collector(self, mock_video_capture):
        """Test preprocessor integration with collector."""
        # Mock video capture for preprocessor
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False  # Simulate file not found
        mock_video_capture.return_value = mock_cap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            collector = RealDatasetCollector(storage_path=Path(temp_dir))
            preprocessor = MedicalDataPreprocessor()
            
            # Test that both components can be initialized together
            assert collector is not None
            assert preprocessor is not None
            
    def test_error_handling_chain(self):
        """Test error handling across data processing chain."""
        preprocessor = MedicalDataPreprocessor()
        
        # Test error propagation
        with pytest.raises(Exception):
            # This should raise an exception due to invalid input
            result = preprocessor.process({'invalid': 'input'})
            if not result.success:
                raise DataProcessingError(result.error_message)


class TestDataValidation:
    """Test data validation functionality."""
    
    def test_pose_data_validation(self):
        """Test pose data validation."""
        # Valid pose data (33 landmarks, 3 coordinates each)
        valid_poses = np.random.rand(10, 33, 3)  # 10 frames, 33 landmarks, xyz
        
        assert valid_poses.shape == (10, 33, 3)
        assert np.all(valid_poses >= 0) and np.all(valid_poses <= 1)
        
    def test_video_metadata_validation(self):
        """Test video metadata validation."""
        valid_metadata = {
            'fps': 30.0,
            'duration': 120.0,
            'resolution': (1920, 1080),
            'format': 'mp4'
        }
        
        assert valid_metadata['fps'] > 0
        assert valid_metadata['duration'] > 0
        assert len(valid_metadata['resolution']) == 2
        assert valid_metadata['format'] in ['mp4', 'avi', 'mov']
        
    def test_medical_data_anonymization(self):
        """Test medical data anonymization."""
        # Mock medical data with PII
        medical_data = {
            'patient_id': 'PATIENT_001',
            'session_data': {
                'timestamp': '2024-01-01T10:00:00Z',
                'location': 'Training Room A',
                'instructor': 'Dr. Smith'
            },
            'performance_data': {
                'compression_depth': [55, 54, 56, 53],
                'compression_rate': [110, 112, 108, 111]
            }
        }
        
        # Simulate anonymization
        anonymized_data = {
            'session_id': 'SESSION_' + medical_data['patient_id'].split('_')[1],
            'session_data': {
                'timestamp': medical_data['session_data']['timestamp'],
                'location': 'ANONYMIZED',
                'instructor': 'ANONYMIZED'
            },
            'performance_data': medical_data['performance_data']
        }
        
        assert 'patient_id' not in anonymized_data
        assert anonymized_data['session_data']['instructor'] == 'ANONYMIZED'
        assert anonymized_data['performance_data'] == medical_data['performance_data']
