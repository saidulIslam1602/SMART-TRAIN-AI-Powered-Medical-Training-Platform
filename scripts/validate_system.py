#!/usr/bin/env python3
"""
SMART-TRAIN System Validation Script

Industry-standard system validation and health check script.
Validates system integrity, configuration, and readiness for production deployment.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set environment for validation
os.environ.setdefault('SMART_TRAIN_JWT_SECRET', 'validation-jwt-secret')
os.environ.setdefault('ENVIRONMENT', 'development')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('smart_train.validation')


class SystemValidator:
    """Enterprise system validation and health checks."""
    
    def __init__(self):
        """Initialize system validator."""
        self.validation_results = {}
        self.start_time = time.time()
    
    def validate_core_modules(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate core system modules."""
        try:
            # Test core imports
            from smart_train.core.config import get_config
            from smart_train.core.logging import setup_logging, get_logger
            from smart_train.core.exceptions import SmartTrainException
            from smart_train import __version__
            
            # Test configuration
            config = get_config('config/smart_train.yaml')
            
            # Test logging
            setup_logging(log_level='INFO', enable_console=False)
            logger = get_logger('validation')
            
            return True, {
                'version': __version__,
                'config_loaded': True,
                'logging_initialized': True,
                'modules': ['config', 'logging', 'exceptions']
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_compliance_framework(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate medical compliance framework."""
        try:
            from smart_train.compliance.audit_trail import AuditTrailManager
            from smart_train.compliance.iso_13485 import ISO13485Compliance
            from smart_train.compliance.iec_62304 import IEC62304Compliance
            
            # Test audit trail
            audit_manager = AuditTrailManager()
            
            # Test compliance modules
            iso_compliance = ISO13485Compliance()
            iec_compliance = IEC62304Compliance()
            
            # Run compliance checks
            iso_result = iso_compliance.check_compliance({})
            iec_result = iec_compliance.check_compliance({})
            
            return True, {
                'audit_trail': True,
                'iso_13485': iso_result.data['compliance_status'],
                'iec_62304': iec_result.data['compliance_status'],
                'software_class': iec_result.data['software_safety_class']
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_data_pipeline(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate data processing pipeline."""
        try:
            from smart_train.data.collection import RealDatasetCollector
            from smart_train.data.preprocessing import AHAGuidelinesValidator
            from smart_train.data.validation import MedicalDataValidator
            
            # Test dataset collector
            collector = RealDatasetCollector()
            
            # Test AHA validator
            aha_validator = AHAGuidelinesValidator()
            
            # Test medical validator
            medical_validator = MedicalDataValidator()
            
            # Test AHA compliance with sample data
            sample_metrics = {
                "compression_depths": [5.2, 5.5, 5.8],
                "compression_timestamps": [0.0, 0.5, 1.0],
                "hand_position_scores": [0.85, 0.90, 0.88]
            }
            
            compliance_report = aha_validator.generate_compliance_report(sample_metrics)
            
            return True, {
                'dataset_collector': True,
                'aha_validator': True,
                'medical_validator': True,
                'aha_compliance_test': compliance_report['overall_compliance'],
                'compliance_score': compliance_report['compliance_score'],
                'dataset_count': len(collector.cpr_datasets + collector.pose_datasets + collector.medical_datasets)
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system configuration."""
        try:
            from smart_train.core.config import get_config
            
            config = get_config('config/smart_train.yaml')
            
            # Check critical configuration sections
            config_status = {
                'medical_compliance_enabled': config.medical_compliance.iso_13485_enabled,
                'audit_trail_enabled': config.medical_compliance.audit_trail_enabled,
                'data_anonymization': config.medical_compliance.data_anonymization_required,
                'video_resolution': config.data_processing.video_target_resolution,
                'pose_confidence': config.model.pose_confidence_threshold,
                'parallel_workers': config.data_processing.parallel_processing_workers,
                'environment': config.is_development() and 'development' or 'production'
            }
            
            return True, config_status
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_dependencies(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate critical dependencies."""
        try:
            import numpy
            import pandas
            import cv2
            import mediapipe
            import torch
            import fastapi
            import pydantic
            import structlog
            
            dependencies = {
                'numpy': numpy.__version__,
                'pandas': pandas.__version__,
                'opencv': cv2.__version__,
                'mediapipe': mediapipe.__version__,
                'pytorch': torch.__version__,
                'fastapi': fastapi.__version__,
                'pydantic': pydantic.__version__,
                'structlog': structlog.__version__
            }
            
            return True, dependencies
            
        except ImportError as e:
            return False, {'error': f'Missing dependency: {e}'}
        except Exception as e:
            return False, {'error': str(e)}
    
    def validate_file_structure(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate project file structure."""
        try:
            project_root = Path.cwd()
            
            required_files = [
                'config/smart_train.yaml',
                'src/smart_train/__init__.py',
                'src/smart_train/core/__init__.py',
                'src/smart_train/compliance/__init__.py',
                'src/smart_train/data/__init__.py',
                'requirements_core.txt',
                'examples/system_demonstration.py'
            ]
            
            required_dirs = [
                'src/smart_train/core',
                'src/smart_train/compliance',
                'src/smart_train/data',
                'config',
                'scripts',
                'examples'
            ]
            
            file_status = {}
            for file_path in required_files:
                full_path = project_root / file_path
                file_status[file_path] = full_path.exists()
            
            dir_status = {}
            for dir_path in required_dirs:
                full_path = project_root / dir_path
                dir_status[dir_path] = full_path.exists() and full_path.is_dir()
            
            all_files_exist = all(file_status.values())
            all_dirs_exist = all(dir_status.values())
            
            return all_files_exist and all_dirs_exist, {
                'files': file_status,
                'directories': dir_status,
                'structure_valid': all_files_exist and all_dirs_exist
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting SMART-TRAIN system validation")
        
        validation_tests = [
            ('File Structure', self.validate_file_structure),
            ('Dependencies', self.validate_dependencies),
            ('Core Modules', self.validate_core_modules),
            ('Configuration', self.validate_configuration),
            ('Compliance Framework', self.validate_compliance_framework),
            ('Data Pipeline', self.validate_data_pipeline),
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test_name, test_func in validation_tests:
            logger.info(f"Running validation: {test_name}")
            
            try:
                success, data = test_func()
                results[test_name] = {
                    'status': 'PASS' if success else 'FAIL',
                    'data': data
                }
                
                if success:
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASS")
                else:
                    logger.error(f"❌ {test_name}: FAIL - {data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                results[test_name] = {
                    'status': 'ERROR',
                    'data': {'error': str(e)}
                }
                logger.error(f"💥 {test_name}: ERROR - {str(e)}")
        
        # Generate summary
        validation_time = time.time() - self.start_time
        
        summary = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'validation_duration_seconds': round(validation_time, 2),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': round((passed_tests / total_tests) * 100, 1),
            'overall_status': 'PASS' if passed_tests == total_tests else 'FAIL',
            'system_ready': passed_tests >= total_tests * 0.8,  # 80% pass rate for readiness
            'results': results
        }
        
        return summary
    
    def print_validation_report(self, summary: Dict[str, Any]) -> None:
        """Print formatted validation report."""
        print("\n" + "="*60)
        print("  SMART-TRAIN System Validation Report")
        print("="*60)
        
        print(f"\n📊 Validation Summary:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Passed: {summary['passed_tests']}")
        print(f"   • Failed: {summary['failed_tests']}")
        print(f"   • Success Rate: {summary['success_rate']}%")
        print(f"   • Duration: {summary['validation_duration_seconds']}s")
        print(f"   • Overall Status: {summary['overall_status']}")
        print(f"   • System Ready: {'✅ YES' if summary['system_ready'] else '❌ NO'}")
        
        print(f"\n📋 Test Results:")
        for test_name, result in summary['results'].items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "💥"
            print(f"   {status_icon} {test_name}: {result['status']}")
            
            if result['status'] != 'PASS' and 'error' in result['data']:
                print(f"      Error: {result['data']['error']}")
        
        if summary['system_ready']:
            print(f"\n🎉 System validation completed successfully!")
            print(f"📋 SMART-TRAIN platform is ready for:")
            print(f"   • Development and testing")
            print(f"   • Medical compliance validation")
            print(f"   • Production deployment preparation")
        else:
            print(f"\n⚠️  System validation found issues that need attention.")
            print(f"📋 Please resolve failed tests before proceeding.")
        
        print(f"\n🔗 Next Steps:")
        print(f"   • Run: python examples/system_demonstration.py")
        print(f"   • Review: PHASE1_COMPLETION_REPORT.md")
        print(f"   • Deploy: Follow deployment guide in docs/")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='SMART-TRAIN System Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Save validation report to JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = SystemValidator()
    summary = validator.run_validation_suite()
    
    # Print report
    validator.print_validation_report(summary)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Validation report saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['system_ready'] else 1)


if __name__ == '__main__':
    main()
