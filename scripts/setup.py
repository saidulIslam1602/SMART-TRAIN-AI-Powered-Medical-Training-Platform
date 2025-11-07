#!/usr/bin/env python3
"""
SMART-TRAIN Platform Setup Script

Industry-standard setup script for initializing the SMART-TRAIN medical AI platform.
Follows Python packaging conventions and enterprise deployment practices.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('smart_train.setup')


class SmartTrainSetup:
    """Enterprise-grade setup manager for SMART-TRAIN platform."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize setup manager."""
        self.project_root = project_root or Path.cwd()
        self.requirements_files = [
            'requirements.txt',
            'requirements_core.txt'
        ]
        
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        min_version = (3, 9)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            logger.error(f"Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}")
            return False
        
        logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}")
        return True
    
    def install_dependencies(self, requirements_file: str = 'requirements_core.txt') -> bool:
        """Install Python dependencies."""
        try:
            requirements_path = self.project_root / requirements_file
            
            if not requirements_path.exists():
                logger.error(f"Requirements file not found: {requirements_path}")
                return False
            
            logger.info(f"Installing dependencies from {requirements_file}")
            
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)
            ], capture_output=True, text=True, check=True)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup environment configuration."""
        try:
            env_example = self.project_root / 'env_example.txt'
            env_file = self.project_root / '.env'
            
            if env_example.exists() and not env_file.exists():
                logger.info("Creating .env file from template")
                
                # Copy template and add development defaults
                with open(env_example, 'r') as f:
                    content = f.read()
                
                # Add development defaults
                content += "\n# Development defaults (auto-generated)\n"
                content += "SMART_TRAIN_JWT_SECRET=dev-jwt-secret-change-in-production\n"
                content += "ENVIRONMENT=development\n"
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                logger.info("Environment file created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create required directory structure."""
        try:
            directories = [
                'logs',
                'data/raw',
                'data/processed',
                'data/models',
                'compliance/audit_logs',
                'compliance/reports',
                'tests/unit',
                'tests/integration',
                'docs/api',
                'deployment/docker',
                'deployment/kubernetes'
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            
            logger.info("Directory structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate setup completion."""
        try:
            logger.info("Validating setup...")
            
            # Check if we can import core modules
            sys.path.insert(0, str(self.project_root / 'src'))
            
            from smart_train.core.config import get_config
            from smart_train.core.logging import setup_logging
            
            # Test configuration loading
            config = get_config(self.project_root / 'config' / 'smart_train.yaml')
            
            # Test logging setup
            setup_logging(log_level='INFO', enable_console=False)
            
            logger.info("Setup validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False
    
    def run_setup(self, install_deps: bool = True) -> bool:
        """Run complete setup process."""
        logger.info("Starting SMART-TRAIN platform setup")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Directory structure", self.create_directories),
            ("Environment setup", self.setup_environment),
        ]
        
        if install_deps:
            steps.append(("Dependency installation", self.install_dependencies))
        
        steps.append(("Setup validation", self.validate_setup))
        
        for step_name, step_func in steps:
            logger.info(f"Running: {step_name}")
            
            if not step_func():
                logger.error(f"Setup failed at: {step_name}")
                return False
        
        logger.info("🎉 SMART-TRAIN platform setup completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Review configuration in config/smart_train.yaml")
        logger.info("  2. Set up Azure credentials (if using cloud features)")
        logger.info("  3. Run: python examples/system_demonstration.py")
        
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='SMART-TRAIN Platform Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup.py                    # Full setup
  python scripts/setup.py --no-deps         # Setup without installing dependencies
  python scripts/setup.py --verbose         # Verbose output
        """
    )
    
    parser.add_argument(
        '--no-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd().parent,  # Assume we're in scripts/ directory
        help='Project root directory'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_manager = SmartTrainSetup(args.project_root)
    success = setup_manager.run_setup(install_deps=not args.no_deps)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
