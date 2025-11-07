#!/usr/bin/env python3
"""
Azure ML Workspace Setup for SMART-TRAIN Medical AI Platform
Configures Azure ML resources for medical training data analysis
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Workspace, 
    AmlCompute, 
    Environment,
    Data,
    Datastore
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureMLSetup:
    """Setup and configure Azure ML workspace for medical AI training"""
    
    def __init__(self, config_path: str = "config/azure_config.yaml"):
        """Initialize Azure ML setup with configuration"""
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Substitute environment variables
        self.config = self._substitute_env_vars(self.config)
        
        # Initialize Azure credentials
        self.credential = DefaultAzureCredential()
        
        # Initialize ML client
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=self.config['azure']['subscription_id'],
            resource_group_name=self.config['azure']['resource_group'],
            workspace_name=self.config['azure']['workspace_name']
        )
        
    def _substitute_env_vars(self, config):
        """Substitute environment variables in configuration"""
        import re
        
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Replace ${VAR_NAME} with environment variable value
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, obj)
                for match in matches:
                    env_value = os.getenv(match)
                    if env_value:
                        obj = obj.replace(f'${{{match}}}', env_value)
                return obj
            return obj
        
        return replace_env_vars(config)
    
    def create_workspace(self):
        """Create Azure ML workspace"""
        try:
            logger.info("Creating Azure ML workspace...")
            
            workspace = Workspace(
                name=self.config['azure']['workspace_name'],
                resource_group=self.config['azure']['resource_group'],
                location=self.config['azure']['location'],
                description="SMART-TRAIN Medical AI Training Platform",
                tags={
                    "project": "smart-train",
                    "domain": "medical-ai",
                    "compliance": "iso-13485"
                }
            )
            
            workspace = self.ml_client.workspaces.begin_create(workspace).result()
            logger.info(f"Workspace '{workspace.name}' created successfully")
            return workspace
            
        except Exception as e:
            logger.error(f"Error creating workspace: {str(e)}")
            raise
    
    def create_compute_clusters(self):
        """Create compute clusters for training and inference"""
        compute_configs = self.config['azure']['compute']
        
        for compute_name, compute_config in compute_configs.items():
            try:
                logger.info(f"Creating compute cluster: {compute_name}")
                
                compute_cluster = AmlCompute(
                    name=compute_config['name'],
                    type=compute_config['type'],
                    size=compute_config['vm_size'],
                    min_instances=compute_config['min_instances'],
                    max_instances=compute_config['max_instances'],
                    idle_time_before_scale_down=compute_config.get('idle_seconds_before_scaledown', 300),
                    tier="Dedicated"
                )
                
                self.ml_client.compute.begin_create_or_update(compute_cluster).result()
                logger.info(f"Compute cluster '{compute_config['name']}' created successfully")
                
            except Exception as e:
                logger.error(f"Error creating compute cluster {compute_name}: {str(e)}")
                # Continue with other clusters
                continue
    
    def setup_data_storage(self):
        """Setup Azure Blob Storage for medical training data"""
        try:
            logger.info("Setting up data storage...")
            
            storage_config = self.config['azure']['storage']
            
            # Create blob datastore
            blob_datastore = Datastore(
                name=storage_config['data_store_name'],
                type="azure_blob",
                account_name=storage_config['account_name'],
                container_name=storage_config['container_name'],
                description="Medical training data storage with HIPAA compliance"
            )
            
            self.ml_client.datastores.create_or_update(blob_datastore)
            logger.info("Data storage configured successfully")
            
            return blob_datastore
            
        except Exception as e:
            logger.error(f"Error setting up data storage: {str(e)}")
            raise
    
    def create_environments(self):
        """Create conda environments for training and inference"""
        env_configs = self.config['azure']['environments']
        
        for env_name, env_config in env_configs.items():
            try:
                logger.info(f"Creating environment: {env_name}")
                
                # Create conda environment file if it doesn't exist
                conda_file_path = env_config['conda_file']
                if not os.path.exists(conda_file_path):
                    self._create_conda_environment_file(conda_file_path, env_name)
                
                environment = Environment(
                    name=env_config['name'],
                    conda_file=conda_file_path,
                    image=env_config['docker_image'],
                    description=f"Environment for {env_name} in SMART-TRAIN"
                )
                
                self.ml_client.environments.create_or_update(environment)
                logger.info(f"Environment '{env_config['name']}' created successfully")
                
            except Exception as e:
                logger.error(f"Error creating environment {env_name}: {str(e)}")
                continue
    
    def _create_conda_environment_file(self, file_path: str, env_type: str):
        """Create conda environment YAML file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if env_type == "training_env":
            conda_content = """
name: smart-train-training
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - opencv
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - pip
  - pip:
    - azure-ai-ml
    - mediapipe
    - ultralytics
    - mlflow
    - wandb
"""
        else:  # inference_env
            conda_content = """
name: smart-train-inference
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy
  - opencv
  - pip
  - pip:
    - fastapi
    - uvicorn
    - azure-ai-ml
    - mediapipe
    - torch
    - torchvision
"""
        
        with open(file_path, 'w') as f:
            f.write(conda_content.strip())
    
    def setup_monitoring_and_logging(self):
        """Setup monitoring and audit logging for medical compliance"""
        try:
            logger.info("Setting up monitoring and compliance logging...")
            
            # Create audit log directory structure
            audit_paths = [
                "compliance/audit_logs",
                "compliance/quality_metrics",
                "compliance/risk_management"
            ]
            
            for path in audit_paths:
                os.makedirs(path, exist_ok=True)
                
            # Create initial compliance configuration
            compliance_config = {
                "medical_device_compliance": {
                    "iso_13485": True,
                    "iec_62304": True,
                    "fda_510k_preparation": True
                },
                "data_privacy": {
                    "hipaa_compliance": True,
                    "gdpr_compliance": True,
                    "anonymization_required": True
                },
                "audit_requirements": {
                    "model_versioning": True,
                    "data_lineage": True,
                    "performance_monitoring": True,
                    "bias_detection": True
                }
            }
            
            with open("compliance/compliance_config.yaml", 'w') as f:
                yaml.dump(compliance_config, f, default_flow_style=False)
                
            logger.info("Monitoring and compliance setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up monitoring: {str(e)}")
            raise
    
    def validate_setup(self):
        """Validate that all Azure ML resources are properly configured"""
        try:
            logger.info("Validating Azure ML setup...")
            
            # Check workspace
            workspace = self.ml_client.workspaces.get(self.config['azure']['workspace_name'])
            logger.info(f"✓ Workspace '{workspace.name}' is accessible")
            
            # Check compute clusters
            computes = list(self.ml_client.compute.list())
            compute_names = [c.name for c in computes]
            
            expected_computes = [
                self.config['azure']['compute']['training_cluster']['name'],
                self.config['azure']['compute']['inference_cluster']['name'],
                self.config['azure']['compute']['gpu_cluster']['name']
            ]
            
            for compute_name in expected_computes:
                if compute_name in compute_names:
                    logger.info(f"✓ Compute '{compute_name}' is available")
                else:
                    logger.warning(f"⚠ Compute '{compute_name}' not found")
            
            # Check datastores
            datastores = list(self.ml_client.datastores.list())
            datastore_names = [d.name for d in datastores]
            
            expected_datastore = self.config['azure']['storage']['data_store_name']
            if expected_datastore in datastore_names:
                logger.info(f"✓ Datastore '{expected_datastore}' is configured")
            else:
                logger.warning(f"⚠ Datastore '{expected_datastore}' not found")
            
            logger.info("Azure ML setup validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Setup validation failed: {str(e)}")
            return False
    
    def run_complete_setup(self):
        """Run complete Azure ML setup process"""
        try:
            logger.info("Starting complete Azure ML setup for SMART-TRAIN...")
            
            # Step 1: Create workspace
            self.create_workspace()
            
            # Step 2: Create compute clusters
            self.create_compute_clusters()
            
            # Step 3: Setup data storage
            self.setup_data_storage()
            
            # Step 4: Create environments
            self.create_environments()
            
            # Step 5: Setup monitoring and compliance
            self.setup_monitoring_and_logging()
            
            # Step 6: Validate setup
            if self.validate_setup():
                logger.info("🎉 Azure ML setup completed successfully!")
                logger.info("Next steps:")
                logger.info("1. Run data collection pipeline")
                logger.info("2. Start model training")
                logger.info("3. Deploy inference endpoints")
            else:
                logger.warning("Setup completed with some issues. Please review the logs.")
                
        except Exception as e:
            logger.error(f"Azure ML setup failed: {str(e)}")
            raise

def main():
    """Main function to run Azure ML setup"""
    try:
        # Check if we're in the correct directory
        if not os.path.exists("config/azure_config.yaml"):
            logger.error("Please run this script from the project root directory")
            return
        
        # Initialize and run setup
        azure_setup = AzureMLSetup()
        azure_setup.run_complete_setup()
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()