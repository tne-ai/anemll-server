#!/usr/bin/env python
"""
Model discovery and registry for dynamic model loading.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from model_info import ModelInfo

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Discovers and catalogs available Anemll models"""
    
    def __init__(self, models_base_dir: str):
        """
        Initialize model registry.
        
        Args:
            models_base_dir: Base directory to scan for models
        """
        self.models_base_dir = Path(models_base_dir).resolve()
        self.available_models: Dict[str, ModelInfo] = {}
        self._scan_recursive = True
        
        logger.info(f"Initializing ModelRegistry with base directory: {self.models_base_dir}")
        
        if not self.models_base_dir.exists():
            logger.error(f"Models base directory does not exist: {self.models_base_dir}")
            raise FileNotFoundError(f"Models directory not found: {self.models_base_dir}")
            
        self.scan_models()
    
    def scan_models(self) -> Dict[str, ModelInfo]:
        """
        Recursively scan for valid Anemll models.
        
        Returns:
            Dictionary mapping model names to ModelInfo objects
        """
        logger.info(f"Scanning for models in {self.models_base_dir}")
        self.available_models.clear()
        
        # Check if the base directory itself is a model
        self._check_single_model(self.models_base_dir)
        
        # If recursive scanning is enabled, check subdirectories
        if self._scan_recursive:
            self._scan_directory_recursive(self.models_base_dir)
        else:
            self._scan_directory_flat(self.models_base_dir)
            
        logger.info(f"Found {len(self.available_models)} valid models: {list(self.available_models.keys())}")
        return self.available_models
    
    def _check_single_model(self, directory: Path) -> bool:
        """
        Check if a single directory contains a valid model.
        
        Args:
            directory: Directory to check
            
        Returns:
            True if valid model found and added
        """
        meta_yaml_path = directory / "meta.yaml"
        
        if not meta_yaml_path.exists():
            return False
            
        model_info = ModelInfo.from_meta_yaml(directory)
        if model_info and model_info.validate_files():
            self.available_models[model_info.name] = model_info
            logger.info(f"Added model: {model_info.name} from {directory}")
            return True
        else:
            logger.warning(f"Invalid model in directory: {directory}")
            return False
    
    def _scan_directory_recursive(self, base_dir: Path):
        """
        Recursively scan directory tree for models.
        
        Args:
            base_dir: Base directory to start scanning from
        """
        try:
            for root, dirs, files in os.walk(base_dir):
                root_path = Path(root)
                
                # Skip the base directory if we already checked it
                if root_path == base_dir:
                    continue
                
                # Skip hidden directories and temporary directories
                if any(part.startswith('.') or part.endswith('.tmp') for part in root_path.parts):
                    continue
                    
                # Check if this directory contains a model
                self._check_single_model(root_path)
                
        except Exception as e:
            logger.error(f"Error during recursive scan: {str(e)}")
    
    def _scan_directory_flat(self, base_dir: Path):
        """
        Scan only immediate subdirectories for models.
        
        Args:
            base_dir: Directory to scan
        """
        try:
            for item in base_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    self._check_single_model(item)
        except Exception as e:
            logger.error(f"Error during flat scan: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get metadata for specific model.
        
        Args:
            model_name: Name of the model to look up
            
        Returns:
            ModelInfo object or None if not found
        """
        return self.available_models.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available model names.
        
        Returns:
            List of model names
        """
        return list(self.available_models.keys())
    
    def refresh(self):
        """Re-scan directory for new models"""
        logger.info("Refreshing model registry")
        self.scan_models()
    
    def has_model(self, model_name: str) -> bool:
        """
        Check if a model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available
        """
        return model_name in self.available_models
    
    def get_models_summary(self) -> Dict[str, Dict]:
        """
        Get summary information for all models.
        
        Returns:
            Dictionary with model summaries
        """
        return {
            name: model_info.get_display_info() 
            for name, model_info in self.available_models.items()
        }
    
    def set_recursive_scan(self, recursive: bool):
        """
        Enable or disable recursive directory scanning.
        
        Args:
            recursive: True to enable recursive scanning
        """
        self._scan_recursive = recursive
        logger.info(f"Recursive scanning {'enabled' if recursive else 'disabled'}")
    
    def validate_all_models(self) -> Dict[str, bool]:
        """
        Validate all registered models.
        
        Returns:
            Dictionary mapping model names to validation status
        """
        validation_results = {}
        
        for name, model_info in self.available_models.items():
            try:
                validation_results[name] = model_info.validate_files()
            except Exception as e:
                logger.error(f"Error validating model {name}: {str(e)}")
                validation_results[name] = False
                
        return validation_results
    
    def get_default_model(self) -> Optional[str]:
        """
        Get the default model name (first alphabetically).
        
        Returns:
            Default model name or None if no models available
        """
        if not self.available_models:
            return None
            
        # Return the first model alphabetically for consistency
        return sorted(self.available_models.keys())[0]
    
    def __len__(self) -> int:
        """Return number of available models"""
        return len(self.available_models)
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model name is in registry"""
        return model_name in self.available_models
    
    def __iter__(self):
        """Iterate over model names"""
        return iter(self.available_models.keys())