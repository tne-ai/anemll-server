#!/usr/bin/env python
"""
Model information and metadata structures for dynamic model loading.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model metadata and file paths extracted from meta.yaml"""
    name: str
    path: Path
    embed_path: str
    ffn_path: str  
    lmhead_path: str
    tokenizer_path: str
    context_length: int
    batch_size: int
    metadata: Dict[str, Any]
    model_prefix: str
    num_chunks: int
    lut_ffn: str
    lut_lmhead: str
    
    @classmethod
    def from_meta_yaml(cls, model_dir: Path) -> Optional['ModelInfo']:
        """
        Create ModelInfo from meta.yaml file in model directory.
        
        Args:
            model_dir: Path to directory containing meta.yaml
            
        Returns:
            ModelInfo instance or None if invalid
        """
        meta_yaml_path = model_dir / "meta.yaml"
        
        if not meta_yaml_path.exists():
            logger.warning(f"No meta.yaml found in {model_dir}")
            return None
            
        try:
            with open(meta_yaml_path, 'r') as f:
                meta_yaml = yaml.safe_load(f)
                
            # Extract parameters from meta.yaml
            params = meta_yaml['model_info']['parameters']
            
            # Get model configuration
            model_prefix = params.get('model_prefix', 'llama')
            context_length = int(params['context_length'])
            batch_size = int(params['batch_size'])
            num_chunks = int(params['num_chunks'])
            lut_ffn = params['lut_ffn']
            lut_lmhead = params['lut_lmhead']
            
            # Build file paths based on meta.yaml configuration
            embed_path = str(model_dir / f"{model_prefix}_embeddings.mlmodelc")
            
            # Handle chunked FFN models
            lut_suffix = f"_lut{lut_ffn}" if lut_ffn != 'none' else ''
            ffn_path = str(model_dir / f"{model_prefix}_FFN_PF{lut_suffix}_chunk_01of{num_chunks:02d}.mlmodelc")
            
            # LM head path
            lmhead_suffix = f"_lut{lut_lmhead}" if lut_lmhead != 'none' else ''
            lmhead_path = str(model_dir / f"{model_prefix}_lm_head{lmhead_suffix}.mlmodelc")
            
            tokenizer_path = str(model_dir)
            
            # Use directory name as model name, or extract from meta.yaml if available
            model_name = model_dir.name
            
            # Validate that required files exist
            required_files = [embed_path, ffn_path, lmhead_path]
            missing_files = [f for f in required_files if not Path(f).exists()]
            
            if missing_files:
                logger.warning(f"Model {model_name} missing files: {missing_files}")
                return None
                
            return cls(
                name=model_name,
                path=model_dir,
                embed_path=embed_path,
                ffn_path=ffn_path,
                lmhead_path=lmhead_path,
                tokenizer_path=tokenizer_path,
                context_length=context_length,
                batch_size=batch_size,
                metadata=params,
                model_prefix=model_prefix,
                num_chunks=num_chunks,
                lut_ffn=lut_ffn,
                lut_lmhead=lut_lmhead
            )
            
        except Exception as e:
            logger.error(f"Error parsing meta.yaml in {model_dir}: {str(e)}")
            return None
    
    def validate_files(self) -> bool:
        """
        Validate that all required model files exist.
        
        Returns:
            True if all files exist, False otherwise
        """
        required_files = [
            self.embed_path,
            self.ffn_path, 
            self.lmhead_path
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Missing required file: {file_path}")
                return False
                
        # Check tokenizer directory
        if not Path(self.tokenizer_path).exists():
            logger.error(f"Missing tokenizer directory: {self.tokenizer_path}")
            return False
            
        return True
    
    def get_display_info(self) -> Dict[str, Any]:
        """
        Get model information suitable for API responses.
        
        Returns:
            Dictionary with model display information
        """
        return {
            "id": self.name,
            "object": "model",
            "owned_by": "anemll",
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "model_prefix": self.model_prefix,
            "path": str(self.path)
        }

@dataclass 
class LoadedModel:
    """Container for loaded model components and runtime state"""
    info: ModelInfo
    embed_model: Any
    ffn_models: Any 
    lmhead_model: Any
    tokenizer: Any
    metadata: Dict[str, Any]
    state: Any
    causal_mask: Any
    load_time: float
    last_used: float
    
    def update_last_used(self, timestamp: float):
        """Update the last used timestamp for LRU tracking"""
        self.last_used = timestamp
        
    def cleanup(self):
        """Clean up model resources (if needed for memory management)"""
        # This could be extended to explicitly free GPU/ANE resources
        # For now, we rely on Python's garbage collection
        pass