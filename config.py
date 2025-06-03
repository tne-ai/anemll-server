#!/usr/bin/env python
"""
Configuration management for dynamic model loading.
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration settings for model management"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        
        # Model discovery settings
        self.model_dir = self._get_model_dir()
        self.scan_recursive = self._get_bool_env("MODEL_SCAN_RECURSIVE", True)
        
        # Cache settings
        self.cache_size = self._get_int_env("MODEL_CACHE_SIZE", 3)
        self.cache_ttl = self._get_int_env("MODEL_CACHE_TTL", 3600)  # seconds
        
        # Model loading settings
        self.load_timeout = self._get_int_env("MODEL_LOAD_TIMEOUT", 60)  # seconds
        self.preload_models = self._get_preload_models()
        
        # Memory management
        self.memory_limit = os.getenv("MODEL_CACHE_MEMORY_LIMIT", "8GB")
        
        # Server settings (existing)
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8400))
        self.allow_truncation = False  # Will be set by command line args
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        self._validate_config()
        self._log_config()
    
    def _get_model_dir(self) -> str:
        """Get model directory from environment with validation"""
        model_dir = os.getenv("MODEL_DIR", "./models")
        
        # Convert to absolute path
        model_path = Path(model_dir).resolve()
        
        if not model_path.exists():
            logger.warning(f"Model directory does not exist: {model_path}")
            # Don't raise error here - let ModelRegistry handle it
            
        return str(model_path)
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable"""
        try:
            value = os.getenv(key)
            if value is None:
                return default
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_preload_models(self) -> List[str]:
        """Get list of models to preload from environment"""
        preload_str = os.getenv("MODEL_PRELOAD", "")
        if not preload_str.strip():
            return []
        
        # Split by comma and clean up whitespace
        models = [model.strip() for model in preload_str.split(",")]
        return [model for model in models if model]  # Remove empty strings
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.cache_size < 1:
            logger.warning(f"Invalid cache size: {self.cache_size}, setting to 1")
            self.cache_size = 1
        
        if self.cache_size > 10:
            logger.warning(f"Large cache size: {self.cache_size}, this may use significant memory")
        
        if self.load_timeout < 10:
            logger.warning(f"Short load timeout: {self.load_timeout}s, model loading may fail")
        
        if self.port < 1024 or self.port > 65535:
            logger.error(f"Invalid port: {self.port}")
            raise ValueError(f"Port must be between 1024 and 65535, got: {self.port}")
    
    def _log_config(self):
        """Log current configuration"""
        logger.info("Model Configuration:")
        logger.info(f"  Model Directory: {self.model_dir}")
        logger.info(f"  Cache Size: {self.cache_size}")
        logger.info(f"  Recursive Scan: {self.scan_recursive}")
        logger.info(f"  Load Timeout: {self.load_timeout}s")
        logger.info(f"  Server: {self.host}:{self.port}")
        
        if self.preload_models:
            logger.info(f"  Preload Models: {self.preload_models}")
        else:
            logger.info("  No models configured for preloading")
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'truncate'):
            self.allow_truncation = args.truncate
            logger.info(f"Truncation: {'Enabled' if self.allow_truncation else 'Disabled'}")
    
    def get_display_config(self) -> dict:
        """Get configuration suitable for API display"""
        return {
            "model_directory": self.model_dir,
            "cache_size": self.cache_size,
            "recursive_scan": self.scan_recursive,
            "load_timeout": self.load_timeout,
            "preload_models": self.preload_models,
            "server_host": self.host,
            "server_port": self.port,
            "truncation_enabled": self.allow_truncation
        }

# Global configuration instance
config = ModelConfig()

def get_config() -> ModelConfig:
    """Get the global configuration instance"""
    return config

def reload_config() -> ModelConfig:
    """Reload configuration from environment"""
    global config
    config = ModelConfig()
    return config