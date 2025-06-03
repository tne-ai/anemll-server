#!/usr/bin/env python
"""
Test script for dynamic model loading functionality.
"""

import os
import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from model_registry import ModelRegistry
from model_manager import ModelManager
from config import get_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dynamic_loading():
    """Test the dynamic model loading system."""
    
    logger.info("=== Testing Dynamic Model Loading ===")
    
    # Test 1: Configuration
    logger.info("\n1. Testing Configuration")
    config = get_config()
    logger.info(f"Model Directory: {config.model_dir}")
    logger.info(f"Cache Size: {config.cache_size}")
    logger.info(f"Recursive Scan: {config.scan_recursive}")
    
    # Test 2: Model Registry
    logger.info("\n2. Testing Model Registry")
    try:
        registry = ModelRegistry(config.model_dir)
        models = registry.list_models()
        logger.info(f"Found {len(models)} models: {models}")
        
        if models:
            # Test model info retrieval
            first_model = models[0]
            model_info = registry.get_model_info(first_model)
            logger.info(f"Model '{first_model}' info:")
            logger.info(f"  Path: {model_info.path}")
            logger.info(f"  Context Length: {model_info.context_length}")
            logger.info(f"  Batch Size: {model_info.batch_size}")
        else:
            logger.warning("No models found - create a test model directory")
            return False
            
    except Exception as e:
        logger.error(f"Model registry test failed: {str(e)}")
        return False
    
    # Test 3: Model Manager
    logger.info("\n3. Testing Model Manager")
    try:
        manager = ModelManager(registry, cache_size=2)
        
        # Test cache stats
        stats = manager.get_cache_stats()
        logger.info(f"Initial cache stats: {stats}")
        
        if models:
            # Test model loading
            logger.info(f"Loading model: {first_model}")
            loaded_model = await manager.get_model(first_model)
            logger.info(f"Successfully loaded model: {first_model}")
            logger.info(f"Model metadata: {loaded_model.metadata}")
            
            # Test cache hit
            logger.info("Testing cache hit...")
            loaded_model_2 = await manager.get_model(first_model)
            logger.info("Cache hit successful")
            
            # Test cache stats after loading
            stats = manager.get_cache_stats()
            logger.info(f"Cache stats after loading: {stats}")
            
    except Exception as e:
        logger.error(f"Model manager test failed: {str(e)}")
        return False
    
    logger.info("\n=== All Tests Passed! ===")
    return True

def test_config_validation():
    """Test configuration validation."""
    logger.info("\n4. Testing Configuration Validation")
    
    # Test environment variable parsing
    original_cache_size = os.environ.get("MODEL_CACHE_SIZE")
    
    try:
        # Test valid cache size
        os.environ["MODEL_CACHE_SIZE"] = "5"
        from config import reload_config
        config = reload_config()
        assert config.cache_size == 5, f"Expected cache size 5, got {config.cache_size}"
        
        # Test invalid cache size (should default)
        os.environ["MODEL_CACHE_SIZE"] = "invalid"
        config = reload_config()
        assert config.cache_size == 3, f"Expected default cache size 3, got {config.cache_size}"
        
        logger.info("Configuration validation passed")
        
    finally:
        # Restore original value
        if original_cache_size:
            os.environ["MODEL_CACHE_SIZE"] = original_cache_size
        elif "MODEL_CACHE_SIZE" in os.environ:
            del os.environ["MODEL_CACHE_SIZE"]
        
        # Reload config to restore original state
        reload_config()

def main():
    """Run all tests."""
    logger.info("Starting dynamic model loading tests...")
    
    try:
        # Test configuration first
        test_config_validation()
        
        # Test async components
        success = asyncio.run(test_dynamic_loading())
        
        if success:
            logger.info("\n🎉 All tests completed successfully!")
            logger.info("Dynamic model loading is ready for use.")
        else:
            logger.error("\n❌ Some tests failed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()