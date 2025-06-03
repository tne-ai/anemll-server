#!/usr/bin/env python
"""
Model manager with LRU caching for dynamic model loading.
"""

import time
import threading
import argparse
from collections import OrderedDict
from typing import Dict, Optional, Any, List
import logging
from pathlib import Path

from model_info import ModelInfo, LoadedModel
from model_registry import ModelRegistry

# Import from chat_full.py for model loading
from chat_full import (
    load_models, 
    initialize_tokenizer, 
    create_unified_state, 
    make_causal_mask
)
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading, caching, and lifecycle with LRU eviction"""
    
    def __init__(self, registry: ModelRegistry, cache_size: int = 3):
        """
        Initialize model manager.
        
        Args:
            registry: ModelRegistry instance for model discovery
            cache_size: Maximum number of models to keep in cache
        """
        self.registry = registry
        self.cache_size = cache_size
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.lru_tracker = OrderedDict()  # model_name -> timestamp
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Statistics tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'models_loaded': 0,
            'models_evicted': 0,
            'load_failures': 0
        }
        
        logger.info(f"ModelManager initialized with cache size: {cache_size}")
    
    async def get_model(self, model_name: str) -> LoadedModel:
        """
        Get model (load if needed, manage LRU cache).
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            LoadedModel instance
            
        Raises:
            ValueError: If model not found in registry
            RuntimeError: If model loading fails
        """
        with self.lock:
            current_time = time.time()
            
            # Check if model is already loaded
            if model_name in self.loaded_models:
                self.stats['cache_hits'] += 1
                loaded_model = self.loaded_models[model_name]
                loaded_model.update_last_used(current_time)
                self._update_lru_tracker(model_name, current_time)
                logger.info(f"Cache hit for model: {model_name}")
                return loaded_model
            
            # Cache miss - need to load the model
            self.stats['cache_misses'] += 1
            logger.info(f"Cache miss for model: {model_name}")
            
            # Check if model exists in registry
            model_info = self.registry.get_model_info(model_name)
            if not model_info:
                raise ValueError(f"Model '{model_name}' not found in registry. Available models: {self.registry.list_models()}")
            
            # Make room in cache if needed
            self._ensure_cache_space()
            
            # Load the model
            try:
                loaded_model = self._load_model(model_info)
                loaded_model.update_last_used(current_time)
                
                # Add to cache
                self.loaded_models[model_name] = loaded_model
                self._update_lru_tracker(model_name, current_time)
                
                self.stats['models_loaded'] += 1
                logger.info(f"Successfully loaded model: {model_name}")
                
                return loaded_model
                
            except Exception as e:
                self.stats['load_failures'] += 1
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _load_model(self, model_info: ModelInfo) -> LoadedModel:
        """
        Load model components from disk.
        
        Args:
            model_info: Model information and paths
            
        Returns:
            LoadedModel instance
        """
        start_time = time.time()
        logger.info(f"Loading model: {model_info.name}")
        
        # Create a chat_args object that matches what load_models expects
        chat_args = argparse.Namespace()
        chat_args.embed = model_info.embed_path
        chat_args.ffn = model_info.ffn_path
        chat_args.lmhead = model_info.lmhead_path
        chat_args.tokenizer = model_info.tokenizer_path
        chat_args.context_length = model_info.context_length
        chat_args.batch_size = model_info.batch_size
        chat_args.d = str(model_info.path)
        chat_args.meta = str(model_info.path / "meta.yaml")
        
        try:
            # Load models using chat_full.py functions
            embed_model, ffn_models, lmhead_model, metadata = load_models(chat_args, {})
            
            # Load tokenizer
            tokenizer = initialize_tokenizer(model_info.tokenizer_path)
            if tokenizer is None:
                raise RuntimeError("Failed to initialize tokenizer")
            
            # Create unified state
            state = create_unified_state(ffn_models, metadata['context_length'])
            
            # Initialize causal mask
            causal_mask = make_causal_mask(metadata['context_length'], 0)
            causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
            
            load_time = time.time() - start_time
            
            logger.info(f"Model {model_info.name} loaded successfully in {load_time:.2f}s")
            logger.info(f"Context length: {metadata['context_length']}, Batch size: {metadata['batch_size']}")
            
            return LoadedModel(
                info=model_info,
                embed_model=embed_model,
                ffn_models=ffn_models,
                lmhead_model=lmhead_model,
                tokenizer=tokenizer,
                metadata=metadata,
                state=state,
                causal_mask=causal_mask,
                load_time=load_time,
                last_used=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error loading model {model_info.name}: {str(e)}")
            
            # Provide more detailed error information
            logger.error(f"Expected files:")
            logger.error(f"  Embeddings: {model_info.embed_path}")
            logger.error(f"  LM Head: {model_info.lmhead_path}")
            logger.error(f"  FFN: {model_info.ffn_path}")
            
            # Check if files exist
            logger.error(f"File existence check:")
            logger.error(f"  Embeddings exists: {Path(model_info.embed_path).exists()}")
            logger.error(f"  LM Head exists: {Path(model_info.lmhead_path).exists()}")
            logger.error(f"  FFN exists: {Path(model_info.ffn_path).exists()}")
            
            raise
    
    def _ensure_cache_space(self):
        """Make room in cache if needed by evicting LRU model"""
        while len(self.loaded_models) >= self.cache_size:
            self._evict_lru_model()
    
    def _evict_lru_model(self):
        """Remove least recently used model from cache"""
        if not self.lru_tracker:
            return
            
        # Get the least recently used model
        lru_model_name = next(iter(self.lru_tracker))
        
        logger.info(f"Evicting LRU model: {lru_model_name}")
        
        # Remove from cache
        if lru_model_name in self.loaded_models:
            loaded_model = self.loaded_models[lru_model_name]
            loaded_model.cleanup()  # Clean up resources
            del self.loaded_models[lru_model_name]
            
        # Remove from LRU tracker
        del self.lru_tracker[lru_model_name]
        
        self.stats['models_evicted'] += 1
        logger.info(f"Successfully evicted model: {lru_model_name}")
    
    def _update_lru_tracker(self, model_name: str, timestamp: float):
        """Update LRU tracker for a model"""
        # Remove if exists (to update position)
        if model_name in self.lru_tracker:
            del self.lru_tracker[model_name]
        
        # Add to end (most recently used)
        self.lru_tracker[model_name] = timestamp
    
    def unload_model(self, model_name: str) -> bool:
        """
        Explicitly unload a model.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not loaded
        """
        with self.lock:
            if model_name in self.loaded_models:
                loaded_model = self.loaded_models[model_name]
                loaded_model.cleanup()
                del self.loaded_models[model_name]
                
                if model_name in self.lru_tracker:
                    del self.lru_tracker[model_name]
                
                logger.info(f"Manually unloaded model: {model_name}")
                return True
            
            return False
    
    def clear_cache(self):
        """Clear all models from cache"""
        with self.lock:
            for model_name, loaded_model in self.loaded_models.items():
                loaded_model.cleanup()
                
            self.loaded_models.clear()
            self.lru_tracker.clear()
            
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return cache hit/miss statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': self.cache_size,
                'loaded_models_count': len(self.loaded_models),
                'available_models_count': len(self.registry),
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'hit_rate_percent': round(hit_rate, 2),
                'models_loaded': self.stats['models_loaded'],
                'models_evicted': self.stats['models_evicted'],
                'load_failures': self.stats['load_failures'],
                'loaded_models': list(self.loaded_models.keys()),
                'lru_order': list(self.lru_tracker.keys())
            }
    
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model names.
        
        Returns:
            List of loaded model names
        """
        with self.lock:
            return list(self.loaded_models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is loaded
        """
        with self.lock:
            return model_name in self.loaded_models
    
    def resize_cache(self, new_size: int):
        """
        Change cache size, evicting models if necessary.
        
        Args:
            new_size: New cache size
        """
        with self.lock:
            if new_size < 1:
                raise ValueError("Cache size must be at least 1")
                
            old_size = self.cache_size
            self.cache_size = new_size
            
            # Evict models if new size is smaller
            while len(self.loaded_models) > new_size:
                self._evict_lru_model()
                
            logger.info(f"Cache resized from {old_size} to {new_size}")