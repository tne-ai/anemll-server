#!/usr/bin/env python
"""
CoreML Error Classification Wrapper for Model Manager Integration.

This module provides non-intrusive integration of the CoreML error classifier
with the existing model manager, preserving error context and minimizing
performance impact while maintaining backward compatibility.
"""

import time
import logging
import functools
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from contextlib import contextmanager

from coreml_error_classifier import (
    CoreMLErrorClassifier, ClassifiedError, ErrorContext, 
    classify_error, attempt_auto_recovery
)
from error_patterns import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

class CoreMLMonitor:
    """
    Context manager for monitoring CoreML operations and classifying errors.
    
    Provides comprehensive error monitoring for CoreML model loading and inference
    operations with minimal performance impact.
    """
    
    def __init__(self, operation_name: str, model_name: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize CoreML monitor.
        
        Args:
            operation_name: Name of the operation being monitored
            model_name: Name of the model involved (if applicable)
            context: Additional context information
        """
        self.operation_name = operation_name
        self.model_name = model_name
        self.context = context or {}
        self.start_time = None
        self.classifier = None
        
        # Monitoring state
        self.errors_detected: List[ClassifiedError] = []
        self.performance_metrics: Dict[str, float] = {}
        
    def __enter__(self):
        """Enter monitoring context."""
        self.start_time = time.time()
        
        # Get classifier instance (lazy initialization for performance)
        try:
            from coreml_error_classifier import get_classifier
            self.classifier = get_classifier()
        except Exception as e:
            logger.warning(f"Could not initialize CoreML classifier: {str(e)}")
            self.classifier = None
        
        logger.debug(f"Starting CoreML monitoring for: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit monitoring context and handle any errors."""
        duration = time.time() - self.start_time if self.start_time else 0
        self.performance_metrics['total_duration'] = duration
        
        if exc_type is not None and self.classifier:
            # An exception occurred - classify it
            try:
                error_context = self._build_monitoring_context()
                classified_error = self.classifier.classify_error(exc_value, error_context)
                self.errors_detected.append(classified_error)
                
                logger.error(f"CoreML error detected in {self.operation_name}: "
                           f"{classified_error.category.value}/{classified_error.severity.value}")
                
                # Attempt auto-recovery for critical errors
                if (classified_error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] and 
                    classified_error.can_auto_recover):
                    
                    logger.info(f"Attempting auto-recovery for {classified_error.category.value} error")
                    recovery_result = self.classifier.attempt_auto_recovery(
                        classified_error, 
                        self._build_recovery_context()
                    )
                    
                    if recovery_result.get("success"):
                        logger.info("✅ Auto-recovery successful - continuing operation")
                        # Don't re-raise the exception if recovery succeeded
                        return True
                    else:
                        logger.error(f"❌ Auto-recovery failed: {recovery_result.get('error')}")
                
                # Log detailed error information
                self._log_classified_error(classified_error)
                
            except Exception as classification_error:
                logger.error(f"Error during CoreML error classification: {str(classification_error)}")
        
        logger.debug(f"CoreML monitoring completed for {self.operation_name} in {duration:.3f}s")
        
        # Don't suppress the original exception unless recovery succeeded
        return False
    
    def _build_monitoring_context(self) -> Dict[str, Any]:
        """Build context for error classification."""
        context = {
            'operation': self.operation_name,
            'model_name': self.model_name,
            'location': f"coreml_wrapper.py:{self.operation_name}",
            'performance_metrics': self.performance_metrics
        }
        context.update(self.context)
        return context
    
    def _build_recovery_context(self) -> Dict[str, Any]:
        """Build context for recovery operations."""
        return {
            'operation': self.operation_name,
            'model_name': self.model_name,
            'performance_metrics': self.performance_metrics,
            'monitoring_context': self.context
        }
    
    def _log_classified_error(self, classified_error: ClassifiedError):
        """Log detailed information about a classified error."""
        logger.error("=" * 80)
        logger.error("COREML ERROR CLASSIFICATION REPORT")
        logger.error("=" * 80)
        logger.error(f"Operation: {self.operation_name}")
        logger.error(f"Model: {self.model_name or 'Unknown'}")
        logger.error(f"Category: {classified_error.category.value}")
        logger.error(f"Severity: {classified_error.severity.value}")
        logger.error(f"Confidence: {classified_error.confidence:.2f}")
        logger.error(f"Error: {classified_error.original_error}")
        logger.error(f"Description: {classified_error.description}")
        
        if classified_error.potential_causes:
            logger.error("Potential Causes:")
            for cause in classified_error.potential_causes:
                logger.error(f"  - {cause}")
        
        if classified_error.recovery_steps:
            logger.error("Recovery Steps:")
            for step in classified_error.recovery_steps:
                logger.error(f"  - {step}")
        
        logger.error("=" * 80)

def monitor_coreml_operation(operation_name: str, model_name: Optional[str] = None):
    """
    Decorator for monitoring CoreML operations with error classification.
    
    Args:
        operation_name: Name of the operation being monitored
        model_name: Name of the model involved (if applicable)
    
    Returns:
        Decorated function with CoreML monitoring
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract model name from arguments if not provided
            actual_model_name = model_name
            if not actual_model_name:
                # Try to extract model name from common argument patterns
                if args and hasattr(args[0], 'name'):
                    actual_model_name = args[0].name
                elif 'model_name' in kwargs:
                    actual_model_name = kwargs['model_name']
                elif 'model_info' in kwargs and hasattr(kwargs['model_info'], 'name'):
                    actual_model_name = kwargs['model_info'].name
            
            # Build context from function arguments
            context = {
                'function': func.__name__,
                'module': func.__module__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            with CoreMLMonitor(operation_name, actual_model_name, context):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

class ModelManagerWrapper:
    """
    Wrapper for ModelManager that adds CoreML error classification without
    modifying the original implementation.
    """
    
    def __init__(self, original_manager):
        """
        Initialize wrapper around existing ModelManager.
        
        Args:
            original_manager: Original ModelManager instance to wrap
        """
        self.original_manager = original_manager
        self.classifier = None
        self.monitoring_enabled = True
        self.error_history: List[ClassifiedError] = []
        
        # Wrap critical methods
        self._wrap_methods()
        
    def _wrap_methods(self):
        """Wrap critical ModelManager methods with error monitoring."""
        # Wrap _load_model method (highest priority crash location)
        original_load_model = self.original_manager._load_model
        
        @monitor_coreml_operation("model_loading")
        def wrapped_load_model(model_info):
            return original_load_model(model_info)
        
        self.original_manager._load_model = wrapped_load_model
        
        # Wrap get_model method (entry point for model operations)
        original_get_model = self.original_manager.get_model
        
        async def wrapped_get_model(model_name: str):
            with CoreMLMonitor("model_retrieval", model_name):
                return await original_get_model(model_name)
        
        self.original_manager.get_model = wrapped_get_model
    
    def enable_monitoring(self):
        """Enable CoreML error monitoring."""
        self.monitoring_enabled = True
        logger.info("CoreML error monitoring enabled")
    
    def disable_monitoring(self):
        """Disable CoreML error monitoring."""
        self.monitoring_enabled = False
        logger.info("CoreML error monitoring disabled")
    
    def get_error_history(self) -> List[ClassifiedError]:
        """Get history of classified errors."""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("CoreML error history cleared")
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original manager."""
        return getattr(self.original_manager, name)

class CoreMLErrorHandler:
    """
    Enhanced error handler for CoreML operations with classification and recovery.
    """
    
    def __init__(self):
        """Initialize CoreML error handler."""
        self.classifier = None
        self.auto_recovery_enabled = True
        self.error_counts: Dict[str, int] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
    def handle_error(self, error: Exception, operation: str, model_name: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a CoreML error with classification and potential recovery.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            model_name: Name of the model involved
            context: Additional context information
            
        Returns:
            Dictionary with handling results
        """
        try:
            # Get classifier instance
            if not self.classifier:
                from coreml_error_classifier import get_classifier
                self.classifier = get_classifier()
            
            # Build error context
            error_context = context or {}
            error_context.update({
                'operation': operation,
                'model_name': model_name,
                'handler': 'CoreMLErrorHandler'
            })
            
            # Classify the error
            classified_error = self.classifier.classify_error(error, error_context)
            
            # Track error counts
            error_key = f"{classified_error.category.value}:{classified_error.error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            result = {
                'classified': True,
                'category': classified_error.category.value,
                'severity': classified_error.severity.value,
                'confidence': classified_error.confidence,
                'can_auto_recover': classified_error.can_auto_recover,
                'recovery_attempted': False,
                'recovery_successful': False
            }
            
            # Attempt auto-recovery if enabled and appropriate
            if (self.auto_recovery_enabled and 
                classified_error.can_auto_recover and 
                classified_error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]):
                
                recovery_result = self.classifier.attempt_auto_recovery(classified_error, error_context)
                result['recovery_attempted'] = recovery_result['attempted']
                result['recovery_successful'] = recovery_result.get('success', False)
                
                # Track recovery history
                self.recovery_history.append({
                    'timestamp': time.time(),
                    'error_category': classified_error.category.value,
                    'operation': operation,
                    'model_name': model_name,
                    'success': recovery_result.get('success', False),
                    'duration': recovery_result.get('duration', 0)
                })
            
            # Log comprehensive error information
            self._log_error_handling(classified_error, result)
            
            return result
            
        except Exception as handling_error:
            logger.error(f"Error in CoreML error handler: {str(handling_error)}")
            return {
                'classified': False,
                'error': str(handling_error),
                'recovery_attempted': False,
                'recovery_successful': False
            }
    
    def _log_error_handling(self, classified_error: ClassifiedError, result: Dict[str, Any]):
        """Log comprehensive error handling information."""
        logger.info(f"CoreML Error Handled: {classified_error.category.value}/{classified_error.severity.value}")
        logger.info(f"Confidence: {classified_error.confidence:.2f}, "
                   f"Auto-recovery: {'✅' if result['recovery_successful'] else '❌' if result['recovery_attempted'] else '⏸️'}")
        
        if classified_error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL CoreML Error: {classified_error.description}")
        elif classified_error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH Priority CoreML Error: {classified_error.description}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r['success'])
        
        return {
            'total_errors_handled': sum(self.error_counts.values()),
            'error_types': self.error_counts.copy(),
            'total_recovery_attempts': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate': (successful_recoveries / total_recoveries) if total_recoveries > 0 else 0,
            'auto_recovery_enabled': self.auto_recovery_enabled
        }

# Global error handler instance
_global_error_handler = None

def get_error_handler() -> CoreMLErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = CoreMLErrorHandler()
    return _global_error_handler

def handle_coreml_error(error: Exception, operation: str, model_name: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to handle CoreML errors using the global handler."""
    return get_error_handler().handle_error(error, operation, model_name, context)

@contextmanager
def coreml_error_context(operation: str, model_name: Optional[str] = None, 
                        auto_recover: bool = True):
    """
    Context manager for automatic CoreML error handling.
    
    Args:
        operation: Name of the operation being performed
        model_name: Name of the model involved
        auto_recover: Whether to attempt automatic recovery
        
    Example:
        with coreml_error_context("model_loading", "my_model"):
            # CoreML operations that might fail
            model = load_coreml_model(path)
    """
    try:
        yield
    except Exception as e:
        context = {'auto_recover': auto_recover}
        result = handle_coreml_error(e, operation, model_name, context)
        
        # Re-raise if recovery was not successful
        if not result.get('recovery_successful', False):
            raise