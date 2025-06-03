#!/usr/bin/env python
"""
CoreML Error Pattern Classifier - Main classification engine.

This module provides the core error classification functionality for CoreML/ANE-specific
errors that lead to segmentation faults in the Anemll server.

Based on the segfault analysis in SEGFAULT_TESTING.md, it focuses on:
1. CoreML/ANE Driver Issues (High Priority)
2. Memory Corruption During Model Loading (High Priority)  
3. Thread Safety Issues (Medium Priority)
4. Resource Exhaustion (Medium Priority)
5. Model File Corruption (Lower Priority)
"""

import re
import time
import logging
import traceback
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from error_patterns import (
    ErrorPatternDatabase, ErrorCategory, ErrorSeverity, ErrorPattern
)
from recovery_strategies import (
    RecoveryStrategyDatabase, RecoveryStrategy, RecoveryExecutor
)

logger = logging.getLogger(__name__)

@dataclass
class ErrorContext:
    """Context information for error classification."""
    timestamp: datetime
    location: str                    # Function/file where error occurred
    model_name: Optional[str] = None
    operation: Optional[str] = None  # What operation was being performed
    system_state: Optional[Dict[str, Any]] = None
    thread_info: Optional[Dict[str, Any]] = None
    memory_info: Optional[Dict[str, Any]] = None
    stacktrace: Optional[str] = None
    previous_errors: Optional[List[str]] = None

@dataclass
class ClassifiedError:
    """Result of error classification."""
    original_error: str
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float
    matched_patterns: List[str]
    context: ErrorContext
    description: str
    potential_causes: List[str]
    recovery_strategy: Optional[str] = None
    recovery_steps: Optional[List[str]] = None
    can_auto_recover: bool = False
    estimated_recovery_time: Optional[float] = None
    classification_time: float = 0.0

class CoreMLErrorClassifier:
    """
    Main CoreML error classification engine.
    
    Classifies CoreML/ANE-related errors based on known patterns from segfault analysis
    and provides recovery recommendations.
    """
    
    def __init__(self):
        """Initialize the CoreML error classifier."""
        self.pattern_db = ErrorPatternDatabase()
        self.strategy_db = RecoveryStrategyDatabase()
        self.recovery_executor = RecoveryExecutor(self.strategy_db)
        
        # Classification statistics
        self.classification_stats = {
            'total_classifications': 0,
            'by_category': {category: 0 for category in ErrorCategory},
            'by_severity': {severity: 0 for severity in ErrorSeverity},
            'auto_recoveries_attempted': 0,
            'auto_recoveries_successful': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("CoreML Error Classifier initialized")
        logger.info(f"Loaded {len(self.pattern_db.get_all_patterns())} error patterns")
        logger.info(f"Loaded {len(self.strategy_db.strategies)} recovery strategies")
    
    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ClassifiedError:
        """
        Classify a CoreML-related error and provide recovery recommendations.
        
        Args:
            error: The exception or error to classify
            context: Additional context information
            
        Returns:
            ClassifiedError with classification results and recovery recommendations
        """
        start_time = time.time()
        
        with self.lock:
            self.classification_stats['total_classifications'] += 1
        
        try:
            # Prepare error context
            error_context = self._build_error_context(error, context or {})
            
            # Extract error information
            error_str = str(error)
            error_type = type(error).__name__
            stacktrace = self._get_stacktrace(error)
            
            # Match against known patterns
            matched_patterns, best_match = self._match_error_patterns(error_str, error_type, stacktrace, error_context)
            
            if best_match:
                category = best_match.category
                severity = best_match.severity
                confidence = self._calculate_confidence(error_str, best_match, error_context)
                description = best_match.description
                potential_causes = best_match.triggers
                
                # Get recovery strategy
                recovery_strategy = self._get_recovery_strategy(category, severity)
                recovery_steps = None
                can_auto_recover = False
                estimated_recovery_time = None
                
                if recovery_strategy:
                    recovery_steps = [step.description for step in recovery_strategy.steps]
                    can_auto_recover = recovery_strategy.auto_recovery_enabled
                    estimated_recovery_time = recovery_strategy.estimated_time
                
            else:
                # Unknown error pattern
                category = ErrorCategory.UNKNOWN
                severity = ErrorSeverity.MEDIUM
                confidence = 0.0
                description = "Unknown error pattern - manual investigation required"
                potential_causes = ["Unknown cause", "Pattern not in database"]
                recovery_strategy = None
                recovery_steps = None
                can_auto_recover = False
                estimated_recovery_time = None
            
            classification_time = time.time() - start_time
            
            classified_error = ClassifiedError(
                original_error=error_str,
                error_type=error_type,
                category=category,
                severity=severity,
                confidence=confidence,
                matched_patterns=[p.name for p in matched_patterns],
                context=error_context,
                description=description,
                potential_causes=potential_causes,
                recovery_strategy=recovery_strategy.name if recovery_strategy else None,
                recovery_steps=recovery_steps,
                can_auto_recover=can_auto_recover,
                estimated_recovery_time=estimated_recovery_time,
                classification_time=classification_time
            )
            
            # Update statistics
            with self.lock:
                self.classification_stats['by_category'][category] += 1
                self.classification_stats['by_severity'][severity] += 1
            
            logger.info(f"Error classified: {category.value}/{severity.value} (confidence: {confidence:.2f})")
            logger.debug(f"Classification details: {classified_error}")
            
            return classified_error
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            traceback.print_exc()
            
            # Return basic classification for the classification error itself
            return ClassifiedError(
                original_error=str(error),
                error_type=type(error).__name__,
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.LOW,
                confidence=0.0,
                matched_patterns=[],
                context=ErrorContext(timestamp=datetime.now(), location="classification_error"),
                description="Error occurred during classification",
                potential_causes=["Classifier malfunction"],
                classification_time=time.time() - start_time
            )
    
    def _build_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Build comprehensive error context."""
        timestamp = datetime.now()
        
        # Extract location information
        location = context.get('location', 'unknown')
        if hasattr(error, '__traceback__') and error.__traceback__:
            tb = error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            location = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        
        # Get system state
        system_state = self._get_system_state()
        
        # Get thread information
        thread_info = {
            'thread_count': threading.active_count(),
            'current_thread': threading.current_thread().name,
            'is_main_thread': threading.current_thread() is threading.main_thread()
        }
        
        # Get memory information
        memory_info = self._get_memory_info()
        
        # Get stack trace
        stacktrace = self._get_stacktrace(error)
        
        return ErrorContext(
            timestamp=timestamp,
            location=location,
            model_name=context.get('model_name'),
            operation=context.get('operation'),
            system_state=system_state,
            thread_info=thread_info,
            memory_info=memory_info,
            stacktrace=stacktrace,
            previous_errors=context.get('previous_errors', [])
        )
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Could not get system state: {str(e)}")
            return {}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            vm = psutil.virtual_memory()
            return {
                'total': vm.total,
                'available': vm.available,
                'percent': vm.percent,
                'used': vm.used,
                'free': vm.free
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {str(e)}")
            return {}
    
    def _get_stacktrace(self, error: Exception) -> Optional[str]:
        """Get stack trace from exception."""
        try:
            if hasattr(error, '__traceback__') and error.__traceback__:
                return ''.join(traceback.format_tb(error.__traceback__))
            return None
        except Exception:
            return None
    
    def _match_error_patterns(self, error_str: str, error_type: str, stacktrace: Optional[str], 
                            context: ErrorContext) -> Tuple[List[ErrorPattern], Optional[ErrorPattern]]:
        """Match error against known patterns."""
        matched_patterns = []
        best_match = None
        best_score = 0.0
        
        all_patterns = self.pattern_db.get_all_patterns()
        
        for pattern_name, pattern in all_patterns.items():
            score = self._calculate_pattern_match_score(
                pattern, error_str, error_type, stacktrace, context
            )
            
            if score > 0:
                matched_patterns.append(pattern)
                if score > best_score:
                    best_score = score
                    best_match = pattern
        
        # Sort patterns by match strength
        matched_patterns.sort(key=lambda p: self._calculate_pattern_match_score(
            p, error_str, error_type, stacktrace, context
        ), reverse=True)
        
        return matched_patterns, best_match
    
    def _calculate_pattern_match_score(self, pattern: ErrorPattern, error_str: str, 
                                     error_type: str, stacktrace: Optional[str], 
                                     context: ErrorContext) -> float:
        """Calculate how well a pattern matches the error."""
        score = 0.0
        
        # Check text signatures
        for signature in pattern.signatures:
            if signature.lower() in error_str.lower():
                score += 1.0
        
        # Check regex patterns
        for regex_pattern in pattern.regex_patterns:
            if regex_pattern.search(error_str):
                score += 1.5  # Regex matches are weighted higher
        
        # Check error type
        if error_type in error_str:
            score += 0.5
        
        # Check stack trace
        if stacktrace:
            for location in pattern.typical_locations:
                if location in stacktrace:
                    score += 1.0
        
        # Check context location
        if context.location:
            for location in pattern.typical_locations:
                if location in context.location:
                    score += 0.5
        
        # Normalize score based on number of signatures
        if len(pattern.signatures) > 0:
            score = score / len(pattern.signatures)
        
        return score
    
    def _calculate_confidence(self, error_str: str, pattern: ErrorPattern, context: ErrorContext) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.0
        
        # Base confidence from pattern match
        signature_matches = sum(1 for sig in pattern.signatures if sig.lower() in error_str.lower())
        regex_matches = sum(1 for regex in pattern.regex_patterns if regex.search(error_str))
        
        total_patterns = len(pattern.signatures) + len(pattern.regex_patterns)
        if total_patterns > 0:
            confidence = (signature_matches + regex_matches) / total_patterns
        
        # Boost confidence based on location match
        if context.location:
            for location in pattern.typical_locations:
                if location in context.location:
                    confidence += 0.2
        
        # Boost confidence for critical patterns with exact matches
        if pattern.severity == ErrorSeverity.CRITICAL and signature_matches > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_recovery_strategy(self, category: ErrorCategory, severity: ErrorSeverity) -> Optional[RecoveryStrategy]:
        """Get the most appropriate recovery strategy for the error category and severity."""
        strategies = self.strategy_db.get_strategies_for_category(category)
        
        if not strategies:
            return None
        
        # Prefer strategies that match the severity level
        matching_severity = [s for s in strategies if s.severity == severity]
        if matching_severity:
            return matching_severity[0]
        
        # Fall back to any strategy for the category
        return strategies[0]
    
    def attempt_auto_recovery(self, classified_error: ClassifiedError, 
                            recovery_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Attempt automatic recovery for the classified error.
        
        Args:
            classified_error: The classified error to recover from
            recovery_context: Additional context for recovery
            
        Returns:
            Dictionary with recovery results
        """
        if not classified_error.can_auto_recover or not classified_error.recovery_strategy:
            return {
                "attempted": False,
                "reason": "Auto-recovery not enabled for this error type"
            }
        
        with self.lock:
            self.classification_stats['auto_recoveries_attempted'] += 1
        
        logger.info(f"Attempting auto-recovery for {classified_error.category.value} error")
        
        try:
            recovery_result = self.recovery_executor.execute_strategy(
                classified_error.recovery_strategy,
                recovery_context or {}
            )
            
            if recovery_result["success"]:
                with self.lock:
                    self.classification_stats['auto_recoveries_successful'] += 1
                
                logger.info("✅ Auto-recovery completed successfully")
            else:
                logger.error(f"❌ Auto-recovery failed: {recovery_result.get('error')}")
            
            return {
                "attempted": True,
                "success": recovery_result["success"],
                "duration": recovery_result["duration"],
                "steps_executed": len(recovery_result["steps_executed"]),
                "error": recovery_result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Exception during auto-recovery: {str(e)}")
            return {
                "attempted": True,
                "success": False,
                "error": str(e)
            }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        with self.lock:
            stats = self.classification_stats.copy()
        
        # Convert enum keys to strings for JSON serialization
        stats['by_category'] = {cat.value: count for cat, count in stats['by_category'].items()}
        stats['by_severity'] = {sev.value: count for sev, count in stats['by_severity'].items()}
        
        # Calculate success rate
        if stats['auto_recoveries_attempted'] > 0:
            stats['auto_recovery_success_rate'] = (
                stats['auto_recoveries_successful'] / stats['auto_recoveries_attempted']
            )
        else:
            stats['auto_recovery_success_rate'] = 0.0
        
        return stats
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of available error patterns."""
        all_patterns = self.pattern_db.get_all_patterns()
        
        summary = {
            'total_patterns': len(all_patterns),
            'by_category': {},
            'by_severity': {},
            'critical_patterns': []
        }
        
        for pattern in all_patterns.values():
            # Count by category
            cat_name = pattern.category.value
            summary['by_category'][cat_name] = summary['by_category'].get(cat_name, 0) + 1
            
            # Count by severity
            sev_name = pattern.severity.value
            summary['by_severity'][sev_name] = summary['by_severity'].get(sev_name, 0) + 1
            
            # Collect critical patterns
            if pattern.severity == ErrorSeverity.CRITICAL:
                summary['critical_patterns'].append({
                    'name': pattern.name,
                    'category': pattern.category.value,
                    'signatures': pattern.signatures[:3]  # First 3 signatures
                })
        
        return summary
    
    def export_classified_error(self, classified_error: ClassifiedError) -> Dict[str, Any]:
        """Export classified error as dictionary for logging/reporting."""
        result = asdict(classified_error)
        
        # Convert enums to strings
        result['category'] = classified_error.category.value
        result['severity'] = classified_error.severity.value
        
        # Convert datetime to string
        if result['context']['timestamp']:
            result['context']['timestamp'] = classified_error.context.timestamp.isoformat()
        
        return result

# Global classifier instance for easy import
_global_classifier = None

def get_classifier() -> CoreMLErrorClassifier:
    """Get or create the global classifier instance."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = CoreMLErrorClassifier()
    return _global_classifier

def classify_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ClassifiedError:
    """Convenience function to classify an error using the global classifier."""
    return get_classifier().classify_error(error, context)

def attempt_auto_recovery(classified_error: ClassifiedError, 
                         recovery_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to attempt auto-recovery using the global classifier."""
    return get_classifier().attempt_auto_recovery(classified_error, recovery_context)