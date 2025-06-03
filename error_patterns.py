#!/usr/bin/env python
"""
Error pattern database for CoreML/ANE-specific error classification.

This module contains comprehensive error signatures and patterns identified 
from segfault analysis, focusing on the highest-priority crash sources:
1. CoreML/ANE Driver Issues
2. Memory Corruption During Model Loading  
3. Thread Safety Issues
4. Resource Exhaustion
5. Model File Corruption
"""

import re
from enum import Enum
from typing import Dict, List, Pattern, Optional, NamedTuple
from dataclasses import dataclass

class ErrorCategory(Enum):
    """Categories of CoreML-related errors."""
    ANE_DRIVER = "ane_driver"
    MEMORY_ALLOCATION = "memory_allocation"
    THREAD_SAFETY = "thread_safety"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_CORRUPTION = "model_corruption"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Severity levels for classified errors."""
    CRITICAL = "critical"      # Immediate crash risk
    HIGH = "high"             # Likely crash
    MEDIUM = "medium"         # Potential issues
    LOW = "low"              # Informational

@dataclass
class ErrorPattern:
    """Represents a specific error pattern."""
    name: str
    category: ErrorCategory
    severity: ErrorSeverity
    signatures: List[str]
    regex_patterns: List[Pattern]
    description: str
    typical_locations: List[str]
    triggers: List[str]

class ErrorPatternDatabase:
    """Database of known CoreML error patterns based on segfault analysis."""
    
    def __init__(self):
        """Initialize the error pattern database."""
        self.patterns = self._build_pattern_database()
        
    def _build_pattern_database(self) -> Dict[str, ErrorPattern]:
        """Build comprehensive pattern database from known crash sources."""
        patterns = {}
        
        # High Priority: ANE Driver Issues
        patterns.update(self._get_ane_driver_patterns())
        
        # High Priority: Memory Allocation Failures
        patterns.update(self._get_memory_allocation_patterns())
        
        # Medium Priority: Thread Safety Issues
        patterns.update(self._get_thread_safety_patterns())
        
        # Medium Priority: Resource Exhaustion
        patterns.update(self._get_resource_exhaustion_patterns())
        
        # Lower Priority: Model Corruption
        patterns.update(self._get_model_corruption_patterns())
        
        return patterns
    
    def _get_ane_driver_patterns(self) -> Dict[str, ErrorPattern]:
        """ANE driver error patterns - highest priority crash source."""
        return {
            "ane_device_unavailable": ErrorPattern(
                name="ANE Device Unavailable",
                category=ErrorCategory.ANE_DRIVER,
                severity=ErrorSeverity.CRITICAL,
                signatures=[
                    "ANE device not available",
                    "Neural Engine initialization error",
                    "ane device creation failed",
                    "Neural Engine not found",
                    "ANE compilation failed"
                ],
                regex_patterns=[
                    re.compile(r"ANE.*(?:not available|unavailable|failed)", re.IGNORECASE),
                    re.compile(r"Neural Engine.*(?:error|failed|not found)", re.IGNORECASE),
                    re.compile(r"ane.*(?:compilation|device).*failed", re.IGNORECASE)
                ],
                description="ANE hardware device is not available or failed to initialize",
                typical_locations=[
                    "model_manager.py:_load_model()",
                    "chat_full.py:load_models()",
                    "CoreML model initialization"
                ],
                triggers=[
                    "Hardware not supporting ANE",
                    "ANE driver issues",
                    "System resource conflicts",
                    "Thermal throttling"
                ]
            ),
            
            "coreml_framework_error": ErrorPattern(
                name="CoreML Framework Error",
                category=ErrorCategory.ANE_DRIVER,
                severity=ErrorSeverity.CRITICAL,
                signatures=[
                    "CoreML model compilation failed",
                    "com.apple.CoreML error",
                    "MLModel prediction failed",
                    "CoreML runtime error",
                    "Metal device creation failed"
                ],
                regex_patterns=[
                    re.compile(r"CoreML.*(?:compilation|prediction|runtime).*(?:failed|error)", re.IGNORECASE),
                    re.compile(r"com\.apple\.CoreML.*error", re.IGNORECASE),
                    re.compile(r"MLModel.*(?:failed|error)", re.IGNORECASE),
                    re.compile(r"Metal device.*(?:creation|failed)", re.IGNORECASE)
                ],
                description="CoreML framework encountered a critical error",
                typical_locations=[
                    "model_manager.py:137-152",
                    "chat_full.py:load_model()",
                    "CoreML model loading"
                ],
                triggers=[
                    "Invalid model format",
                    "Hardware incompatibility",
                    "Driver version mismatch",
                    "Framework corruption"
                ]
            ),
            
            "ane_communication_timeout": ErrorPattern(
                name="ANE Communication Timeout", 
                category=ErrorCategory.ANE_DRIVER,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "ANE communication timeout",
                    "Neural Engine timeout",
                    "ANE driver timeout",
                    "Neural Engine not responding"
                ],
                regex_patterns=[
                    re.compile(r"ANE.*(?:communication|driver).*timeout", re.IGNORECASE),
                    re.compile(r"Neural Engine.*(?:timeout|not responding)", re.IGNORECASE)
                ],
                description="ANE device communication timed out",
                typical_locations=[
                    "model_manager.py:_load_model()",
                    "StreamingTokenGenerator operations"
                ],
                triggers=[
                    "High system load",
                    "Thermal throttling",
                    "Driver instability",
                    "Hardware issues"
                ]
            )
        }
    
    def _get_memory_allocation_patterns(self) -> Dict[str, ErrorPattern]:
        """Memory allocation error patterns - highest priority crash source."""
        return {
            "tensor_allocation_failure": ErrorPattern(
                name="Tensor Allocation Failure",
                category=ErrorCategory.MEMORY_ALLOCATION,
                severity=ErrorSeverity.CRITICAL,
                signatures=[
                    "Failed to allocate tensor memory",
                    "Cannot allocate memory for tensor",
                    "Tensor allocation failed",
                    "Out of memory for tensor operation"
                ],
                regex_patterns=[
                    re.compile(r"(?:Failed to allocate|Cannot allocate).*tensor.*memory", re.IGNORECASE),
                    re.compile(r"Tensor allocation failed", re.IGNORECASE),
                    re.compile(r"Out of memory.*tensor", re.IGNORECASE)
                ],
                description="Failed to allocate memory for large tensors during model loading",
                typical_locations=[
                    "model_manager.py:create_unified_state()",
                    "chat_full.py:create_unified_state()",
                    "Model tensor initialization"
                ],
                triggers=[
                    "Large model size",
                    "Insufficient system memory",
                    "Memory fragmentation",
                    "Multiple models loaded"
                ]
            ),
            
            "gpu_memory_exhausted": ErrorPattern(
                name="GPU Memory Exhausted",
                category=ErrorCategory.MEMORY_ALLOCATION,
                severity=ErrorSeverity.CRITICAL,
                signatures=[
                    "GPU memory exhausted",
                    "CUDA out of memory",
                    "Metal out of memory",
                    "GPU allocation failed"
                ],
                regex_patterns=[
                    re.compile(r"GPU.*(?:memory exhausted|out of memory)", re.IGNORECASE),
                    re.compile(r"CUDA out of memory", re.IGNORECASE),
                    re.compile(r"Metal.*out of memory", re.IGNORECASE),
                    re.compile(r"GPU allocation failed", re.IGNORECASE)
                ],
                description="GPU/ANE memory exhausted during model operations",
                typical_locations=[
                    "model_manager.py:_load_model()",
                    "Model inference operations",
                    "LRU cache operations"
                ],
                triggers=[
                    "Large models exceeding GPU memory",
                    "Multiple concurrent models",
                    "Memory leaks",
                    "Insufficient GPU memory"
                ]
            ),
            
            "memory_corruption": ErrorPattern(
                name="Memory Corruption",
                category=ErrorCategory.MEMORY_ALLOCATION,
                severity=ErrorSeverity.CRITICAL,
                signatures=[
                    "malloc(): corrupted top size",
                    "double free or corruption",
                    "heap corruption detected",
                    "segfault in create_unified_state"
                ],
                regex_patterns=[
                    re.compile(r"malloc\(\):.*corrupted", re.IGNORECASE),
                    re.compile(r"double free.*corruption", re.IGNORECASE),
                    re.compile(r"heap corruption", re.IGNORECASE),
                    re.compile(r"segfault.*create_unified_state", re.IGNORECASE)
                ],
                description="Memory corruption during tensor operations",
                typical_locations=[
                    "model_manager.py:137-152",
                    "chat_full.py:create_unified_state()",
                    "Tensor memory operations"
                ],
                triggers=[
                    "Buffer overflows",
                    "Use after free",
                    "Threading issues",
                    "Hardware memory errors"
                ]
            ),
            
            "memory_error": ErrorPattern(
                name="Python Memory Error",
                category=ErrorCategory.MEMORY_ALLOCATION,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "MemoryError",
                    "Out of memory",
                    "Cannot allocate memory"
                ],
                regex_patterns=[
                    re.compile(r"MemoryError", re.IGNORECASE),
                    re.compile(r"Out of memory", re.IGNORECASE),
                    re.compile(r"Cannot allocate memory", re.IGNORECASE)
                ],
                description="Python-level memory allocation failure",
                typical_locations=[
                    "model_manager.py:_load_model()",
                    "Large data structure creation",
                    "Model loading operations"
                ],
                triggers=[
                    "Insufficient system RAM",
                    "Memory fragmentation",
                    "Large model loading",
                    "Memory leaks"
                ]
            )
        }
    
    def _get_thread_safety_patterns(self) -> Dict[str, ErrorPattern]:
        """Thread safety error patterns - medium priority crash source."""
        return {
            "concurrent_access_violation": ErrorPattern(
                name="Concurrent Access Violation",
                category=ErrorCategory.THREAD_SAFETY,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "concurrent access violation",
                    "race condition detected",
                    "threading violation",
                    "concurrent modification"
                ],
                regex_patterns=[
                    re.compile(r"concurrent.*(?:access|modification).*violation", re.IGNORECASE),
                    re.compile(r"race condition", re.IGNORECASE),
                    re.compile(r"threading violation", re.IGNORECASE)
                ],
                description="Concurrent access to shared resources without proper synchronization",
                typical_locations=[
                    "anemll-server.py:StreamingTokenGenerator",
                    "model_manager.py:LRU operations",
                    "Queue operations"
                ],
                triggers=[
                    "Multiple simultaneous requests",
                    "Model switching during inference",
                    "Cache operations during loading",
                    "Queue access conflicts"
                ]
            ),
            
            "threading_deadlock": ErrorPattern(
                name="Threading Deadlock",
                category=ErrorCategory.THREAD_SAFETY,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "deadlock detected",
                    "threading lock timeout",
                    "lock acquisition timeout",
                    "ThreadError"
                ],
                regex_patterns=[
                    re.compile(r"deadlock detected", re.IGNORECASE),
                    re.compile(r"(?:threading|lock).*timeout", re.IGNORECASE),
                    re.compile(r"lock acquisition timeout", re.IGNORECASE),
                    re.compile(r"ThreadError", re.IGNORECASE)
                ],
                description="Threading deadlock or lock timeout",
                typical_locations=[
                    "model_manager.py:ModelManager locks",
                    "StreamingTokenGenerator operations",
                    "LRU cache operations"
                ],
                triggers=[
                    "Nested lock acquisition",
                    "Lock ordering issues",
                    "Long-running operations",
                    "Resource contention"
                ]
            ),
            
            "queue_operation_failed": ErrorPattern(
                name="Queue Operation Failed",
                category=ErrorCategory.THREAD_SAFETY,
                severity=ErrorSeverity.MEDIUM,
                signatures=[
                    "queue operation failed",
                    "queue timeout",
                    "queue full",
                    "queue empty"
                ],
                regex_patterns=[
                    re.compile(r"queue.*(?:operation failed|timeout|full|empty)", re.IGNORECASE)
                ],
                description="Queue operations failed in streaming token generation",
                typical_locations=[
                    "anemll-server.py:StreamingTokenGenerator",
                    "Token generation queues",
                    "Response streaming"
                ],
                triggers=[
                    "High request volume",
                    "Slow model inference",
                    "Queue size limits",
                    "Consumer/producer imbalance"
                ]
            )
        }
    
    def _get_resource_exhaustion_patterns(self) -> Dict[str, ErrorPattern]:
        """Resource exhaustion error patterns - medium priority crash source."""
        return {
            "system_resource_exhaustion": ErrorPattern(
                name="System Resource Exhaustion",
                category=ErrorCategory.RESOURCE_EXHAUSTION,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "Resource temporarily unavailable",
                    "Too many open files",
                    "Process limit exceeded",
                    "System overloaded"
                ],
                regex_patterns=[
                    re.compile(r"Resource temporarily unavailable", re.IGNORECASE),
                    re.compile(r"Too many open files", re.IGNORECASE),
                    re.compile(r"Process limit exceeded", re.IGNORECASE),
                    re.compile(r"System overloaded", re.IGNORECASE)
                ],
                description="System resources exhausted during model operations",
                typical_locations=[
                    "Model file loading",
                    "Process creation",
                    "Network operations"
                ],
                triggers=[
                    "Too many concurrent requests",
                    "File descriptor leaks",
                    "Process limit reached",
                    "System resource pressure"
                ]
            ),
            
            "disk_space_exhausted": ErrorPattern(
                name="Disk Space Exhausted",
                category=ErrorCategory.RESOURCE_EXHAUSTION,
                severity=ErrorSeverity.MEDIUM,
                signatures=[
                    "No space left on device",
                    "Disk full",
                    "Cannot write to disk"
                ],
                regex_patterns=[
                    re.compile(r"No space left on device", re.IGNORECASE),
                    re.compile(r"Disk full", re.IGNORECASE),
                    re.compile(r"Cannot write to disk", re.IGNORECASE)
                ],
                description="Insufficient disk space for model operations",
                typical_locations=[
                    "Model downloading",
                    "Cache file creation",
                    "Log file writing"
                ],
                triggers=[
                    "Large model downloads",
                    "Log file growth",
                    "Cache accumulation",
                    "Temporary file creation"
                ]
            )
        }
    
    def _get_model_corruption_patterns(self) -> Dict[str, ErrorPattern]:
        """Model corruption error patterns - lower priority crash source."""
        return {
            "invalid_model_format": ErrorPattern(
                name="Invalid Model Format",
                category=ErrorCategory.MODEL_CORRUPTION,
                severity=ErrorSeverity.HIGH,
                signatures=[
                    "Invalid model format",
                    "Corrupted model file",
                    "Model file corrupted",
                    "Invalid .mlmodelc"
                ],
                regex_patterns=[
                    re.compile(r"Invalid model format", re.IGNORECASE),
                    re.compile(r"(?:Corrupted|corrupted).*model", re.IGNORECASE),
                    re.compile(r"Invalid.*\.mlmodelc", re.IGNORECASE)
                ],
                description="Model file format is invalid or corrupted",
                typical_locations=[
                    "model_manager.py:_load_model()",
                    "chat_full.py:load_model()",
                    "CoreML model loading"
                ],
                triggers=[
                    "Incomplete model download",
                    "File corruption",
                    "Wrong model format",
                    "Version incompatibility"
                ]
            ),
            
            "metadata_mismatch": ErrorPattern(
                name="Metadata Mismatch",
                category=ErrorCategory.MODEL_CORRUPTION,
                severity=ErrorSeverity.MEDIUM,
                signatures=[
                    "Metadata mismatch",
                    "Invalid metadata",
                    "meta.yaml error",
                    "Missing model component"
                ],
                regex_patterns=[
                    re.compile(r"Metadata.*(?:mismatch|invalid)", re.IGNORECASE),
                    re.compile(r"meta\.yaml.*error", re.IGNORECASE),
                    re.compile(r"Missing model component", re.IGNORECASE)
                ],
                description="Model metadata is inconsistent or invalid",
                typical_locations=[
                    "model_registry.py:scan_models()",
                    "model_manager.py:_load_model()",
                    "Model validation"
                ],
                triggers=[
                    "Incomplete model files",
                    "Version mismatches",
                    "Manual file modifications",
                    "Download interruptions"
                ]
            )
        }
    
    def get_all_patterns(self) -> Dict[str, ErrorPattern]:
        """Get all error patterns."""
        return self.patterns
    
    def get_patterns_by_category(self, category: ErrorCategory) -> Dict[str, ErrorPattern]:
        """Get patterns filtered by category."""
        return {
            name: pattern for name, pattern in self.patterns.items()
            if pattern.category == category
        }
    
    def get_patterns_by_severity(self, severity: ErrorSeverity) -> Dict[str, ErrorPattern]:
        """Get patterns filtered by severity."""
        return {
            name: pattern for name, pattern in self.patterns.items()
            if pattern.severity == severity
        }
    
    def get_critical_patterns(self) -> Dict[str, ErrorPattern]:
        """Get all critical severity patterns."""
        return self.get_patterns_by_severity(ErrorSeverity.CRITICAL)
    
    def get_high_priority_patterns(self) -> Dict[str, ErrorPattern]:
        """Get critical and high severity patterns."""
        critical = self.get_patterns_by_severity(ErrorSeverity.CRITICAL)
        high = self.get_patterns_by_severity(ErrorSeverity.HIGH)
        return {**critical, **high}