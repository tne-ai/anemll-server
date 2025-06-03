# Decision Log

This file records architectural and implementation decisions using a list format.
2025-06-01 22:44:44 - Log of updates made.

## Decision: LRU Cache-Based Dynamic Model Loading

[2025-06-01 23:04:32] - Selected LRU cache with configurable size for model management

## Rationale

* **Memory Efficiency**: Only keeps recently used models in memory
* **Performance Balance**: Fast access to cached models, acceptable loading time for new models
* **Configurability**: Admin can adjust cache size based on available system resources
* **Predictable Memory Usage**: LRU eviction provides bounded memory consumption
* **Production Ready**: Well-tested pattern with clear behavior under load

## Implementation Details

* Use Python's `functools.lru_cache` or custom LRU implementation
* Default cache size: 3 models (configurable via environment variable)
* Thread-safe operations using threading locks
* Graceful degradation when cache is full
* Memory monitoring and logging for cache hits/misses
* Automatic cleanup of evicted models to free GPU/ANE resources

## Decision: Model Directory Structure

[2025-06-01 23:04:32] - Support recursive scanning of MODEL_DIR with meta.yaml detection

## Rationale

* **Flexibility**: Supports both flat and nested directory structures
* **Validation**: meta.yaml presence confirms valid Anemll model
* **Metadata**: meta.yaml contains all required model configuration
* **Scalability**: Easy to add new models by dropping them in subdirectories

## Implementation Details

* Recursive directory scanning starting from MODEL_DIR
* Look for meta.yaml as model validity indicator
* Extract model name from directory name or meta.yaml
* Validate presence of all required model components (.mlmodelc files)
## Decision: Segmentation Fault Testing Framework Design

[2025-06-02 10:32:15] - Created comprehensive testing framework for segfault detection and recovery during model switching

## Rationale

* **Proactive Problem Detection**: Identify segfault sources before production deployment
* **Recovery Validation**: Ensure system can handle and recover from crashes gracefully
* **Systematic Approach**: Target specific failure modes identified through code analysis
* **Automated Testing**: Reduce manual testing burden and improve reproducibility
* **Documentation**: Provide clear debugging guidance for future issues

## Implementation Details

* Two-tier testing approach: comprehensive recovery tests + focused trigger tests
* Target 5 primary segfault sources: CoreML drivers, memory corruption, thread safety, resource exhaustion, file corruption
* Automatic server lifecycle management in recovery tests
* Stress testing with concurrent access, rapid switching, and memory pressure
* Integration with existing Makefile for easy execution
* Detailed logging and health monitoring during tests
* Recovery metrics tracking (crash detection, restart success, continued operation)
## Decision: Segfault Root Cause Analysis Results

[2025-06-02 17:00:45] - Completed comprehensive diagnostic testing to identify crash sources in dynamic model loading system

## Rationale

* **Systematic Testing Approach**: Used three-tier testing framework to isolate crash conditions
* **Evidence-Based Analysis**: Collected 227 seconds of stress testing data with memory monitoring
* **Pattern Recognition**: CoreML error classifier identified specific failure categories
* **Version Comparison**: Confirmed DeepSeek model v0.2.0 shows major stability improvement vs v0.1.1

## Implementation Details

* **Primary Root Cause**: ANE driver instability during intensive model loading (4/13 classified errors)
* **Secondary Root Cause**: Memory allocation failures under stress conditions (3/13 classified errors)  
* **Recovery Success**: 100% recovery rate validates crash handling mechanisms
* **Stability Improvement**: Crash frequency reduced significantly but not eliminated
* **Testing Framework**: Enhanced with CoreML error classification and automated recovery validation
* **Next Steps**: Implement ANE health monitoring, request throttling, and CPU fallback mechanisms