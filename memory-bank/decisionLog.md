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
## Decision: Convert to uv for Python Package Control

[2025-06-02 17:07:15] - Migrated dependency management from requirements.txt to pyproject.toml with uv integration

## Rationale

* **Modern Python Standards**: pyproject.toml is the modern standard for Python project configuration
* **Single Source of Truth**: Consolidates project metadata and dependencies in one file
* **Better Dependency Resolution**: uv provides faster and more reliable dependency resolution
* **Lockfile Management**: uv.lock provides deterministic builds and better reproducibility
* **Toolchain Consistency**: Aligns with modern Python packaging best practices

## Implementation Details

* Moved all dependencies from requirements.txt to pyproject.toml [project.dependencies] section
* Updated Makefile install target to use `uv pip install .` instead of `uv pip install -r requirements.txt`
* Preserved existing dependency versions to maintain compatibility
* requirements.txt can be removed after testing confirms the new setup works correctly
* uv.lock should be regenerated with `uv lock` command to reflect new dependency structure
[2025-01-02 17:44:56] - **Model Memory Pre-Check Decision**: Implement predictive memory checking before model loading using linear scaling based on model parameters. Requirements: 1B models need 300MB free memory, 8B models need 1GB, with linear scaling between sizes. This prevents crashes by rejecting model loads that exceed available unified memory.
[2025-01-02 17:47:23] - **Refined Memory Assessment Strategy**: Updated design to use RSS-based memory assessment instead of system-wide memory, and extract model parameters from Huggingface model config files in ./models directory rather than parsing model names. This provides more accurate ANE memory availability and reliable parameter detection.
[2025-01-02 17:51:43] - **Simplified Memory Assessment**: Updated memory status function to return current RSS directly as available memory for models, simplifying the calculation and using RSS as the direct proxy for ANE memory availability.
## Decision: Model Parameter Determination via Name Parsing

[2025-06-02 18:05:56] - Changed model parameter determination strategy to parse model names (e.g., "1B", "8B") instead of reading `config.json` files. This is a global server change.

## Rationale

*   **User Directive**: This change is based on a direct user request to modify the existing behavior.
*   **Simplified Parameter Source (for some use cases)**: Relies on a naming convention rather than external configuration files for parameter size.

## Implementation Details

*   Modify `anemll-server.py`:
    *   Introduce `get_model_parameters_from_name(model_name: str) -> Optional[float]` to parse size (e.g., "XB") from the model name string using regex.
    *   Update `can_load_model` to use `get_model_parameters_from_name` instead of `get_model_parameters_from_config`.
    *   The `get_model_parameters_from_config` function will be deprecated/removed.
*   **Impact**: This is a global change affecting how the server estimates memory requirements for all models. Existing tests will implicitly use this new logic.
*   **Supersedes**: This decision supersedes the part of "[2025-01-02 17:47:23] - Refined Memory Assessment Strategy" that specified extracting parameters from Huggingface config files.
## Decision: Graceful Memory Exhaustion Handling

[2025-06-02 22:43:00] - Fixed FastAPI exception crash when insufficient memory by implementing graceful degradation instead of raising HTTPException during streaming responses.

## Rationale

* **ASGI Compatibility**: HTTPException cannot be raised after streaming response has started without crashing ASGI application
* **User Experience**: Graceful error messages are better than server crashes
* **API Consistency**: Error responses should follow OpenAI API format even for resource exhaustion
* **Operational Stability**: Server should continue running and serving other requests even when some models can't be loaded

## Implementation Details

* **Streaming Responses**: Return async generator that yields proper SSE-formatted error messages instead of raising HTTPException
* **Non-Streaming Responses**: Return proper JSON error response with 'error' finish_reason instead of raising HTTPException  
* **Comprehensive Coverage**: Applied same pattern to all error conditions (memory insufficient, model not found, loading failed)
* **Diagnostic Logging**: Added clear logging to distinguish between memory checks and graceful degradation
* **Error Message Format**: Consistent "❌ [error description]. Please try a smaller model or free up memory." format