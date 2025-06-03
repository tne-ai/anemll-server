# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions.
2025-06-01 22:44:25 - Log of updates made.

## Current Focus

**Server Health & Memory Analysis**: Investigated how server health, particularly memory availability, is checked and logged, including the source of specific "Server health - Memory: ... CPU: ..." log messages and the `memory_exhaustion_attack` test method.
**Global Change to Model Parameter Determination**: Planning and initiating global change to server logic for model parameter determination. New strategy involves parsing model names (e.g., "1B", "8B") instead of reading `config.json` files. This will affect memory pre-checks.

## Recent Changes

[2025-06-01 22:46:04] - Completed initial codebase review and Memory Bank population
[2025-06-01 23:01:53] - Received user requirement for dynamic model switching functionality
[2025-06-02 17:30:00] - Clarified that `anemll-server.py`'s `check_memory_availability()` checks system-wide free memory for pre-load conditions and error logging.
[2025-06-02 17:30:00] - Identified that `trigger_segfault_test.py`'s `memory_exhaustion_attack()` stresses server memory by sending large requests, not by querying free memory.
[2025-06-02 17:30:00] - Pinpointed `test_segfault_recovery.py`'s `SegfaultTestCase.monitor_server_health()` as the source of "Server health - Memory: X MB, CPU: Y%" logs, which monitors the server process's RSS and CPU during tests.
[2025-06-02 18:06:19] - **Architectural Decision**: Approved plan to globally change model parameter determination from `config.json` reading to model name parsing (e.g., "1B", "8B") for memory pre-checks.

## Open Questions/Issues

* How should we handle model discovery in subdirectories vs flat structure?
* What's the best strategy for model caching vs loading on-demand?
* Should we pre-load all available models or load them lazily?
* How to handle model loading failures gracefully?
* What's the optimal way to organize model metadata for quick lookup?
[2025-06-02 16:35:49] - Updated DeepSeek model version from 0.1.1 to 0.2.0 across all test files and configuration files
[2025-06-02 17:00:35] - **DIAGNOSTIC COMPLETION**: Comprehensive segfault analysis completed with key findings:
  - ANE driver instability identified as primary crash source (4/13 errors)
  - Memory allocation failures under stress as secondary source (3/13 errors)
  - DeepSeek model v0.2.0 shows major stability improvement vs v0.1.1
  - System recovery mechanisms validated with 100% success rate
  - Crash frequency reduced but not eliminated under stress testing
[2025-01-02 17:42:03] - **RSS as ANE Memory Proxy**: Identified that Anemll server RSS serves as effective proxy for ANE memory usage since server has minimal overhead (~0.2MB baseline) and model loading causes dramatic RSS increases (300+MB for 1B model) representing unified memory allocation for ANE access.
[2025-01-02 17:55:38] - **Current Status**: Model Memory Pre-Check System implementation complete. The system now uses RSS as a proxy for ANE memory usage and performs predictive memory checking before model loading to prevent crashes. All changes integrated into anemll-server.py with empirical scaling formula (300MB for 1B + 100MB per additional billion parameters).
[2025-06-02 18:20:44] - **COMPREHENSIVE TEST SUITE EXECUTION RESULTS**
  - **Dynamic Loading**: ✅ All components operational (ModelRegistry, ModelManager, caching)
  - **Memory Pre-Check System**: ✅ Working correctly - preventing unsafe DeepSeek-8B loads (requires 1000MB vs 323MB available)
  - **Recovery System**: ✅ 100% success rate (5/5 recoveries) - server automatically restarts after crashes
  - **Error Classification**: ✅ Pattern recognition at 100% accuracy for CoreML error types
  - **System Stability**: Meta-Llama-3.2-1B (smaller model) loads successfully, larger models correctly rejected