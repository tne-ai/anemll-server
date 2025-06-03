# Segmentation Fault Testing for Anemll Server

This document describes the segmentation fault testing framework for the Anemll server's dynamic model loading system.

## Overview

The segmentation fault testing framework consists of two main test scripts designed to detect, trigger, and validate recovery from server crashes during model switching operations.

## Test Scripts

### 1. `test_segfault_recovery.py` - Comprehensive Recovery Testing

This script provides a complete test suite that:

- **Starts and stops the server automatically**
- **Monitors server health** during operations
- **Detects crashes** and measures recovery capabilities
- **Stress tests** model switching under various conditions
- **Provides detailed reporting** of crashes and recovery success rates

#### Key Test Scenarios:

- **Memory Pressure Test**: Rapid model switching to stress memory management
- **Concurrent Request Test**: Simultaneous requests to different models
- **Stress Test**: Extended period of model switching with health monitoring
- **Server Recovery**: Automatic detection and recovery from crashes

#### Usage:
```bash
# Run comprehensive recovery tests
make test-recovery

# Or run directly
python test_segfault_recovery.py
```

### 2. `trigger_segfault_test.py` - Focused Crash Triggering

This script specifically targets scenarios most likely to cause segmentation faults:

- **Cache Thrashing**: Rapid LRU cache eviction and model loading
- **Rapid Model Switching**: High-frequency model changes to stress CoreML loading
- **Concurrent Model Access**: Thread safety stress testing
- **Memory Exhaustion**: Large context requests to trigger memory issues
- **Interrupt During Loading**: Request cancellation during model loading

#### Usage:
```bash
# Trigger segfault scenarios (requires running server)
make test-segfault

# Or run directly (with server running)
python trigger_segfault_test.py
```

### 3. `enhanced_segfault_tests.py` - Advanced Testing with CoreML Error Classification

This script provides comprehensive testing with the new CoreML error classification system:

- **Error Pattern Recognition**: Validates classification accuracy for known error patterns
- **Enhanced Recovery Testing**: Tests automated recovery capabilities
- **Context-Aware Classification**: Demonstrates how context affects classification
- **Performance Monitoring**: Tracks classification and recovery performance
- **Comprehensive Reporting**: Generates detailed analysis reports

#### Key Features:

- **Real-time Error Classification**: Automatically classifies errors during testing
- **Automated Recovery Testing**: Tests recovery strategies for critical errors
- **Pattern Recognition Validation**: Ensures 90%+ accuracy in error classification
- **Performance Metrics**: Tracks classification time and recovery success rates
- **Integration with Existing Tests**: Extends current test framework seamlessly

#### Usage:
```bash
# Run enhanced tests with CoreML error classification
make test-enhanced

# Or run directly
python enhanced_segfault_tests.py
```

### 4. `coreml_classifier_demo.py` - CoreML Error Classification Demo

This script demonstrates the capabilities of the CoreML error classification system:

- **Error Pattern Recognition Demo**: Shows how different error types are classified
- **Recovery Strategy Demo**: Demonstrates automated recovery recommendations
- **Context-Aware Analysis**: Shows how context influences classification accuracy
- **Monitoring Integration**: Demonstrates real-time error monitoring
- **Performance Analysis**: Shows classification speed and accuracy metrics

#### Usage:
```bash
# Run CoreML error classification demo
make test-classifier

# Or run directly
python coreml_classifier_demo.py
```

## Most Likely Segfault Sources

Based on code analysis, the most probable segfault sources are:

### 1. **CoreML/ANE Driver Issues** (High Priority)
- **Location**: [`model_manager.py:_load_model()`](model_manager.py:113-186)
- **Trigger**: During `load_models()` call with CoreML components
- **Symptoms**: Immediate crash during model loading, driver-level failures

### 2. **Memory Corruption During Model Loading** (High Priority)
- **Location**: [`model_manager.py:137-152`](model_manager.py:137-152)
- **Trigger**: Large model tensors, memory allocation failures
- **Symptoms**: Crashes during `create_unified_state()` or tensor operations

### 3. **Thread Safety Issues** (Medium Priority)
- **Location**: [`anemll-server.py:StreamingTokenGenerator`](anemll-server.py:84-314)
- **Trigger**: Concurrent model access, queue operations
- **Symptoms**: Race conditions, inconsistent state

### 4. **Resource Exhaustion** (Medium Priority)
- **Location**: Model loading and LRU cache management
- **Trigger**: Multiple large models, memory pressure
- **Symptoms**: System-level crashes, OOM conditions

### 5. **Model File Corruption** (Lower Priority)
- **Location**: `.mlmodelc` file loading
- **Trigger**: Corrupted model files, incomplete downloads
- **Symptoms**: Crashes during CoreML model initialization

## Expected Test Behavior

### Normal Operation
- **No crashes**: Server handles all model switches successfully
- **Graceful degradation**: Server returns appropriate error messages for failures
- **Resource cleanup**: Proper cleanup of models during cache eviction

### Segfault Scenarios
- **Process termination**: Server process exits unexpectedly
- **Signal 11 (SIGSEGV)**: Segmentation fault signal in system logs
- **CoreML errors**: Driver-level error messages before crash
- **Memory errors**: malloc/free corruption messages

## Running the Tests

### Prerequisites
```bash
# Ensure models are downloaded
make models

# Install dependencies
make install
```

### Test Execution

#### Option 1: Run all tests including enhanced CoreML classification (recommended)
```bash
make test-all-enhanced
```

#### Option 2: Run enhanced tests only
```bash
make test-enhanced
```

#### Option 3: Run specific tests

**Enhanced Testing** (with CoreML error classification):
```bash
make test-enhanced
```

**Recovery Testing** (starts/stops server automatically):
```bash
make test-recovery
```

**Trigger Testing** (requires manually started server):
```bash
# Terminal 1: Start server
make run

# Terminal 2: Run trigger tests
make test-segfault
```

**CoreML Classification Demo**:
```bash
make test-classifier
```

#### Option 4: Run traditional tests only
```bash
make test-all
```

#### Option 5: Manual execution
```bash
# Start server
python anemll-server.py

# In another terminal, run tests
python trigger_segfault_test.py
python test_segfault_recovery.py
python enhanced_segfault_tests.py
```

## Interpreting Results

### Success Indicators
- ✅ **No crashes detected**: Server handles all stress tests
- ✅ **High success rate**: >95% of model switches successful
- ✅ **Stable memory usage**: No significant memory leaks
- ✅ **Clean recovery**: Proper error handling for failures

### Failure Indicators
- ❌ **Process crashes**: Server terminates unexpectedly
- ❌ **Low success rate**: <90% of model switches successful
- ❌ **Memory growth**: Continuous memory usage increase
- ❌ **Hung requests**: Requests timeout without response

### Recovery Validation
- 🔄 **Crash detection**: Automatic detection of server termination
- 🔄 **Recovery attempts**: Successful server restart after crash
- 🔄 **Continued operation**: Normal operation after recovery

## Debugging Segfaults

If segfaults are detected:

### 1. **Enable Core Dumps**
```bash
ulimit -c unlimited
export PYTHONMALLOC=debug
python anemll-server.py
```

### 2. **Run with GDB**
```bash
gdb python
(gdb) run anemll-server.py
# When crash occurs:
(gdb) bt
(gdb) info registers
```

### 3. **Check System Logs**
```bash
# macOS
sudo dmesg | grep python

# Check for CoreML/ANE errors
grep -i coreml /var/log/system.log
```

### 4. **Memory Debugging**
```bash
# Enable malloc debugging
export MALLOC_CHECK_=1
export PYTHONMALLOC=debug
python anemll-server.py
```

## Known Issues

### Python GIL Limitations
- **Issue**: After multiple model switches, Python's GIL may become corrupted
- **Symptoms**: Freezing after successful operations
- **Workaround**: Server restart required
- **Reference**: Documented in README.md

### CoreML Memory Management
- **Issue**: CoreML models may not release GPU/ANE memory immediately
- **Symptoms**: Memory usage growth with model switching
- **Mitigation**: LRU cache limits concurrent loaded models

## Contributing

When adding new test scenarios:

1. **Target specific failure modes** identified in code analysis
2. **Include proper logging** for debugging
3. **Handle exceptions gracefully** to continue testing
4. **Document expected behavior** and failure conditions
5. **Update this documentation** with new scenarios

## Contact

For issues with segfault testing or questions about the framework, refer to the main project documentation and issue tracker.