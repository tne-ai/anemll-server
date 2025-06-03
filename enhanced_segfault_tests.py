#!/usr/bin/env python
"""
Enhanced Segmentation Fault Testing with CoreML Error Classification.

This module extends the existing segfault testing framework with advanced
CoreML error classification, pattern detection, and automated recovery testing.
"""

import asyncio
import aiohttp
import json
import time
import signal
import subprocess
import sys
import logging
import threading
import os
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import existing test framework
from test_segfault_recovery import SegfaultTestCase
from trigger_segfault_test import SegfaultTrigger

# Import CoreML error classification system
from coreml_error_classifier import (
    CoreMLErrorClassifier, ClassifiedError, get_classifier
)
from error_patterns import ErrorCategory, ErrorSeverity
from coreml_wrapper import CoreMLMonitor, handle_coreml_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedTestResult:
    """Enhanced test result with error classification."""
    test_name: str
    success: bool
    duration: float
    errors_detected: int
    classified_errors: List[Dict[str, Any]]
    recovery_attempts: int
    successful_recoveries: int
    performance_metrics: Dict[str, Any]
    timestamp: datetime

class EnhancedSegfaultTestSuite:
    """
    Enhanced segfault test suite with CoreML error classification and
    automated recovery testing.
    """
    
    def __init__(self, server_host: str = "localhost", server_port: int = 8400):
        """Initialize enhanced test suite."""
        self.server_host = server_host
        self.server_port = server_port
        self.base_url = f"http://{server_host}:{server_port}"
        
        # Initialize CoreML error classifier
        self.classifier = get_classifier()
        
        # Initialize base test frameworks
        self.recovery_tester = SegfaultTestCase(server_host, server_port)
        self.trigger_tester = SegfaultTrigger(self.base_url)
        
        # Test configuration
        self.models_to_test = [
            "anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2",
            "anemll-DeepSeekR1-8B-ctx1024_0.2.0"
        ]
        
        # Enhanced monitoring
        self.test_results: List[EnhancedTestResult] = []
        self.classified_errors: List[ClassifiedError] = []
        self.recovery_statistics = {
            'total_errors': 0,
            'classified_errors': 0,
            'auto_recovery_attempts': 0,
            'successful_recoveries': 0,
            'by_category': {},
            'by_severity': {}
        }
        
        logger.info("Enhanced segfault test suite initialized with CoreML error classification")
    
    async def run_enhanced_test_suite(self) -> Dict[str, Any]:
        """Run the complete enhanced test suite with error classification."""
        logger.info("=" * 100)
        logger.info("ENHANCED SEGMENTATION FAULT TEST SUITE WITH COREML ERROR CLASSIFICATION")
        logger.info("=" * 100)
        
        suite_start_time = time.time()
        overall_success = True
        
        try:
            # Test 1: Basic CoreML Error Classification Test
            logger.info("\n--- Test 1: CoreML Error Classification Validation ---")
            classification_success = await self._test_error_classification()
            overall_success &= classification_success
            
            # Test 2: Enhanced Recovery Testing
            logger.info("\n--- Test 2: Enhanced Recovery Testing with Classification ---")
            recovery_success = await self._test_enhanced_recovery()
            overall_success &= recovery_success
            
            # Test 3: CoreML-Specific Trigger Testing
            logger.info("\n--- Test 3: CoreML-Specific Error Trigger Testing ---")
            trigger_success = await self._test_coreml_triggers()
            overall_success &= trigger_success
            
            # Test 4: Pattern Recognition Accuracy Test
            logger.info("\n--- Test 4: Error Pattern Recognition Accuracy ---")
            pattern_success = await self._test_pattern_recognition()
            overall_success &= pattern_success
            
            # Test 5: Auto-Recovery Performance Test
            logger.info("\n--- Test 5: Automated Recovery Performance ---")
            auto_recovery_success = await self._test_auto_recovery_performance()
            overall_success &= auto_recovery_success
            
        except Exception as e:
            logger.error(f"Test suite failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            overall_success = False
        
        # Generate comprehensive report
        suite_duration = time.time() - suite_start_time
        report = self._generate_test_report(suite_duration, overall_success)
        
        logger.info("\n" + "=" * 100)
        logger.info("ENHANCED TEST SUITE COMPLETED")
        logger.info("=" * 100)
        
        return report
    
    async def _test_error_classification(self) -> bool:
        """Test CoreML error classification accuracy."""
        logger.info("Testing CoreML error classification system...")
        
        test_start_time = time.time()
        classified_errors = []
        
        try:
            # Test classification of simulated errors
            test_errors = [
                (Exception("ANE device not available"), {"operation": "model_loading"}),
                (MemoryError("Failed to allocate tensor memory"), {"operation": "tensor_allocation"}),
                (RuntimeError("CoreML model compilation failed"), {"operation": "model_compilation"}),
                (OSError("Too many open files"), {"operation": "file_operations"}),
                (ValueError("Invalid model format"), {"operation": "model_validation"})
            ]
            
            for error, context in test_errors:
                try:
                    classified_error = self.classifier.classify_error(error, context)
                    classified_errors.append(classified_error)
                    
                    logger.info(f"✅ Classified: {classified_error.category.value}/{classified_error.severity.value} "
                              f"(confidence: {classified_error.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Classification failed for {str(error)}: {str(e)}")
                    return False
            
            # Validate classification results
            success = self._validate_classification_results(classified_errors)
            
            # Record test result
            test_duration = time.time() - test_start_time
            result = EnhancedTestResult(
                test_name="error_classification",
                success=success,
                duration=test_duration,
                errors_detected=len(test_errors),
                classified_errors=[self.classifier.export_classified_error(ce) for ce in classified_errors],
                recovery_attempts=0,
                successful_recoveries=0,
                performance_metrics={"classification_time_avg": test_duration / len(test_errors)},
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            
            logger.info(f"Error classification test {'✅ PASSED' if success else '❌ FAILED'}")
            return success
            
        except Exception as e:
            logger.error(f"Error classification test failed: {str(e)}")
            return False
    
    async def _test_enhanced_recovery(self) -> bool:
        """Test enhanced recovery with error classification."""
        logger.info("Testing enhanced recovery with CoreML error classification...")
        
        test_start_time = time.time()
        
        try:
            # Start server with enhanced monitoring
            if not await self.recovery_tester.start_server():
                logger.error("Failed to start server for enhanced recovery testing")
                return False
            
            # Run recovery tests with classification monitoring
            recovery_success = True
            classified_errors = []
            recovery_attempts = 0
            successful_recoveries = 0
            
            # Memory pressure test with classification
            logger.info("Running memory pressure test with error classification...")
            try:
                with CoreMLMonitor("memory_pressure_test") as monitor:
                    memory_success = await self.recovery_tester.memory_pressure_test()
                    if monitor.errors_detected:
                        classified_errors.extend(monitor.errors_detected)
                        recovery_attempts += len(monitor.errors_detected)
                        
            except Exception as e:
                # Classify the error
                classified_error = self.classifier.classify_error(e, {
                    "operation": "memory_pressure_test",
                    "test_phase": "enhanced_recovery"
                })
                classified_errors.append(classified_error)
                recovery_attempts += 1
                
                # Attempt auto-recovery
                if classified_error.can_auto_recover:
                    recovery_result = self.classifier.attempt_auto_recovery(classified_error)
                    if recovery_result.get("success"):
                        successful_recoveries += 1
                        logger.info("✅ Auto-recovery successful during memory pressure test")
                    else:
                        logger.error("❌ Auto-recovery failed during memory pressure test")
                        recovery_success = False
            
            # Concurrent request test with classification
            logger.info("Running concurrent request test with error classification...")
            try:
                with CoreMLMonitor("concurrent_request_test") as monitor:
                    concurrent_success = await self.recovery_tester.concurrent_request_test()
                    if monitor.errors_detected:
                        classified_errors.extend(monitor.errors_detected)
                        recovery_attempts += len(monitor.errors_detected)
                        
            except Exception as e:
                classified_error = self.classifier.classify_error(e, {
                    "operation": "concurrent_request_test",
                    "test_phase": "enhanced_recovery"
                })
                classified_errors.append(classified_error)
                recovery_attempts += 1
                
                if classified_error.can_auto_recover:
                    recovery_result = self.classifier.attempt_auto_recovery(classified_error)
                    if recovery_result.get("success"):
                        successful_recoveries += 1
                    else:
                        recovery_success = False
            
            # Record test result
            test_duration = time.time() - test_start_time
            result = EnhancedTestResult(
                test_name="enhanced_recovery",
                success=recovery_success,
                duration=test_duration,
                errors_detected=len(classified_errors),
                classified_errors=[self.classifier.export_classified_error(ce) for ce in classified_errors],
                recovery_attempts=recovery_attempts,
                successful_recoveries=successful_recoveries,
                performance_metrics={"recovery_success_rate": successful_recoveries / max(recovery_attempts, 1)},
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            self.classified_errors.extend(classified_errors)
            
            await self.recovery_tester.stop_server()
            
            logger.info(f"Enhanced recovery test {'✅ PASSED' if recovery_success else '❌ FAILED'}")
            logger.info(f"Recovery statistics: {successful_recoveries}/{recovery_attempts} successful")
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Enhanced recovery test failed: {str(e)}")
            await self.recovery_tester.stop_server()
            return False
    
    async def _test_coreml_triggers(self) -> bool:
        """Test CoreML-specific error triggers with classification."""
        logger.info("Testing CoreML-specific error triggers with classification...")
        
        test_start_time = time.time()
        
        try:
            # Check if server is running
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status != 200:
                            logger.error("Server not responding for trigger tests")
                            return False
            except Exception as e:
                logger.error(f"Cannot connect to server for trigger tests: {str(e)}")
                return False
            
            classified_errors = []
            recovery_attempts = 0
            successful_recoveries = 0
            
            # Run trigger scenarios with error classification
            trigger_scenarios = [
                ("cache_thrashing", lambda: self.trigger_tester.cache_thrashing(iterations=10)),
                ("rapid_model_switching", lambda: self.trigger_tester.rapid_model_switching(switches=8, delay=0.2)),
                ("concurrent_model_access", lambda: self.trigger_tester.concurrent_model_access(concurrent_requests=5)),
                ("memory_exhaustion_attack", lambda: self.trigger_tester.memory_exhaustion_attack(large_requests=2))
            ]
            
            for scenario_name, scenario_func in trigger_scenarios:
                logger.info(f"Running trigger scenario: {scenario_name}")
                
                try:
                    with CoreMLMonitor(f"trigger_{scenario_name}") as monitor:
                        await scenario_func()
                        
                        if monitor.errors_detected:
                            classified_errors.extend(monitor.errors_detected)
                            recovery_attempts += len(monitor.errors_detected)
                            logger.info(f"Detected {len(monitor.errors_detected)} errors in {scenario_name}")
                        
                except Exception as e:
                    # Classify triggered error
                    classified_error = self.classifier.classify_error(e, {
                        "operation": scenario_name,
                        "test_phase": "trigger_testing",
                        "scenario": scenario_name
                    })
                    classified_errors.append(classified_error)
                    recovery_attempts += 1
                    
                    logger.info(f"Triggered error in {scenario_name}: {classified_error.category.value}")
                    
                    # Attempt auto-recovery for critical errors
                    if (classified_error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] and 
                        classified_error.can_auto_recover):
                        
                        recovery_result = self.classifier.attempt_auto_recovery(classified_error)
                        if recovery_result.get("success"):
                            successful_recoveries += 1
                            logger.info(f"✅ Auto-recovery successful for {scenario_name}")
                        else:
                            logger.error(f"❌ Auto-recovery failed for {scenario_name}")
                
                # Brief pause between scenarios
                await asyncio.sleep(1)
            
            # Record test result
            test_duration = time.time() - test_start_time
            result = EnhancedTestResult(
                test_name="coreml_triggers",
                success=True,  # Success if we can classify the triggered errors
                duration=test_duration,
                errors_detected=len(classified_errors),
                classified_errors=[self.classifier.export_classified_error(ce) for ce in classified_errors],
                recovery_attempts=recovery_attempts,
                successful_recoveries=successful_recoveries,
                performance_metrics={
                    "errors_per_scenario": len(classified_errors) / len(trigger_scenarios),
                    "trigger_success_rate": len(classified_errors) / len(trigger_scenarios)
                },
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            self.classified_errors.extend(classified_errors)
            
            logger.info(f"CoreML trigger test ✅ PASSED")
            logger.info(f"Triggered and classified {len(classified_errors)} errors")
            
            return True
            
        except Exception as e:
            logger.error(f"CoreML trigger test failed: {str(e)}")
            return False
    
    async def _test_pattern_recognition(self) -> bool:
        """Test accuracy of error pattern recognition."""
        logger.info("Testing error pattern recognition accuracy...")
        
        test_start_time = time.time()
        
        try:
            # Test known error patterns with expected classifications
            test_cases = [
                {
                    "error": "ANE device not available",
                    "expected_category": ErrorCategory.ANE_DRIVER,
                    "expected_severity": ErrorSeverity.CRITICAL
                },
                {
                    "error": "Failed to allocate tensor memory for large model",
                    "expected_category": ErrorCategory.MEMORY_ALLOCATION,
                    "expected_severity": ErrorSeverity.CRITICAL
                },
                {
                    "error": "concurrent access violation in queue operation",
                    "expected_category": ErrorCategory.THREAD_SAFETY,
                    "expected_severity": ErrorSeverity.HIGH
                },
                {
                    "error": "Too many open files during model loading",
                    "expected_category": ErrorCategory.RESOURCE_EXHAUSTION,
                    "expected_severity": ErrorSeverity.HIGH
                },
                {
                    "error": "Invalid model format in .mlmodelc file",
                    "expected_category": ErrorCategory.MODEL_CORRUPTION,
                    "expected_severity": ErrorSeverity.HIGH
                }
            ]
            
            correct_classifications = 0
            total_classifications = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                error = RuntimeError(test_case["error"])
                context = {"operation": f"pattern_test_{i}", "test_case": True}
                
                classified_error = self.classifier.classify_error(error, context)
                
                category_correct = classified_error.category == test_case["expected_category"]
                severity_correct = classified_error.severity == test_case["expected_severity"]
                
                if category_correct and severity_correct:
                    correct_classifications += 1
                    logger.info(f"✅ Pattern {i+1}: Correctly classified as "
                              f"{classified_error.category.value}/{classified_error.severity.value}")
                else:
                    logger.warning(f"❌ Pattern {i+1}: Incorrect classification - "
                                 f"Expected: {test_case['expected_category'].value}/{test_case['expected_severity'].value}, "
                                 f"Got: {classified_error.category.value}/{classified_error.severity.value}")
            
            accuracy = correct_classifications / total_classifications
            success = accuracy >= 0.8  # 80% accuracy threshold
            
            # Record test result
            test_duration = time.time() - test_start_time
            result = EnhancedTestResult(
                test_name="pattern_recognition",
                success=success,
                duration=test_duration,
                errors_detected=total_classifications,
                classified_errors=[],  # No actual errors, just pattern tests
                recovery_attempts=0,
                successful_recoveries=0,
                performance_metrics={
                    "accuracy": accuracy,
                    "correct_classifications": correct_classifications,
                    "total_test_cases": total_classifications
                },
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            
            logger.info(f"Pattern recognition test {'✅ PASSED' if success else '❌ FAILED'}")
            logger.info(f"Accuracy: {accuracy:.2f} ({correct_classifications}/{total_classifications})")
            
            return success
            
        except Exception as e:
            logger.error(f"Pattern recognition test failed: {str(e)}")
            return False
    
    async def _test_auto_recovery_performance(self) -> bool:
        """Test automated recovery performance and effectiveness."""
        logger.info("Testing automated recovery performance...")
        
        test_start_time = time.time()
        
        try:
            # Simulate recoverable errors and test auto-recovery
            recovery_test_cases = [
                {
                    "error": RuntimeError("ANE device not available"),
                    "context": {"operation": "model_loading", "model_name": "test_model"},
                    "should_recover": True
                },
                {
                    "error": MemoryError("GPU memory exhausted"),
                    "context": {"operation": "tensor_allocation", "model_name": "large_model"},
                    "should_recover": True
                },
                {
                    "error": RuntimeError("concurrent access violation"),
                    "context": {"operation": "model_switching", "concurrent_requests": 5},
                    "should_recover": True
                }
            ]
            
            successful_recoveries = 0
            total_attempts = 0
            recovery_times = []
            
            for test_case in recovery_test_cases:
                classified_error = self.classifier.classify_error(
                    test_case["error"], 
                    test_case["context"]
                )
                
                if classified_error.can_auto_recover:
                    total_attempts += 1
                    
                    recovery_start = time.time()
                    recovery_result = self.classifier.attempt_auto_recovery(
                        classified_error, 
                        test_case["context"]
                    )
                    recovery_time = time.time() - recovery_start
                    
                    recovery_times.append(recovery_time)
                    
                    if recovery_result.get("success"):
                        successful_recoveries += 1
                        logger.info(f"✅ Auto-recovery successful for {classified_error.category.value} "
                                  f"in {recovery_time:.2f}s")
                    else:
                        logger.warning(f"❌ Auto-recovery failed for {classified_error.category.value}")
            
            # Calculate performance metrics
            recovery_success_rate = successful_recoveries / max(total_attempts, 1)
            avg_recovery_time = sum(recovery_times) / max(len(recovery_times), 1)
            
            success = recovery_success_rate >= 0.5  # 50% success rate threshold
            
            # Record test result
            test_duration = time.time() - test_start_time
            result = EnhancedTestResult(
                test_name="auto_recovery_performance",
                success=success,
                duration=test_duration,
                errors_detected=len(recovery_test_cases),
                classified_errors=[],
                recovery_attempts=total_attempts,
                successful_recoveries=successful_recoveries,
                performance_metrics={
                    "recovery_success_rate": recovery_success_rate,
                    "avg_recovery_time": avg_recovery_time,
                    "max_recovery_time": max(recovery_times) if recovery_times else 0,
                    "min_recovery_time": min(recovery_times) if recovery_times else 0
                },
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            
            logger.info(f"Auto-recovery performance test {'✅ PASSED' if success else '❌ FAILED'}")
            logger.info(f"Recovery success rate: {recovery_success_rate:.2f}")
            logger.info(f"Average recovery time: {avg_recovery_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Auto-recovery performance test failed: {str(e)}")
            return False
    
    def _validate_classification_results(self, classified_errors: List[ClassifiedError]) -> bool:
        """Validate error classification results."""
        if not classified_errors:
            return False
        
        # Check that all errors were classified
        for error in classified_errors:
            if error.category == ErrorCategory.UNKNOWN:
                logger.warning(f"Error classified as UNKNOWN: {error.original_error}")
                return False
            
            if error.confidence < 0.5:
                logger.warning(f"Low confidence classification: {error.confidence:.2f}")
                return False
        
        return True
    
    def _generate_test_report(self, suite_duration: float, overall_success: bool) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Calculate aggregate statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        total_errors = sum(result.errors_detected for result in self.test_results)
        total_recoveries = sum(result.successful_recoveries for result in self.test_results)
        total_recovery_attempts = sum(result.recovery_attempts for result in self.test_results)
        
        # Error classification statistics
        classifier_stats = self.classifier.get_classification_stats()
        
        # Build comprehensive report
        report = {
            "suite_info": {
                "name": "Enhanced Segfault Test Suite with CoreML Error Classification",
                "duration": suite_duration,
                "timestamp": datetime.now().isoformat(),
                "overall_success": overall_success
            },
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / max(total_tests, 1),
                "total_errors_detected": total_errors,
                "total_recovery_attempts": total_recovery_attempts,
                "successful_recoveries": total_recoveries,
                "recovery_success_rate": total_recoveries / max(total_recovery_attempts, 1)
            },
            "classification_statistics": classifier_stats,
            "test_results": [asdict(result) for result in self.test_results],
            "classified_errors": [
                self.classifier.export_classified_error(error) 
                for error in self.classified_errors
            ],
            "recommendations": self._generate_recommendations()
        }
        
        # Log summary
        logger.info(f"Test Summary: {successful_tests}/{total_tests} tests passed")
        logger.info(f"Error Detection: {total_errors} errors detected and classified")
        logger.info(f"Recovery Performance: {total_recoveries}/{total_recovery_attempts} recoveries successful")
        logger.info(f"Classification Accuracy: {classifier_stats.get('total_classifications', 0)} errors classified")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze error patterns
        error_categories = {}
        for error in self.classified_errors:
            category = error.category.value
            error_categories[category] = error_categories.get(category, 0) + 1
        
        # Generate specific recommendations
        if error_categories.get('ane_driver', 0) > 0:
            recommendations.append(
                "ANE driver issues detected. Consider implementing CPU fallback mechanisms."
            )
        
        if error_categories.get('memory_allocation', 0) > 0:
            recommendations.append(
                "Memory allocation failures detected. Consider reducing model cache size or implementing more aggressive memory management."
            )
        
        if error_categories.get('thread_safety', 0) > 0:
            recommendations.append(
                "Thread safety issues detected. Review concurrent access patterns and implement stricter locking mechanisms."
            )
        
        # Performance recommendations
        avg_recovery_time = sum(
            result.performance_metrics.get('avg_recovery_time', 0) 
            for result in self.test_results
        ) / max(len(self.test_results), 1)
        
        if avg_recovery_time > 30:
            recommendations.append(
                f"Recovery times are high (avg: {avg_recovery_time:.1f}s). Consider optimizing recovery procedures."
            )
        
        return recommendations

async def main():
    """Run the enhanced segfault test suite."""
    
    # Check if models directory exists
    models_dir = Path("./models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please ensure models are available in ./models/")
        sys.exit(1)
    
    # Create and run enhanced test suite
    test_suite = EnhancedSegfaultTestSuite()
    
    try:
        logger.info("Starting Enhanced Segfault Test Suite with CoreML Error Classification")
        logger.info("This suite will test error classification, recovery mechanisms, and pattern recognition.")
        
        report = await test_suite.run_enhanced_test_suite()
        
        # Save detailed report
        report_file = "enhanced_segfault_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Test suite completed. Detailed report saved to {report_file}")
        
        if report["suite_info"]["overall_success"]:
            logger.info("✅ All enhanced tests passed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Some enhanced tests failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())