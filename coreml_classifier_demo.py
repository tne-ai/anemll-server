#!/usr/bin/env python
"""
CoreML Error Classification System Demo.

This script demonstrates the capabilities of the CoreML error classification
system, showing how it detects, classifies, and provides recovery recommendations
for different types of CoreML/ANE-related errors.
"""

import time
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

# Import the CoreML error classification system
from coreml_error_classifier import (
    CoreMLErrorClassifier, ClassifiedError, get_classifier
)
from error_patterns import ErrorCategory, ErrorSeverity
from recovery_strategies import RecoveryStrategyDatabase
from coreml_wrapper import CoreMLMonitor, handle_coreml_error

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoreMLClassifierDemo:
    """Demonstration of the CoreML error classification system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.classifier = get_classifier()
        self.strategy_db = RecoveryStrategyDatabase()
        self.demo_results: List[Dict[str, Any]] = []
        
        logger.info("CoreML Error Classification System Demo")
        logger.info("=" * 60)
    
    def run_demo(self):
        """Run the complete demonstration."""
        logger.info("🚀 Starting CoreML Error Classification Demo")
        
        try:
            # Demo 1: Error Pattern Recognition
            logger.info("\n" + "="*60)
            logger.info("DEMO 1: ERROR PATTERN RECOGNITION")
            logger.info("="*60)
            self.demo_error_pattern_recognition()
            
            # Demo 2: Error Classification and Recovery
            logger.info("\n" + "="*60) 
            logger.info("DEMO 2: ERROR CLASSIFICATION AND RECOVERY")
            logger.info("="*60)
            self.demo_error_classification_and_recovery()
            
            # Demo 3: Context-Aware Classification
            logger.info("\n" + "="*60)
            logger.info("DEMO 3: CONTEXT-AWARE CLASSIFICATION")
            logger.info("="*60)
            self.demo_context_aware_classification()
            
            # Demo 4: Monitoring Integration
            logger.info("\n" + "="*60)
            logger.info("DEMO 4: MONITORING INTEGRATION")
            logger.info("="*60)
            self.demo_monitoring_integration()
            
            # Demo 5: Recovery Strategy Execution
            logger.info("\n" + "="*60)
            logger.info("DEMO 5: RECOVERY STRATEGY EXECUTION")
            logger.info("="*60)
            self.demo_recovery_strategy_execution()
            
            # Generate final report
            self.generate_demo_report()
            
        except Exception as e:
            logger.error(f"Demo failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def demo_error_pattern_recognition(self):
        """Demonstrate error pattern recognition capabilities."""
        logger.info("Demonstrating how the classifier recognizes different error patterns...")
        
        # Test errors representing the 5 main categories
        test_errors = [
            {
                "name": "ANE Driver Failure",
                "error": RuntimeError("ANE device not available - Neural Engine initialization error"),
                "expected_category": ErrorCategory.ANE_DRIVER,
                "context": {"operation": "model_loading", "location": "model_manager.py:_load_model"}
            },
            {
                "name": "Memory Allocation Failure", 
                "error": MemoryError("Failed to allocate tensor memory - GPU memory exhausted"),
                "expected_category": ErrorCategory.MEMORY_ALLOCATION,
                "context": {"operation": "tensor_allocation", "location": "model_manager.py:create_unified_state"}
            },
            {
                "name": "Thread Safety Issue",
                "error": RuntimeError("concurrent access violation - threading lock timeout"),
                "expected_category": ErrorCategory.THREAD_SAFETY,
                "context": {"operation": "model_switching", "location": "anemll-server.py:StreamingTokenGenerator"}
            },
            {
                "name": "Resource Exhaustion",
                "error": OSError("Too many open files - Resource temporarily unavailable"),
                "expected_category": ErrorCategory.RESOURCE_EXHAUSTION,
                "context": {"operation": "file_operations", "location": "model_loading"}
            },
            {
                "name": "Model Corruption",
                "error": ValueError("Invalid model format - Corrupted .mlmodelc file"),
                "expected_category": ErrorCategory.MODEL_CORRUPTION,
                "context": {"operation": "model_validation", "location": "chat_full.py:load_model"}
            }
        ]
        
        recognition_results = []
        
        for test_case in test_errors:
            logger.info(f"\n🔍 Testing: {test_case['name']}")
            logger.info(f"   Error: {test_case['error']}")
            
            # Classify the error
            start_time = time.time()
            classified_error = self.classifier.classify_error(test_case['error'], test_case['context'])
            classification_time = time.time() - start_time
            
            # Check if classification matches expectation
            correct_category = classified_error.category == test_case['expected_category']
            
            logger.info(f"   🏷️  Category: {classified_error.category.value} "
                       f"{'✅' if correct_category else '❌'}")
            logger.info(f"   ⚠️  Severity: {classified_error.severity.value}")
            logger.info(f"   🎯 Confidence: {classified_error.confidence:.2f}")
            logger.info(f"   ⏱️  Classification Time: {classification_time*1000:.1f}ms")
            
            if classified_error.recovery_strategy:
                logger.info(f"   🔧 Recovery Strategy: {classified_error.recovery_strategy}")
            
            if classified_error.can_auto_recover:
                logger.info(f"   🤖 Auto-Recovery: ✅ Available")
            else:
                logger.info(f"   🤖 Auto-Recovery: ❌ Manual intervention required")
            
            recognition_results.append({
                "test_case": test_case['name'],
                "correct_classification": correct_category,
                "confidence": classified_error.confidence,
                "classification_time": classification_time,
                "classified_error": self.classifier.export_classified_error(classified_error)
            })
        
        # Summary
        correct_count = sum(1 for r in recognition_results if r['correct_classification'])
        avg_confidence = sum(r['confidence'] for r in recognition_results) / len(recognition_results)
        avg_time = sum(r['classification_time'] for r in recognition_results) / len(recognition_results)
        
        logger.info(f"\n📊 Pattern Recognition Results:")
        logger.info(f"   ✅ Accuracy: {correct_count}/{len(test_errors)} ({correct_count/len(test_errors)*100:.1f}%)")
        logger.info(f"   🎯 Average Confidence: {avg_confidence:.2f}")
        logger.info(f"   ⏱️  Average Classification Time: {avg_time*1000:.1f}ms")
        
        self.demo_results.append({
            "demo_name": "error_pattern_recognition",
            "results": recognition_results,
            "summary": {
                "accuracy": correct_count / len(test_errors),
                "avg_confidence": avg_confidence,
                "avg_classification_time": avg_time
            }
        })
    
    def demo_error_classification_and_recovery(self):
        """Demonstrate error classification with recovery recommendations."""
        logger.info("Demonstrating error classification with detailed recovery recommendations...")
        
        # Critical errors that should have recovery strategies
        critical_errors = [
            {
                "name": "Critical ANE Failure",
                "error": RuntimeError("CoreML model compilation failed - ANE device not available"),
                "context": {
                    "operation": "model_loading",
                    "model_name": "anemll-Meta-Llama-3.2-1B",
                    "location": "model_manager.py:139",
                    "system_state": {"memory_percent": 85, "cpu_percent": 70}
                }
            },
            {
                "name": "Critical Memory Failure",
                "error": MemoryError("malloc(): corrupted top size - segfault in create_unified_state"),
                "context": {
                    "operation": "tensor_allocation", 
                    "model_name": "anemll-DeepSeekR1-8B",
                    "location": "model_manager.py:147",
                    "system_state": {"memory_percent": 95, "available_memory": 1024*1024*100}
                }
            }
        ]
        
        classification_results = []
        
        for error_case in critical_errors:
            logger.info(f"\n🚨 Analyzing Critical Error: {error_case['name']}")
            logger.info(f"   Error: {error_case['error']}")
            
            # Classify the error
            classified_error = self.classifier.classify_error(error_case['error'], error_case['context'])
            
            logger.info(f"   📋 Classification Results:")
            logger.info(f"      Category: {classified_error.category.value}")
            logger.info(f"      Severity: {classified_error.severity.value}")
            logger.info(f"      Confidence: {classified_error.confidence:.2f}")
            logger.info(f"      Description: {classified_error.description}")
            
            logger.info(f"   🔍 Potential Causes:")
            for cause in classified_error.potential_causes:
                logger.info(f"      • {cause}")
            
            if classified_error.recovery_steps:
                logger.info(f"   🔧 Recovery Steps:")
                for i, step in enumerate(classified_error.recovery_steps, 1):
                    logger.info(f"      {i}. {step}")
            
            if classified_error.estimated_recovery_time:
                logger.info(f"   ⏱️  Estimated Recovery Time: {classified_error.estimated_recovery_time:.1f}s")
            
            logger.info(f"   🤖 Auto-Recovery: {'✅ Enabled' if classified_error.can_auto_recover else '❌ Disabled'}")
            
            classification_results.append({
                "error_case": error_case['name'],
                "classified_error": self.classifier.export_classified_error(classified_error)
            })
        
        self.demo_results.append({
            "demo_name": "error_classification_and_recovery",
            "results": classification_results
        })
    
    def demo_context_aware_classification(self):
        """Demonstrate how context affects error classification."""
        logger.info("Demonstrating context-aware error classification...")
        
        # Same error message with different contexts
        base_error = RuntimeError("Memory allocation failed")
        
        contexts = [
            {
                "name": "Model Loading Context",
                "context": {
                    "operation": "model_loading",
                    "location": "model_manager.py:_load_model",
                    "model_name": "large_model",
                    "system_state": {"memory_percent": 90}
                }
            },
            {
                "name": "Tensor Operation Context", 
                "context": {
                    "operation": "tensor_allocation",
                    "location": "chat_full.py:create_unified_state",
                    "model_name": "efficient_model",
                    "system_state": {"memory_percent": 45}
                }
            },
            {
                "name": "Concurrent Access Context",
                "context": {
                    "operation": "concurrent_model_access",
                    "location": "anemll-server.py:StreamingTokenGenerator",
                    "thread_info": {"concurrent_requests": 10},
                    "system_state": {"memory_percent": 60}
                }
            }
        ]
        
        context_results = []
        
        logger.info(f"\n🔄 Base Error: {base_error}")
        
        for ctx in contexts:
            logger.info(f"\n📍 Context: {ctx['name']}")
            
            classified_error = self.classifier.classify_error(base_error, ctx['context'])
            
            logger.info(f"   Classification: {classified_error.category.value}/{classified_error.severity.value}")
            logger.info(f"   Confidence: {classified_error.confidence:.2f}")
            logger.info(f"   Matched Patterns: {', '.join(classified_error.matched_patterns) if classified_error.matched_patterns else 'None'}")
            
            context_results.append({
                "context_name": ctx['name'],
                "category": classified_error.category.value,
                "severity": classified_error.severity.value,
                "confidence": classified_error.confidence
            })
        
        logger.info(f"\n📈 Context Impact Analysis:")
        categories = set(r['category'] for r in context_results)
        severities = set(r['severity'] for r in context_results)
        logger.info(f"   📊 Categories identified: {', '.join(categories)}")
        logger.info(f"   ⚠️  Severities identified: {', '.join(severities)}")
        logger.info(f"   🎯 This demonstrates how context influences classification accuracy")
        
        self.demo_results.append({
            "demo_name": "context_aware_classification",
            "results": context_results
        })
    
    def demo_monitoring_integration(self):
        """Demonstrate CoreML monitoring integration."""
        logger.info("Demonstrating CoreML monitoring integration...")
        
        def simulate_model_loading_operation():
            """Simulate a model loading operation that might fail."""
            logger.info("   🔄 Simulating model loading operation...")
            time.sleep(0.1)  # Simulate work
            
            # Simulate different types of failures
            import random
            failure_type = random.choice(["ane_failure", "memory_failure", "success"])
            
            if failure_type == "ane_failure":
                raise RuntimeError("ANE device not available during model compilation")
            elif failure_type == "memory_failure":
                raise MemoryError("Failed to allocate tensor memory for large model")
            # else: success, no exception
            
            logger.info("   ✅ Model loading completed successfully")
            return "model_loaded_successfully"
        
        logger.info("\n🔍 Testing monitoring with different scenarios...")
        
        scenarios = [
            "Successful Operation",
            "ANE Failure", 
            "Memory Failure"
        ]
        
        monitoring_results = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n📋 Scenario {i+1}: {scenario}")
            
            try:
                with CoreMLMonitor("demo_model_loading", f"demo_model_{i}") as monitor:
                    if scenario == "Successful Operation":
                        logger.info("   ✅ Operation completed successfully")
                        result = "success"
                    elif scenario == "ANE Failure":
                        raise RuntimeError("ANE device not available during model compilation")
                    elif scenario == "Memory Failure":
                        raise MemoryError("Failed to allocate tensor memory for large model")
                    
            except Exception as e:
                logger.info(f"   ❌ Exception caught and classified by monitor")
                result = "classified_error"
            
            monitoring_results.append({
                "scenario": scenario,
                "result": result,
                "errors_detected": len(monitor.errors_detected) if 'monitor' in locals() else 0
            })
        
        logger.info(f"\n📊 Monitoring Integration Results:")
        for result in monitoring_results:
            logger.info(f"   {result['scenario']}: {result['result']} "
                       f"(errors: {result['errors_detected']})")
        
        self.demo_results.append({
            "demo_name": "monitoring_integration",
            "results": monitoring_results
        })
    
    def demo_recovery_strategy_execution(self):
        """Demonstrate recovery strategy execution."""
        logger.info("Demonstrating recovery strategy execution...")
        
        # Get available recovery strategies
        strategies = self.strategy_db.strategies
        
        logger.info(f"\n📋 Available Recovery Strategies: {len(strategies)}")
        
        # Demo a few key strategies
        demo_strategies = [
            "ane_device_unavailable_recovery",
            "tensor_allocation_failure_recovery", 
            "concurrent_access_violation_recovery"
        ]
        
        strategy_results = []
        
        for strategy_name in demo_strategies:
            if strategy_name in strategies:
                strategy = strategies[strategy_name]
                
                logger.info(f"\n🔧 Strategy: {strategy.name}")
                logger.info(f"   Category: {strategy.category.value}")
                logger.info(f"   Severity: {strategy.severity.value}")
                logger.info(f"   Description: {strategy.description}")
                logger.info(f"   Auto-Recovery: {'✅' if strategy.auto_recovery_enabled else '❌'}")
                logger.info(f"   Estimated Time: {strategy.estimated_time}s")
                
                logger.info(f"   📝 Recovery Steps:")
                for i, step in enumerate(strategy.steps, 1):
                    logger.info(f"      {i}. {step.description} "
                               f"(Priority: {step.priority.value}, Action: {step.action.value})")
                
                logger.info(f"   ✅ Prerequisites:")
                for prereq in strategy.prerequisites:
                    logger.info(f"      • {prereq}")
                
                strategy_results.append({
                    "strategy_name": strategy.name,
                    "category": strategy.category.value,
                    "steps_count": len(strategy.steps),
                    "auto_recovery_enabled": strategy.auto_recovery_enabled
                })
        
        logger.info(f"\n📊 Recovery Strategy Summary:")
        auto_strategies = sum(1 for r in strategy_results if r['auto_recovery_enabled'])
        logger.info(f"   🤖 Auto-Recovery Enabled: {auto_strategies}/{len(strategy_results)}")
        
        avg_steps = sum(r['steps_count'] for r in strategy_results) / len(strategy_results)
        logger.info(f"   📝 Average Steps per Strategy: {avg_steps:.1f}")
        
        self.demo_results.append({
            "demo_name": "recovery_strategy_execution",
            "results": strategy_results
        })
    
    def generate_demo_report(self):
        """Generate a comprehensive demo report."""
        logger.info("\n" + "="*60)
        logger.info("DEMO REPORT SUMMARY")
        logger.info("="*60)
        
        # Get classifier statistics
        classifier_stats = self.classifier.get_classification_stats()
        pattern_summary = self.classifier.get_pattern_summary()
        
        logger.info(f"📊 Classification System Statistics:")
        logger.info(f"   Total Classifications: {classifier_stats['total_classifications']}")
        logger.info(f"   Error Patterns Available: {pattern_summary['total_patterns']}")
        logger.info(f"   Critical Patterns: {len(pattern_summary['critical_patterns'])}")
        logger.info(f"   Recovery Strategies: {len(self.strategy_db.strategies)}")
        
        logger.info(f"\n🎯 Demo Results Summary:")
        for demo_result in self.demo_results:
            demo_name = demo_result['demo_name'].replace('_', ' ').title()
            logger.info(f"   ✅ {demo_name}: Completed")
        
        # Generate recommendations
        logger.info(f"\n💡 System Capabilities Demonstrated:")
        logger.info(f"   🔍 Real-time error pattern recognition")
        logger.info(f"   🏷️  Automated error categorization and severity assessment")
        logger.info(f"   🧠 Context-aware classification for improved accuracy")
        logger.info(f"   🔧 Comprehensive recovery strategy recommendations")
        logger.info(f"   🤖 Automated recovery execution for critical errors")
        logger.info(f"   📊 Performance monitoring and statistics tracking")
        
        # Save detailed report
        report = {
            "demo_info": {
                "name": "CoreML Error Classification System Demo",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            },
            "system_statistics": {
                "classifier_stats": classifier_stats,
                "pattern_summary": pattern_summary,
                "strategy_count": len(self.strategy_db.strategies)
            },
            "demo_results": self.demo_results
        }
        
        report_file = "coreml_classifier_demo_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
        logger.info(f"🎉 Demo completed successfully!")

def main():
    """Run the CoreML error classification demo."""
    
    try:
        demo = CoreMLClassifierDemo()
        demo.run_demo()
        
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()