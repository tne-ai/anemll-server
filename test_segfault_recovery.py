#!/usr/bin/env python
"""
Test case for server segmentation fault during model switching.

This test simulates various scenarios that could cause segfaults during model
switching and validates the recovery mechanisms.
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
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SegfaultTestCase:
    """Test case for segmentation fault recovery during model switching."""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 8400):
        self.server_host = server_host
        self.server_port = server_port
        self.base_url = f"http://{server_host}:{server_port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.test_results: Dict[str, any] = {}
        
        # Test configuration
        self.models_to_test = [
            "anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2",
            "anemll-DeepSeekR1-8B-ctx1024_0.2.0"
        ]
        
        # Monitoring
        self.crash_count = 0
        self.recovery_count = 0
        self.successful_switches = 0
        self.failed_switches = 0
        
    async def start_server(self, crash_simulation: bool = False):
        """Start the Anemll server process."""
        logger.info("Starting Anemll server...")
        
        cmd = [sys.executable, "anemll-server-core.py"]
        if crash_simulation:
            # Add environment variables that might trigger edge cases
            env = os.environ.copy()
            env["MALLOC_CHECK_"] = "1"  # Enable malloc debugging
            env["PYTHONMALLOC"] = "debug"  # Enable Python memory debugging
        else:
            env = None
            
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            # Wait for server to start
            await self.wait_for_server_ready()
            logger.info("Server started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            return False
    
    async def wait_for_server_ready(self, timeout: int = 30):
        """Wait for server to be ready to accept requests."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/v1/models") as response:
                        if response.status == 200:
                            logger.info("Server is ready")
                            return True
            except:
                pass
                
            await asyncio.sleep(1)
        
        raise TimeoutError("Server failed to start within timeout")
    
    def is_server_alive(self):
        """Check if server process is still alive."""
        return self.server_process and self.server_process.poll() is None
    
    async def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            try:
                # Try graceful shutdown first
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.server_process.kill()
                    self.server_process.wait()
            except Exception as e:
                logger.error(f"Error stopping server: {str(e)}")
    
    async def make_chat_request(self, model: str, message: str = "Hello! This is a test message.") -> Dict:
        """Make a chat completion request to the server."""
        request_data = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": 50,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "response": result}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                        
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Network error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    async def stress_test_model_switching(self, duration: int = 60, switch_interval: float = 2.0):
        """Perform stress testing of model switching."""
        logger.info(f"Starting stress test for {duration} seconds with {switch_interval}s intervals")
        
        start_time = time.time()
        current_model_idx = 0
        
        while time.time() - start_time < duration:
            if not self.is_server_alive():
                logger.error("Server crashed during stress test!")
                self.crash_count += 1
                
                # Attempt recovery
                logger.info("Attempting server recovery...")
                await self.start_server()
                self.recovery_count += 1
                
                # Continue testing after recovery
                continue
            
            # Switch to next model
            model = self.models_to_test[current_model_idx % len(self.models_to_test)]
            current_model_idx += 1
            
            logger.info(f"Testing model switch to: {model}")
            
            # Make request
            result = await self.make_chat_request(
                model, 
                f"Model switch test #{current_model_idx}. Please respond briefly."
            )
            
            if result["success"]:
                self.successful_switches += 1
                logger.info(f"Model switch successful: {model}")
            else:
                self.failed_switches += 1
                logger.error(f"Model switch failed: {result['error']}")
                
                # Check if failure indicates a crash
                if "Network error" in result["error"] or "Connection" in result["error"]:
                    logger.warning("Network error detected - server may have crashed")
            
            await asyncio.sleep(switch_interval)
    
    async def memory_pressure_test(self):
        """Test model switching under memory pressure."""
        logger.info("Starting memory pressure test...")
        
        # Rapid model switching to stress memory management
        for i in range(10):
            for model in self.models_to_test:
                if not self.is_server_alive():
                    logger.error(f"Server crashed during memory pressure test (iteration {i})")
                    self.crash_count += 1
                    return False
                
                logger.info(f"Memory pressure test - iteration {i+1}, model: {model}")
                result = await self.make_chat_request(
                    model, 
                    "Memory pressure test. Generate a medium-length response to stress memory."
                )
                
                if not result["success"]:
                    logger.error(f"Request failed during memory pressure test: {result['error']}")
                    return False
                
                # No delay between requests to maximize pressure
        
        logger.info("Memory pressure test completed successfully")
        return True
    
    async def concurrent_request_test(self, num_concurrent: int = 5):
        """Test concurrent requests to different models."""
        logger.info(f"Starting concurrent request test with {num_concurrent} concurrent requests")
        
        async def make_concurrent_request(model: str, request_id: int):
            result = await self.make_chat_request(
                model, 
                f"Concurrent request #{request_id} to {model}"
            )
            return request_id, model, result
        
        # Create concurrent requests alternating between models
        tasks = []
        for i in range(num_concurrent):
            model = self.models_to_test[i % len(self.models_to_test)]
            task = make_concurrent_request(model, i)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Concurrent request failed with exception: {str(result)}")
                else:
                    request_id, model, response = result
                    if response["success"]:
                        success_count += 1
                    else:
                        logger.error(f"Concurrent request {request_id} to {model} failed: {response['error']}")
            
            logger.info(f"Concurrent test completed: {success_count}/{num_concurrent} successful")
            return success_count == num_concurrent
            
        except Exception as e:
            logger.error(f"Concurrent request test failed: {str(e)}")
            return False
    
    async def monitor_server_health(self):
        """Monitor server health during tests."""
        if not self.server_process:
            return
        
        try:
            process = psutil.Process(self.server_process.pid)
            
            # Get memory and CPU usage
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            logger.info(f"Server health - Memory: {memory_info.rss / 1024 / 1024:.1f} MB, CPU: {cpu_percent:.1f}%")
            
            # Check for high memory usage (potential memory leak)
            if memory_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
                logger.warning("High memory usage detected - potential memory leak")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning("Could not monitor server health - process may have crashed")
    
    async def run_segfault_tests(self):
        """Run the complete segfault test suite."""
        logger.info("=" * 80)
        logger.info("STARTING SEGMENTATION FAULT RECOVERY TEST SUITE")
        logger.info("=" * 80)
        
        test_start_time = time.time()
        
        try:
            # Test 1: Basic server startup and model availability
            logger.info("\n--- Test 1: Basic Server Startup ---")
            if not await self.start_server():
                logger.error("FAILED: Could not start server")
                return False
            
            # Verify models are available
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model["id"] for model in models_data["data"]]
                        logger.info(f"Available models: {available_models}")
                        
                        # Update test models based on what's actually available
                        self.models_to_test = [model for model in self.models_to_test if model in available_models]
                        
                        if len(self.models_to_test) < 2:
                            logger.error("FAILED: Need at least 2 models for switching tests")
                            return False
                    else:
                        logger.error("FAILED: Could not retrieve model list")
                        return False
            
            # Test 2: Memory pressure test
            logger.info("\n--- Test 2: Memory Pressure Test ---")
            if await self.memory_pressure_test():
                logger.info("PASSED: Memory pressure test")
            else:
                logger.warning("FAILED: Memory pressure test")
            
            # Test 3: Concurrent request test
            logger.info("\n--- Test 3: Concurrent Request Test ---")
            if await self.concurrent_request_test():
                logger.info("PASSED: Concurrent request test")
            else:
                logger.warning("FAILED: Concurrent request test")
            
            # Test 4: Stress test with model switching
            logger.info("\n--- Test 4: Stress Test with Model Switching ---")
            
            # Monitor health during stress test
            health_monitor_task = asyncio.create_task(self._periodic_health_monitor())
            
            await self.stress_test_model_switching(duration=30, switch_interval=1.0)
            
            # Stop health monitoring
            health_monitor_task.cancel()
            
            logger.info("COMPLETED: Stress test with model switching")
            
        except Exception as e:
            logger.error(f"Test suite failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await self.stop_server()
        
        # Print test results
        total_time = time.time() - test_start_time
        logger.info("\n" + "=" * 80)
        logger.info("SEGMENTATION FAULT RECOVERY TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total test duration: {total_time:.2f} seconds")
        logger.info(f"Server crashes detected: {self.crash_count}")
        logger.info(f"Successful recoveries: {self.recovery_count}")
        logger.info(f"Successful model switches: {self.successful_switches}")
        logger.info(f"Failed model switches: {self.failed_switches}")
        
        if self.crash_count > 0:
            logger.warning(f"⚠️  SEGFAULTS DETECTED: {self.crash_count} server crashes occurred")
            logger.info(f"✅ RECOVERY SUCCESS: {self.recovery_count}/{self.crash_count} recoveries successful")
        else:
            logger.info("✅ NO CRASHES: No segmentation faults detected during testing")
        
        success_rate = (self.successful_switches / (self.successful_switches + self.failed_switches) * 100) if (self.successful_switches + self.failed_switches) > 0 else 0
        logger.info(f"Model switch success rate: {success_rate:.1f}%")
        
        return True
    
    async def _periodic_health_monitor(self):
        """Periodically monitor server health."""
        try:
            while True:
                await self.monitor_server_health()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass


async def main():
    """Main function to run the segfault test."""
    
    # Check if models directory exists
    models_dir = Path("./models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please ensure models are available in ./models/")
        sys.exit(1)
    
    # Run the test
    test_case = SegfaultTestCase()
    success = await test_case.run_segfault_tests()
    
    if success:
        logger.info("✅ Segfault recovery test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Segfault recovery test failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)