#!/usr/bin/env python
"""
Focused test to trigger segmentation fault conditions during model switching.

This script specifically targets the most likely segfault scenarios:
1. Memory corruption during CoreML model loading
2. Concurrent access race conditions
3. Resource exhaustion during rapid switching
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
import logging
import threading
import os
import signal
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegfaultTrigger:
    """Trigger specific scenarios that may cause segfaults."""
    
    def __init__(self, server_url: str = "http://localhost:8400"):
        self.server_url = server_url
        self.models = [
            "anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2",
            "anemll-DeepSeekR1-8B-ctx1024_0.2.0"
        ]
        
    async def rapid_model_switching(self, switches: int = 20, delay: float = 0.1):
        """
        Trigger rapid model switching to stress the LRU cache and model loading.
        This targets memory corruption during frequent CoreML model loading/unloading.
        """
        logger.info(f"🔥 TRIGGERING: Rapid model switching ({switches} switches, {delay}s delay)")
        
        for i in range(switches):
            model = self.models[i % len(self.models)]
            
            try:
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "model": model,
                        "messages": [{"role": "user", "content": f"Switch #{i} to {model}"}],
                        "max_tokens": 10,
                        "stream": False
                    }
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        duration = time.time() - start_time
                        
                        if response.status == 200:
                            logger.info(f"✅ Switch {i+1}/{switches}: {model} ({duration:.2f}s)")
                        else:
                            error = await response.text()
                            logger.error(f"❌ Switch {i+1}/{switches}: HTTP {response.status} - {error}")
                            
            except Exception as e:
                logger.error(f"💥 Switch {i+1}/{switches} FAILED: {str(e)}")
                # Continue despite errors to maximize stress
                
            await asyncio.sleep(delay)
    
    async def concurrent_model_access(self, concurrent_requests: int = 10):
        """
        Trigger concurrent access to different models simultaneously.
        This targets thread safety issues and race conditions.
        """
        logger.info(f"🔥 TRIGGERING: Concurrent model access ({concurrent_requests} concurrent requests)")
        
        async def make_request(request_id: int, model: str):
            try:
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "model": model,
                        "messages": [{"role": "user", "content": f"Concurrent request #{request_id}"}],
                        "max_tokens": 15,
                        "stream": False
                    }
                    
                    async with session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ Concurrent {request_id}: {model} SUCCESS")
                            return True
                        else:
                            error = await response.text()
                            logger.error(f"❌ Concurrent {request_id}: {model} HTTP {response.status}")
                            return False
                            
            except Exception as e:
                logger.error(f"💥 Concurrent {request_id}: {model} EXCEPTION - {str(e)}")
                return False
        
        # Create concurrent requests alternating between models
        tasks = []
        for i in range(concurrent_requests):
            model = self.models[i % len(self.models)]
            task = make_request(i, model)
            tasks.append(task)
        
        # Launch all requests simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"Concurrent test result: {success_count}/{concurrent_requests} successful")
        
        return success_count
    
    async def memory_exhaustion_attack(self, large_requests: int = 5):
        """
        Trigger memory exhaustion by making requests with large context.
        This targets memory corruption during large tensor operations.
        """
        logger.info(f"🔥 TRIGGERING: Memory exhaustion attack ({large_requests} large requests)")
        
        # Create a very large message to stress memory allocation
        large_message = "This is a large message. " * 500  # ~10KB per message
        
        for i in range(large_requests):
            model = self.models[i % len(self.models)]
            
            try:
                # Build a conversation with multiple large messages
                messages = []
                for j in range(5):  # Multiple turns to increase context size
                    messages.append({"role": "user", "content": f"{large_message} Turn {j}"})
                    messages.append({"role": "assistant", "content": f"Response to turn {j}. {large_message}"})
                
                # Final user message
                messages.append({"role": "user", "content": f"Final large request #{i}"})
                
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 100,
                        "stream": False
                    }
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        duration = time.time() - start_time
                        
                        if response.status == 200:
                            logger.info(f"✅ Large request {i+1}/{large_requests}: {model} ({duration:.2f}s)")
                        else:
                            error = await response.text()
                            logger.error(f"❌ Large request {i+1}/{large_requests}: HTTP {response.status}")
                            logger.error(f"Error details: {error[:200]}...")
                            
            except Exception as e:
                logger.error(f"💥 Large request {i+1}/{large_requests} EXCEPTION: {str(e)}")
    
    async def cache_thrashing(self, iterations: int = 15):
        """
        Trigger LRU cache thrashing by rapidly switching between models.
        This targets memory management issues during cache eviction.
        """
        logger.info(f"🔥 TRIGGERING: Cache thrashing ({iterations} iterations)")
        
        # Assuming cache size is 3, we'll switch between models to force evictions
        for i in range(iterations):
            model = self.models[i % len(self.models)]
            
            try:
                # Force a cache miss by requesting each model alternately
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "model": model,
                        "messages": [{"role": "user", "content": f"Cache thrash #{i}"}],
                        "max_tokens": 5,
                        "stream": False
                    }
                    
                    async with session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=45)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"✅ Cache thrash {i+1}/{iterations}: {model}")
                        else:
                            error = await response.text()
                            logger.error(f"❌ Cache thrash {i+1}/{iterations}: HTTP {response.status}")
                            
            except Exception as e:
                logger.error(f"💥 Cache thrash {i+1}/{iterations} EXCEPTION: {str(e)}")
            
            # Very short delay to maximize cache pressure
            await asyncio.sleep(0.05)
    
    async def interrupt_during_loading(self):
        """
        Trigger requests and then immediately interrupt to test cleanup.
        This targets resource cleanup issues during model loading.
        """
        logger.info("🔥 TRIGGERING: Interrupt during loading")
        
        for model in self.models:
            try:
                logger.info(f"Starting request to {model} then cancelling...")
                
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "model": model,
                        "messages": [{"role": "user", "content": "This will be interrupted"}],
                        "max_tokens": 50,
                        "stream": True  # Use streaming to have a longer-running request
                    }
                    
                    # Start the request
                    response = await session.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=1)  # Very short timeout to force interruption
                    )
                    
                    # Try to read a little bit then let it timeout
                    try:
                        async for chunk in response.content.iter_chunked(1024):
                            logger.info(f"Received chunk, now cancelling...")
                            break
                    except asyncio.TimeoutError:
                        logger.info(f"Request to {model} interrupted as expected")
                    
            except Exception as e:
                logger.info(f"Request to {model} interrupted: {str(e)}")
    
    async def run_all_triggers(self):
        """Run all segfault trigger scenarios."""
        logger.info("🚨 STARTING SEGFAULT TRIGGER TEST SUITE 🚨")
        logger.info("=" * 60)
        
        # Check server health first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/v1/models") as response:
                    if response.status != 200:
                        logger.error("Server not responding properly")
                        return False
        except Exception as e:
            logger.error(f"Cannot connect to server: {str(e)}")
            return False
        
        logger.info("Server is responding, starting trigger tests...")
        
        # Run each trigger scenario
        scenarios = [
            ("Cache Thrashing", lambda: self.cache_thrashing(iterations=20)),
            ("Rapid Model Switching", lambda: self.rapid_model_switching(switches=15, delay=0.1)),
            ("Concurrent Model Access", lambda: self.concurrent_model_access(concurrent_requests=8)),
            ("Interrupt During Loading", lambda: self.interrupt_during_loading()),
            ("Memory Exhaustion", lambda: self.memory_exhaustion_attack(large_requests=3)),
        ]
        
        for scenario_name, scenario_func in scenarios:
            logger.info(f"\n--- {scenario_name} ---")
            try:
                await scenario_func()
                logger.info(f"✅ {scenario_name} completed")
            except Exception as e:
                logger.error(f"💥 {scenario_name} TRIGGERED EXCEPTION: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Brief pause between scenarios
            await asyncio.sleep(2)
        
        logger.info("\n" + "=" * 60)
        logger.info("🚨 SEGFAULT TRIGGER TEST SUITE COMPLETED 🚨")
        logger.info("Check server logs and process status for any crashes.")
        
        return True


async def main():
    """Run the segfault trigger tests."""
    
    # Check if server is running
    trigger = SegfaultTrigger()
    
    logger.info("Starting segfault trigger tests...")
    logger.info("⚠️  WARNING: These tests are designed to stress the server and may cause crashes!")
    logger.info("⚠️  Monitor the server process for segmentation faults.")
    
    try:
        success = await trigger.run_all_triggers()
        if success:
            logger.info("✅ All trigger tests completed")
        else:
            logger.error("❌ Trigger tests failed")
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Tests failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())