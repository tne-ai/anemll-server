#!/usr/bin/env python
"""
Anemll Server v2 with Automatic Restart on Segfault

This is the enhanced version of the anemll server that provides automatic 
restart capabilities when segmentation faults occur, with RSS memory monitoring.
"""

import sys
import subprocess
import time
import logging
import signal
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('anemll_server_v2_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

class AnemllServerV2Monitor:
    """Monitor and restart the anemll server on segfaults with enhanced logging."""
    
    def __init__(self, max_restarts=50, restart_delay=3.0):
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.process = None
        self.running = True
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping Anemll Server v2 monitor...")
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)
    
    def start_server(self):
        """Start the anemll-server-core.py process."""
        command = [sys.executable, "anemll-server-core.py"]

        try:
            logger.info(f"Starting anemll-server-core.py (attempt {self.restart_count + 1})")
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start anemll-server-core.py: {str(e)}")
            return False
    
    def monitor_and_restart(self):
        """Monitor server and restart on crashes."""
        # Install signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 80)
        logger.info("ANEMLL SERVER v2 MONITOR WITH AUTO-RESTART & RSS MEMORY TRACKING")
        logger.info("=" * 80)
        logger.info(f"Monitoring process: anemll-server-core.py")
        logger.info(f"Max restarts: {self.max_restarts}")
        logger.info(f"Restart delay: {self.restart_delay}s")
        logger.info("Enhanced with RSS memory logging after each completion")
        logger.info("Press Ctrl+C to stop monitoring")
        logger.info("=" * 80)
        
        while self.running and self.restart_count <= self.max_restarts:
            if not self.start_server():
                logger.error("Failed to start anemll-server-core.py, waiting before retry...")
                time.sleep(self.restart_delay)
                self.restart_count += 1
                continue
            
            # Monitor process output and wait for completion
            try:
                logger.info(f"anemll-server-core.py started with PID: {self.process.pid}")
                
                # Stream output in real-time
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        break
                    print(line.rstrip())
                    
                    # Check if server is ready
                    if "Uvicorn running on" in line:
                        logger.info("✅ Anemll Server v2 is ready and accepting connections")
                        logger.info("🔍 RSS memory monitoring active - watch for memory patterns before crashes")
                
                # Wait for process to complete
                return_code = self.process.wait()
                
                if not self.running:
                    break
                
                if return_code == -11:  # SIGSEGV
                    logger.error("🔥 SEGMENTATION FAULT DETECTED IN ANEMLL-SERVER-CORE.PY!")
                    logger.error(f"anemll-server-core.py crashed with segfault (exit code: {return_code})")
                    self.restart_count += 1
                    
                    if self.restart_count <= self.max_restarts:
                        logger.info(f"🔄 RESTARTING ANEMLL-SERVER.PY in {self.restart_delay} seconds...")
                        logger.info(f"Restart attempt: {self.restart_count}/{self.max_restarts}")
                        time.sleep(self.restart_delay)
                    else:
                        logger.error(f"❌ Maximum restart attempts ({self.max_restarts}) reached for anemll-server-core.py")
                        break
                        
                elif return_code == 0:
                    logger.info("anemll-server-core.py exited normally")
                    break
                else:
                    logger.error(f"anemll-server-core.py exited with unexpected code: {return_code}")
                    self.restart_count += 1
                    
                    if self.restart_count <= self.max_restarts:
                        logger.info(f"Restarting anemll-server-core.py in {self.restart_delay} seconds...")
                        time.sleep(self.restart_delay)
                    else:
                        break
                        
            except KeyboardInterrupt:
                logger.info("Anemll Server v2 monitor interrupted by user")
                self.running = False
                if self.process:
                    self.process.terminate()
                break
            except Exception as e:
                logger.error(f"Error monitoring anemll-server-core.py: {str(e)}")
                self.restart_count += 1
                time.sleep(self.restart_delay)
        
        logger.info("=" * 80)
        logger.info("ANEMLL SERVER v2 MONITORING COMPLETED")
        logger.info(f"Total restarts of anemll-server-core.py: {self.restart_count}")
        logger.info("=" * 80)

def main():
    """Main function for Anemll Server v2."""
    print("🚀 Anemll Server v2 - Enhanced with Segfault Recovery & RSS Memory Monitoring")
    print("=" * 80)
    
    # Check if server file exists
    if not os.path.exists("anemll-server-core.py"):
        logger.error("anemll-server-core.py not found in current directory")
        logger.error("Please ensure anemll-server-core.py is in the same directory as anemll-server.py")
        sys.exit(1)
    
    # Create and start monitor
    monitor = AnemllServerV2Monitor(max_restarts=50, restart_delay=3.0)
    
    try:
        monitor.monitor_and_restart()
    except Exception as e:
        logger.error(f"Anemll Server v2 monitor failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()