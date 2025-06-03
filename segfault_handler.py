#!/usr/bin/env python
"""
Segmentation Fault Handler for Anemll Server

This module provides signal handling to catch segmentation faults (SIGSEGV)
and implement graceful recovery mechanisms.
"""

import signal
import sys
import os
import traceback
import logging
import time
import subprocess
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class SegmentationFaultHandler:
    """Handle segmentation faults with graceful recovery."""
    
    def __init__(self, recovery_callback: Optional[Callable] = None):
        """
        Initialize segmentation fault handler.
        
        Args:
            recovery_callback: Optional callback function to execute before exit
        """
        self.recovery_callback = recovery_callback
        self.fault_count = 0
        self.last_fault_time = 0
        
    def install_handler(self):
        """Install the signal handler for SIGSEGV."""
        signal.signal(signal.SIGSEGV, self._segfault_handler)
        logger.info("Segmentation fault handler installed")
        
    def _segfault_handler(self, signum, frame):
        """
        Handle segmentation fault signal with minimal operations.
        
        Args:
            signum: Signal number (should be SIGSEGV)
            frame: Current stack frame
        """
        # Write directly to stderr to avoid logging recursion issues
        try:
            sys.stderr.write(f"\n=== SEGMENTATION FAULT DETECTED ===\n")
            sys.stderr.write(f"Signal: {signum} (SIGSEGV)\n")
            sys.stderr.write(f"Process PID: {os.getpid()}\n")
            sys.stderr.flush()
        except:
            pass
        
        # Execute minimal recovery callback if provided
        if self.recovery_callback:
            try:
                self.recovery_callback()
            except:
                pass
        
        # Force exit immediately (can't continue after segfault)
        os._exit(1)

class ProcessMonitor:
    """Monitor and restart processes that crash with segfaults."""
    
    def __init__(self, command: list, max_restarts: int = 5, restart_delay: float = 2.0):
        """
        Initialize process monitor.
        
        Args:
            command: Command to execute as a list
            max_restarts: Maximum number of restart attempts
            restart_delay: Delay between restart attempts in seconds
        """
        self.command = command
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.process: Optional[subprocess.Popen] = None
        
    def start_with_monitoring(self):
        """Start the process with segfault monitoring and auto-restart."""
        logger.info(f"Starting process with monitoring: {' '.join(self.command)}")
        
        while self.restart_count <= self.max_restarts:
            try:
                logger.info(f"Starting process (attempt {self.restart_count + 1})")
                
                self.process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Monitor process output
                for line in iter(self.process.stdout.readline, ''):
                    print(line.rstrip())
                
                # Wait for process to complete
                return_code = self.process.wait()
                
                if return_code == -11:  # SIGSEGV
                    logger.error(f"Process crashed with segmentation fault (exit code: {return_code})")
                    self.restart_count += 1
                    
                    if self.restart_count <= self.max_restarts:
                        logger.info(f"Restarting in {self.restart_delay} seconds...")
                        time.sleep(self.restart_delay)
                    else:
                        logger.error(f"Maximum restart attempts ({self.max_restarts}) reached")
                        break
                else:
                    logger.info(f"Process exited normally with code: {return_code}")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Process monitoring interrupted by user")
                if self.process:
                    self.process.terminate()
                break
            except Exception as e:
                logger.error(f"Error in process monitoring: {str(e)}")
                self.restart_count += 1
                time.sleep(self.restart_delay)

def create_monitored_server():
    """Create a monitored version of the anemll server."""
    
    def recovery_callback():
        """Recovery actions before server exit."""
        logger.info("Performing cleanup before exit...")
        # Add any cleanup operations here
        # Note: Keep this minimal as we're in a crashed state
        
    # Install segfault handler
    handler = SegmentationFaultHandler(recovery_callback)
    handler.install_handler()
    
    return handler

def run_server_with_restart_monitoring():
    """Run the server with automatic restart on segfault."""
    command = [sys.executable, "anemll-server-core.py"]
    monitor = ProcessMonitor(command, max_restarts=10, restart_delay=3.0)
    
    logger.info("Starting anemll server with segfault monitoring and auto-restart")
    monitor.start_with_monitoring()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        # Run with external process monitoring
        run_server_with_restart_monitoring()
    else:
        # Run with signal handler (integrate into existing server)
        handler = create_monitored_server()
        print("Segfault handler installed. Import this module in anemll-server-core.py")