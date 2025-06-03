#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
import json
import asyncio
import threading
import queue
import time
from pathlib import Path
import argparse
import uvicorn
import logging
import torch
import torch.nn.functional as F
import uuid
import psutil
import re

# Import segmentation fault handler
from segfault_handler import create_monitored_server

# Import from chat_full.py
from chat_full import (
    generate_next_token,
    run_prefill,
)

# Import new dynamic model loading components
from model_registry import ModelRegistry
from model_manager import ModelManager
from model_info import LoadedModel
from config import get_config

# Import CoreML error classification components
from coreml_error_classifier import get_classifier, classify_error
from coreml_wrapper import CoreMLMonitor, handle_coreml_error
from error_patterns import ErrorCategory, ErrorSeverity

# Configure logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anemll API Server - Dynamic Model Loading")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global model management
model_registry: Optional[ModelRegistry] = None
model_manager: Optional[ModelManager] = None

# Global CoreML error classifier
coreml_classifier = None

def check_memory_availability(min_memory_mb: int = 400) -> Dict[str, any]:
    """
    Check if sufficient memory is available before loading a model.
    
    Args:
        min_memory_mb: Minimum required memory in MB (default 400MB)
        
    Returns:
        Dict with availability status and memory info
    """
    try:
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)  # Convert bytes to MB
        
        memory_info = {
            "available_mb": round(available_mb, 2),
            "available_bytes": vm.available,
            "total_mb": round(vm.total / (1024 * 1024), 2),
            "used_percent": vm.percent,
            "sufficient": available_mb >= min_memory_mb,
            "required_mb": min_memory_mb
        }
        
        logger.info(f"Memory check: {available_mb:.2f}MB available, {min_memory_mb}MB required")
        
        return memory_info
        
    except Exception as e:
        logger.error(f"Error checking memory availability: {str(e)}")
        return {
            "available_mb": 0,
            "available_bytes": 0,
            "total_mb": 0,
            "used_percent": 0,
            "sufficient": False,
            "required_mb": min_memory_mb,
            "error": str(e)
        }

def get_server_memory_status():
    """Get current RSS as available memory for models (same calculation as monitor_server_health)"""
    try:
        # Same calculation as "Server health - Memory: X MB"
        current_process = psutil.Process(os.getpid())
        memory_info = current_process.memory_info()
        current_rss_mb = memory_info.rss / 1024 / 1024
        
        logger.debug(f"Current server RSS: {current_rss_mb:.1f} MB")
        return current_rss_mb
        
    except Exception as e:
        logger.error(f"Error getting server memory status: {str(e)}")
        return 0

def get_model_parameters_from_name(model_name: str) -> Optional[float]:
    """Get model parameters by parsing the model name for patterns like '1B', '8B', '500M', etc."""
    try:
        # Look for patterns like "1B", "8B" (billion parameters) or "500M" (million parameters/MB)
        # Pattern matches digits followed by B or M (case insensitive)
        pattern = r'(\d+)(B|M)'
        matches = re.findall(pattern, model_name, re.IGNORECASE)
        
        if matches:
            # Take the first match (in case there are multiple, which should be rare)
            value_str, unit = matches[0]
            value = float(value_str)
            unit = unit.upper()
            
            if unit == 'B':
                # Direct billion parameter count
                params_billions = value
                logger.info(f"Parsed {params_billions}B parameters from model name: {model_name}")
            elif unit == 'M':
                # Convert millions to billions (M could be million parameters or MB memory)
                # Treat as millions of parameters: 1000M = 1B
                params_billions = value / 1000.0
                logger.info(f"Parsed {value}M → {params_billions:.3f}B parameters from model name: {model_name}")
            else:
                logger.warning(f"Unknown unit '{unit}' in model name: {model_name}")
                return None
                
            return params_billions
        else:
            logger.warning(f"Could not parse parameter size from model name: {model_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing model name {model_name}: {str(e)}")
        return None

def get_model_parameters_from_config(model_name: str) -> Optional[float]:
    """
    DEPRECATED: Get model parameters from Huggingface config file
    
    This function is no longer used. The server now uses get_model_parameters_from_name()
    to parse parameter size directly from model names (e.g., "1B", "8B").
    """
    try:
        config_path = Path(f"./models/{model_name}/config.json")
        
        if not config_path.exists():
            logger.warning(f"Config file not found for model {model_name}: {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Try different parameter fields
        if 'num_parameters' in config:
            params = config['num_parameters']
        elif 'n_params' in config:
            params = config['n_params']
        elif 'n_parameters' in config:
            params = config['n_parameters']
        else:
            # Try to estimate from architecture
            if 'hidden_size' in config and 'num_hidden_layers' in config:
                hidden_size = config['hidden_size']
                num_layers = config['num_hidden_layers']
                vocab_size = config.get('vocab_size', 32000)
                
                # Rough estimation: embedding + transformer layers + output head
                # This is a simplified calculation
                estimated_params = (
                    vocab_size * hidden_size +  # input embedding
                    num_layers * (hidden_size * hidden_size * 4 + hidden_size * 2) +  # transformer layers (simplified)
                    vocab_size * hidden_size  # output head
                )
                params = estimated_params
                logger.info(f"Estimated {estimated_params:,} parameters for {model_name}")
            else:
                logger.warning(f"Cannot determine parameter count for model {model_name}")
                return None
        
        # Convert to billions
        params_billions = params / 1_000_000_000
        logger.info(f"Model {model_name} has {params_billions:.1f}B parameters")
        return params_billions
        
    except Exception as e:
        logger.error(f"Error reading config for model {model_name}: {str(e)}")
        return None

def can_load_model(model_name: str) -> tuple[bool, str]:
    """Check if model can be loaded based on memory requirements"""
    try:
        # Get model parameters by parsing model name
        model_params_billions = get_model_parameters_from_name(model_name)
        
        if model_params_billions is None:
            # Fallback: assume it's a medium model if we can't determine size
            logger.warning(f"Cannot determine model size for {model_name}, assuming 3B parameters")
            model_params_billions = 3.0
        
        # Calculate required memory using empirical formula
        # 300MB for 1B + 100MB for each additional billion parameters
        required_memory_mb = 300 + max(0, (model_params_billions - 1) * 100)
        
        # Get available memory using same RSS calculation as monitor_server_health
        available_memory_mb = get_server_memory_status()
        
        if available_memory_mb < required_memory_mb:
            error_msg = f"Model requires {required_memory_mb:.0f}MB, only {available_memory_mb:.1f}MB available"
            logger.warning(f"Memory check failed for {model_name}: {error_msg}")
            return False, error_msg
        
        logger.info(f"Memory check passed for {model_name}: {available_memory_mb:.1f}MB available for {required_memory_mb:.0f}MB required")
        return True, f"Sufficient memory available"
        
    except Exception as e:
        error_msg = f"Error checking model memory requirements: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def log_error_details(error: Exception, operation: str, model_name: str = None):
    """
    Log detailed error information using CoreML error classifier.
    
    Args:
        error: The exception that occurred
        operation: The operation being performed when error occurred
        model_name: Name of the model involved (if applicable)
    """
    try:
        if coreml_classifier is None:
            logger.error(f"Error in {operation}: {str(error)} (classifier not available)")
            return
            
        # Build context for error classification
        context = {
            "operation": operation,
            "model_name": model_name,
            "location": "anemll-server-core.py",
            "server_component": True
        }
        
        # Classify the error
        classified_error = coreml_classifier.classify_error(error, context)
        
        # Log comprehensive error information
        logger.error("=" * 80)
        logger.error("COREML ERROR CLASSIFICATION REPORT")
        logger.error("=" * 80)
        logger.error(f"Operation: {operation}")
        logger.error(f"Model: {model_name or 'Unknown'}")
        logger.error(f"Category: {classified_error.category.value}")
        logger.error(f"Severity: {classified_error.severity.value}")
        logger.error(f"Confidence: {classified_error.confidence:.2f}")
        logger.error(f"Error: {classified_error.original_error}")
        logger.error(f"Description: {classified_error.description}")
        
        if classified_error.potential_causes:
            logger.error("Potential Causes:")
            for cause in classified_error.potential_causes:
                logger.error(f"  - {cause}")
        
        if classified_error.recovery_steps:
            logger.error("Recovery Steps:")
            for step in classified_error.recovery_steps:
                logger.error(f"  - {step}")
        
        # Log memory information
        memory_info = check_memory_availability()
        logger.error(f"Memory Status: {memory_info['available_mb']:.2f}MB available")
        
        logger.error("=" * 80)
        
        # Attempt auto-recovery for critical errors
        if (classified_error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] and
            classified_error.can_auto_recover):
            
            logger.info(f"Attempting auto-recovery for {classified_error.category.value} error")
            recovery_result = coreml_classifier.attempt_auto_recovery(classified_error, context)
            
            if recovery_result.get("success"):
                logger.info("✅ Auto-recovery successful")
            else:
                logger.error(f"❌ Auto-recovery failed: {recovery_result.get('error')}")
        
    except Exception as classification_error:
        logger.error(f"Error during error classification: {str(classification_error)}")
        logger.error(f"Original error in {operation}: {str(error)}")

# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]

class StreamingTokenGenerator:
    """Handles streaming of tokens for the API response."""
    def __init__(self, loaded_model: LoadedModel, messages, temperature=0.7):
        # Extract components from loaded_model
        self.loaded_model = loaded_model
        self.embed_model = loaded_model.embed_model
        self.ffn_models = loaded_model.ffn_models
        self.lmhead_model = loaded_model.lmhead_model
        self.tokenizer = loaded_model.tokenizer
        self.metadata = loaded_model.metadata
        self.state = loaded_model.state
        self.causal_mask = loaded_model.causal_mask
        
        self.messages = messages
        self.temperature = temperature
        self.token_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.generation_thread = None
        self.context_length = self.metadata['context_length']
        self.batch_size = self.metadata['batch_size']
        self.queue_lock = threading.Lock()  # Add a lock for thread safety

    def start_generation(self):
        """Start token generation in a separate thread."""
        self.generation_thread = threading.Thread(target=self._generate_tokens)
        self.generation_thread.daemon = True
        self.generation_thread.start()

    def _generate_tokens(self):
        """Generate tokens and put them in the queue."""
        try:
            # Convert Pydantic Message objects to dictionaries for the tokenizer
            conversation = []
            for message in self.messages:
                conversation.append({"role": message.role, "content": message.content})
            
            # Format using chat template with full history
            base_input_ids = self.tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(torch.int32)
            
            # Check if the input exceeds context length
            if base_input_ids.size(1) > self.context_length:
                if config.allow_truncation:
                    logger.warning(f"Input length ({base_input_ids.size(1)}) exceeds context length ({self.context_length}). Truncating input.")
                    # Take the last context_length tokens to maintain the most recent context
                    base_input_ids = base_input_ids[:, -self.context_length:]
                else:
                    error_message = f"""
=============================================================================
ERROR: INPUT EXCEEDS MODEL CONTEXT LENGTH
=============================================================================
Input length: {base_input_ids.size(1)} tokens
Model context length: {self.context_length} tokens

Anemll models have a strict context limit, which is specified at conversion time and entered into the name as the ctx value.

To automatically truncate inputs to fit the model context window, restart the 
server with the --truncate flag:

    python anemll-server-core.py --truncate
=============================================================================
"""
                    logger.error(error_message)
                    with self.queue_lock:
                        self.token_queue.put({"error": "Input exceeds model context length. Use --truncate to enable automatic truncation."})
                    return
            
            # Convert to tensor with batch dimension
            input_ids = base_input_ids
            
            # Run prefill
            current_pos = input_ids.size(1)
            
            # Pad the input_ids tensor to the full context length
            # This is crucial to allow appending new tokens during generation
            input_ids = F.pad(
                input_ids,
                (0, self.context_length - current_pos),
                value=0
            )
            
            # The MLState object doesn't have an items() method
            state_copy = self.state
            
            logger.info(f"Running prefill with {current_pos} tokens")
            logger.info(f"Input shape: {input_ids.shape}")
            logger.info(f"Batch size: {self.batch_size}")
            
            # Run prefill for the conversation context
            try:
                run_prefill(
                    self.embed_model, 
                    self.ffn_models, 
                    input_ids,
                    current_pos, 
                    self.context_length, 
                    self.batch_size,
                    state_copy, 
                    self.causal_mask
                )
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f"CoreML model error: {error_msg}")
                
                # Use CoreML error classifier for detailed error analysis
                try:
                    log_error_details(e, "model_inference", self.loaded_model.info.name)
                except Exception as classification_error:
                    logger.error(f"Error during error classification: {str(classification_error)}")
                
                if "Shape" in error_msg and "was not in enumerated set of allowed shapes" in error_msg:
                    logger.error(f"The model requires specific input shapes. Please ensure the batch_size in meta.yaml ({self.batch_size}) matches what the model expects.")
                    
                # Put error message in queue and exit
                with self.queue_lock:
                    self.token_queue.put({"error": f"Model error: {error_msg}"})
                return
            
            # Generate tokens one by one
            pos = current_pos
            generated_ids = []
            
            logger.info(f"Starting token generation from position {pos}")
            
            while not self.stop_event.is_set():
                # Check if we need to shift window (similar to chat_full.py)
                if pos >= self.context_length - 2:
                    # Calculate shift to maintain full batches
                    batch_size = self.batch_size
                    # Calculate max batches that fit in context
                    max_batches = self.context_length // batch_size
                    desired_batches = max(1, max_batches - 2)  # Leave room for new tokens
                    new_size = min(desired_batches * batch_size, self.context_length - batch_size)
                    
                    # Create shifted input_ids
                    tmp = torch.zeros((1, self.context_length), dtype=torch.int32)
                    tmp[:,0:new_size] = input_ids[:,pos-new_size:pos]
                    input_ids = tmp
                    
                    # Run prefill on the shifted window
                    run_prefill(
                        self.embed_model,
                        self.ffn_models,
                        input_ids,
                        new_size,  # Prefill the entire shifted content
                        self.context_length,
                        self.batch_size,
                        state_copy,
                        self.causal_mask
                    )
                    
                    # Start generating from the next position
                    pos = new_size
                    logger.info(f"Window shifted, continuing from position {pos}")
                
                # Make sure input_ids has the right shape [1, seq_len]
                if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
                    input_ids = input_ids.view(1, -1)
                
                # Generate next token with minimal overhead
                next_token_id = generate_next_token(
                    self.embed_model,
                    self.ffn_models,
                    self.lmhead_model,
                    input_ids,
                    pos,
                    self.context_length,
                    state_copy,
                    self.causal_mask,
                    temperature=self.temperature
                )
                
                # Check for EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    logger.info(f"EOS token detected (ID: {next_token_id})")
                    self.stop_event.set()
                    with self.queue_lock:
                        self.token_queue.put(None)  # Signal end of generation
                    break
                
                # Update input_ids for next iteration
                input_ids[0, pos] = next_token_id
                
                # Decode and queue the token with minimal locking
                token_text = self.tokenizer.decode([next_token_id])
                with self.queue_lock:
                    self.token_queue.put(token_text)
                
                generated_ids.append(next_token_id)
                pos += 1
                
        except Exception as e:
            logger.error(f"Error in token generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Use CoreML error classifier for detailed error analysis
            try:
                log_error_details(e, "token_generation", self.loaded_model.info.name)
            except Exception as classification_error:
                logger.error(f"Error during error classification: {str(classification_error)}")
            
            with self.queue_lock:
                self.token_queue.put(None)  # Signal end of generation

    def get_tokens(self):
        """Get the next token from the queue."""
        try:
            # Use a non-blocking approach to reduce GIL contention
            try:
                # First try without lock and without waiting
                return self.token_queue.get_nowait()
            except queue.Empty:
                # If empty, use a very short timeout with lock
                with self.queue_lock:
                    return self.token_queue.get(timeout=0.01)
        except queue.Empty:
            return None

    def stop(self):
        """Stop token generation."""
        # Set the stop event to signal the generation thread to stop
        self.stop_event.set()
        
        # Clear the queue without locking if possible
        try:
            while not self.token_queue.empty():
                try:
                    self.token_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass
            
        # Join the thread with a short timeout to avoid blocking
        if self.generation_thread and self.generation_thread.is_alive():
            try:
                self.generation_thread.join(timeout=0.1)
            except Exception:
                pass


async def stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion response."""
    # Create a unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Check model-specific memory requirements before loading
    logger.info(f"🔍 MEMORY CHECK: Checking memory for streaming request: {request.model}")
    can_load, memory_msg = can_load_model(request.model)
    if not can_load:
        error_msg = f"Insufficient memory to load model '{request.model}': {memory_msg}"
        logger.error(f"🚫 MEMORY INSUFFICIENT: {error_msg}")
        logger.info(f"✅ GRACEFUL DEGRADATION: Returning error via streaming instead of HTTPException")
        
        # Return error through streaming response instead of raising HTTPException
        async def error_stream():
            # Send initial chunk indicating assistant role
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Send error message as content
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': f'❌ {error_msg}. Please try a smaller model or free up memory.'}, 'finish_reason': 'error'}]})}\n\n"
            
            # End the stream
            yield "data: [DONE]\n\n"
        
        return error_stream()
    
    try:
        
        # Get the requested model with CoreML monitoring
        with CoreMLMonitor("model_loading", request.model) as monitor:
            loaded_model = await model_manager.get_model(request.model)
        
        generator = StreamingTokenGenerator(
            loaded_model=loaded_model,
            messages=request.messages,
            temperature=request.temperature
        )
    except ValueError as e:
        # Model not found in registry
        log_error_details(e, "model_retrieval", request.model)
        available_models = model_registry.list_models()
        
        # Return error through streaming response instead of raising HTTPException
        async def model_not_found_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': f'❌ Model \'{request.model}\' not found. Available models: {available_models}'}, 'finish_reason': 'error'}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return model_not_found_stream()
    except RuntimeError as e:
        # Model loading failed
        log_error_details(e, "model_loading", request.model)
        
        async def model_load_failed_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': f'❌ Failed to load model \'{request.model}\': {str(e)}'}, 'finish_reason': 'error'}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return model_load_failed_stream()
    except Exception as e:
        # Any other error during model loading
        log_error_details(e, "model_loading", request.model)
        
        async def generic_error_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': f'❌ Error loading model \'{request.model}\': {str(e)}'}, 'finish_reason': 'error'}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return generic_error_stream()
    
    generator.start_generation()
    
    try:
        # Send the initial chunk
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        
        content_so_far = ""
        
        # Use a more efficient polling approach
        consecutive_empty_polls = 0
        max_empty_polls = 5
        
        while True:
            # Get token with minimal executor overhead
            token = generator.get_tokens()  # Direct call instead of using run_in_executor
            
            # Check if token is None (end of generation or timeout)
            if token is None:
                # Check if we're just waiting for more tokens or if the EOS token was detected
                if not generator.stop_event.is_set():
                    consecutive_empty_polls += 1
                    if consecutive_empty_polls >= max_empty_polls:
                        # After several empty polls, use a longer sleep to reduce CPU usage
                        await asyncio.sleep(0.05)
                        consecutive_empty_polls = 0
                    else:
                        # Short sleep for quick response
                        await asyncio.sleep(0.01)
                    continue
                
                # End of generation (EOS token was detected or generation was stopped)
                logger.info("End of generation detected (EOS token or stop event)")
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                break
            
            # Reset counter since we got a token
            consecutive_empty_polls = 0
            
            # Check if token is an error message
            if isinstance(token, dict) and "error" in token:
                error_msg = token["error"]
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': f'Error: {error_msg}'}, 'finish_reason': 'error'}]})}\n\n"
                yield "data: [DONE]\n\n"
                break
            
            content_so_far += token
            
            # Send the token
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
    finally:
        # Ensure generator is stopped even if an exception occurs
        generator.stop()

async def generate_chat_completion(request: ChatCompletionRequest):
    """Generate non-streaming chat completion response."""
    # Check model-specific memory requirements before loading
    logger.info(f"🔍 MEMORY CHECK: Checking memory for non-streaming request: {request.model}")
    can_load, memory_msg = can_load_model(request.model)
    if not can_load:
        error_msg = f"Insufficient memory to load model '{request.model}': {memory_msg}"
        logger.error(f"🚫 MEMORY INSUFFICIENT: {error_msg}")
        logger.info(f"✅ GRACEFUL DEGRADATION: Returning error response instead of HTTPException")
        
        # Return error response instead of raising HTTPException
        response_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            'id': response_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': f'❌ {error_msg}. Please try a smaller model or free up memory.'
                },
                'finish_reason': 'error'
            }]
        }
    
    try:
        
        # Get the requested model with CoreML monitoring
        with CoreMLMonitor("model_loading", request.model) as monitor:
            loaded_model = await model_manager.get_model(request.model)
        
        generator = StreamingTokenGenerator(
            loaded_model=loaded_model,
            messages=request.messages,
            temperature=request.temperature
        )
    except ValueError as e:
        # Model not found in registry
        log_error_details(e, "model_retrieval", request.model)
        available_models = model_registry.list_models()
        
        # Return error response instead of raising HTTPException
        response_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            'id': response_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': f'❌ Model \'{request.model}\' not found. Available models: {available_models}'
                },
                'finish_reason': 'error'
            }]
        }
    except RuntimeError as e:
        # Model loading failed
        log_error_details(e, "model_loading", request.model)
        
        response_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            'id': response_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': f'❌ Failed to load model \'{request.model}\': {str(e)}'
                },
                'finish_reason': 'error'
            }]
        }
    except Exception as e:
        # Any other error during model loading
        log_error_details(e, "model_loading", request.model)
        
        response_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            'id': response_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': f'❌ Error loading model \'{request.model}\': {str(e)}'
                },
                'finish_reason': 'error'
            }]
        }
    
    try:
        # Start token generation
        generator.start_generation()
        
        # Collect all tokens
        full_content = ""
        while True:
            token = await asyncio.get_event_loop().run_in_executor(None, generator.get_tokens)
            if token is None:
                if generator.stop_event.is_set():
                    break
                await asyncio.sleep(0.05)
                continue
            
            if isinstance(token, dict) and "error" in token:
                raise Exception(token["error"])
                
            full_content += token
    finally:
        try:
            # Make sure to stop the generator in a way that doesn't block
            await asyncio.get_event_loop().run_in_executor(None, generator.stop)
        except Exception as e:
            logger.error(f"Error stopping generator: {str(e)}")
    
    # Create response
    response_id = f"chatcmpl-{uuid.uuid4()}"
    return {
        'id': response_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': request.model,
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': full_content
            },
            'finish_reason': 'stop'
        }]
    }

def log_memory_status_after_completion(request_model: str, completion_success: bool):
    """Log RSS memory status after chat completion for crash analysis."""
    try:
        current_rss_mb = get_server_memory_status()
        
        # Get system memory info for context
        vm = psutil.virtual_memory()
        system_available_mb = vm.available / (1024 * 1024)
        system_used_percent = vm.percent
        
        status_icon = "✅" if completion_success else "❌"
        logger.info(f"📊 {status_icon} COMPLETION RSS MEMORY: {current_rss_mb:.1f}MB "
                   f"(System: {system_available_mb:.1f}MB available, {system_used_percent:.1f}% used) "
                   f"Model: {request_model}")
        
        # Warning thresholds for memory pressure
        if current_rss_mb > 800:
            logger.warning(f"⚠️  HIGH RSS MEMORY: {current_rss_mb:.1f}MB - Potential crash risk")
        elif current_rss_mb > 600:
            logger.warning(f"⚠️  ELEVATED RSS MEMORY: {current_rss_mb:.1f}MB - Monitor for accumulation")
            
    except Exception as e:
        logger.error(f"Error logging memory status: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    completion_success = False
    try:
        if request.stream:
            result = StreamingResponse(
                stream_chat_completion(request),
                media_type="text/event-stream"
            )
            completion_success = True
            return result
        else:
            result = await generate_chat_completion(request)
            completion_success = True
            return result
    except Exception as e:
        completion_success = False
        raise
    finally:
        # Always log memory status after completion attempt
        log_memory_status_after_completion(request.model, completion_success)


@app.get("/v1/models")
@app.options("/v1/models")
async def list_models_v1():
    """List available models (v1 endpoint) - required for Open WebUI compatibility."""
    try:
        models = model_registry.list_models()
        created_time = int(time.time())
        
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "anemll",
                    "permission": [],
                    "root": model_name,
                    "parent": None
                }
                for model_name in models
            ]
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.get("/v1/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    model_info = model_registry.get_model_info(model_name)
    if not model_info:
        available_models = model_registry.list_models()
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {available_models}"
        )
    
    return model_info.get_display_info()

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get model cache statistics (admin endpoint)."""
    return model_manager.get_cache_stats()

@app.post("/admin/cache/clear")
async def clear_cache():
    """Clear the model cache (admin endpoint)."""
    model_manager.clear_cache()
    return {"message": "Cache cleared successfully"}

@app.post("/admin/models/refresh")
async def refresh_models():
    """Refresh the model registry (admin endpoint)."""
    model_registry.refresh()
    return {
        "message": "Model registry refreshed",
        "available_models": model_registry.list_models()
    }

def initialize_model_system():
    """Initialize the dynamic model loading system."""
    global model_registry, model_manager, coreml_classifier
    
    logger.info("Initializing dynamic model loading system")
    
    try:
        # Initialize CoreML error classifier
        try:
            coreml_classifier = get_classifier()
            logger.info("CoreML error classifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CoreML error classifier: {str(e)}")
            coreml_classifier = None
        
        # Initialize model registry
        model_registry = ModelRegistry(config.model_dir)
        logger.info(f"Model registry initialized with {len(model_registry)} models")
        
        # Initialize model manager with configured cache size
        model_manager = ModelManager(model_registry, config.cache_size)
        logger.info(f"Model manager initialized with cache size: {config.cache_size}")
        
        # Log available models
        available_models = model_registry.list_models()
        if available_models:
            logger.info(f"Available models: {available_models}")
            
            # Preload models if configured
            if config.preload_models:
                logger.info(f"Preloading models: {config.preload_models}")
                # Note: Preloading will happen on first request if not done here
                # Could be enhanced to preload during startup in a background task
        else:
            logger.warning("No valid models found in model directory")
            logger.warning("Server will start but no models will be available for inference")
        
    except FileNotFoundError as e:
        logger.error(f"Model directory not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize model system: {str(e)}")
        sys.exit(1)

async def preload_models():
    """Preload configured models."""
    for model_name in config.preload_models:
        try:
            logger.info(f"Preloading model: {model_name}")
            await model_manager.get_model(model_name)
            logger.info(f"Successfully preloaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to preload model {model_name}: {str(e)}")

def main():
    """Main function to start the server."""
    
    # Install segmentation fault handler first
    logger.info("Installing segmentation fault handler...")
    segfault_handler = create_monitored_server()
    logger.info("Segmentation fault handler installed - server will log detailed crash information")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Anemll API Server - Dynamic Model Loading")
    parser.add_argument("--truncate", action="store_true", help="Enable automatic truncation of inputs that exceed model context length")
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    config.update_from_args(args)
    
    # Initialize the dynamic model loading system
    initialize_model_system()
    
    # Start the server
    logger.info(f"Starting server on {config.host}:{config.port} (Truncation: {'Enabled' if config.allow_truncation else 'Disabled'})")
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main() 
