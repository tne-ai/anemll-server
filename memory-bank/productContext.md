# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-06-01 22:39:43 - Log of updates made will be appended as footnotes to the end of this file.

## Project Goal

Create an OpenAI-compatible API server for Anemll models that provides `/v1/chat/completions` and `/v1/models` endpoints, enabling integration with existing OpenAI-compatible frontends like Open WebUI while leveraging Apple Neural Engine (ANE) for efficient on-device ML inference.

## Key Features

* **OpenAI API Compatibility**: Full compatibility with OpenAI chat completions API format
* **Streaming Responses**: Support for both streaming and non-streaming response modes
* **Conversation History**: Support for system prompts and multi-turn conversations
* **Open WebUI Integration**: Works seamlessly with Open WebUI and other compatible frontends
* **Apple Neural Engine Acceleration**: Utilizes ANE for efficient local model inference
* **Context Window Management**: Automatic sliding window with truncation options for long conversations
* **Multi-model Support**: Can load and serve different Anemll model variants
* **CORS Support**: Full CORS middleware for web frontend compatibility

## Overall Architecture

* **FastAPI Backend**: Modern async Python web framework for API endpoints
* **Streaming Token Generation**: Threaded token generation with queue-based streaming
* **CoreML Integration**: Direct integration with Apple's CoreML for ANE acceleration
* **Model Components**: Separate embedding, FFN (Feed-Forward Network), and language model head components
* **State Management**: Unified state management for model inference across requests
* **Configuration**: Environment-based configuration with YAML metadata support