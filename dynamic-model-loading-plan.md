# Dynamic Model Loading Implementation Plan

## Overview
Transform the Anemll server from single-model to multi-model architecture with LRU caching for optimal memory usage and performance.

## New Architecture Components

### 1. ModelRegistry Class
```python
class ModelRegistry:
    """Discovers and catalogs available Anemll models"""
    
    def __init__(self, models_base_dir: str):
        self.models_base_dir = Path(models_base_dir)
        self.available_models = {}  # model_name -> ModelInfo
        self.scan_models()
    
    def scan_models(self) -> Dict[str, ModelInfo]:
        """Recursively scan for valid Anemll models"""
        
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get metadata for specific model"""
        
    def list_models(self) -> List[str]:
        """List all available model names"""
        
    def refresh(self):
        """Re-scan directory for new models"""
```

### 2. ModelInfo Data Class
```python
@dataclass
class ModelInfo:
    """Model metadata and file paths"""
    name: str
    path: Path
    embed_path: str
    ffn_path: str  
    lmhead_path: str
    tokenizer_path: str
    context_length: int
    batch_size: int
    metadata: Dict[str, Any]
    
    @classmethod
    def from_meta_yaml(cls, model_dir: Path) -> 'ModelInfo':
        """Create ModelInfo from meta.yaml file"""
```

### 3. ModelManager Class
```python
class ModelManager:
    """Handles model loading, caching, and lifecycle"""
    
    def __init__(self, registry: ModelRegistry, cache_size: int = 3):
        self.registry = registry
        self.cache_size = cache_size
        self.loaded_models = {}  # model_name -> LoadedModel
        self.lru_tracker = OrderedDict()  # model_name -> timestamp
        self.lock = threading.RLock()
    
    async def get_model(self, model_name: str) -> LoadedModel:
        """Get model (load if needed, manage LRU cache)"""
        
    def _load_model(self, model_info: ModelInfo) -> LoadedModel:
        """Load model components from disk"""
        
    def _evict_lru_model(self):
        """Remove least recently used model from cache"""
        
    def unload_model(self, model_name: str):
        """Explicitly unload a model"""
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics"""
```

### 4. LoadedModel Data Class
```python
@dataclass  
class LoadedModel:
    """Container for loaded model components"""
    info: ModelInfo
    embed_model: Any
    ffn_models: Any 
    lmhead_model: Any
    tokenizer: Any
    metadata: Dict[str, Any]
    state: Any
    causal_mask: Any
    load_time: float
    last_used: float
```

## Integration Strategy

### Current Code Changes Required

#### 1. Replace Global Variables
**Before:**
```python
# Global variables to store model components
embed_model = None
ffn_models = None  
lmhead_model = None
tokenizer = None
metadata = {}
state = None
causal_mask = None
```

**After:**
```python
# Global model management
model_registry = None
model_manager = None
```

#### 2. Update StreamingTokenGenerator
**Current Constructor:**
```python
def __init__(self, embed_model, ffn_models, lmhead_model, tokenizer, metadata, state, causal_mask, messages, temperature=0.7):
```

**New Constructor:**
```python  
def __init__(self, loaded_model: LoadedModel, messages, temperature=0.7):
    self.loaded_model = loaded_model
    # Extract components from loaded_model
    self.embed_model = loaded_model.embed_model
    self.ffn_models = loaded_model.ffn_models
    # ... etc
```

#### 3. Modify Chat Completion Endpoints
**Before:**
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    generator = StreamingTokenGenerator(
        embed_model=embed_model,
        ffn_models=ffn_models,
        # ... global variables
    )
```

**After:**
```python
@app.post("/v1/chat/completions") 
async def chat_completions(request: ChatCompletionRequest):
    # Get the requested model
    loaded_model = await model_manager.get_model(request.model)
    
    generator = StreamingTokenGenerator(
        loaded_model=loaded_model,
        messages=request.messages,
        temperature=request.temperature
    )
```

#### 4. Update Models Endpoint
**Before:**
```python
@app.get("/v1/models")
async def list_models_v1():
    return {
        "object": "list", 
        "data": [{"id": "anemll-model", ...}]
    }
```

**After:**
```python
@app.get("/v1/models")
async def list_models_v1():
    models = model_registry.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "anemll",
                "permission": [],
                "root": model_name,
                "parent": None
            }
            for model_name in models
        ]
    }
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. **Create new classes** (`ModelRegistry`, `ModelInfo`, `ModelManager`, `LoadedModel`)
2. **Add model discovery logic** with meta.yaml parsing
3. **Implement LRU cache mechanics** with thread safety
4. **Add configuration options** for cache size and behavior

### Phase 2: Integration
1. **Refactor startup process** to initialize registry and manager instead of loading single model
2. **Update StreamingTokenGenerator** to use LoadedModel
3. **Modify chat completion endpoints** to request specific models
4. **Update models endpoint** to return dynamic list

### Phase 3: Error Handling & Validation
1. **Add model validation** during discovery
2. **Implement graceful fallbacks** for model loading failures  
3. **Add comprehensive logging** for debugging
4. **Create health check endpoints** for monitoring

### Phase 4: Configuration & Optimization
1. **Add environment variables** for cache configuration
2. **Implement cache warming** options
3. **Add metrics and monitoring** for cache performance
4. **Performance testing** with multiple models

## Configuration Options

### Environment Variables
```bash
# Model discovery
MODEL_DIR=/path/to/models                    # Base directory for model scanning
MODEL_CACHE_SIZE=3                          # Number of models to keep in cache
MODEL_SCAN_RECURSIVE=true                   # Recursive directory scanning

# Cache behavior  
MODEL_CACHE_TTL=3600                        # Time to live for cached models (seconds)
MODEL_PRELOAD=""                            # Comma-separated list of models to preload
MODEL_LOAD_TIMEOUT=60                       # Timeout for model loading (seconds)

# Memory management
MODEL_CACHE_MEMORY_LIMIT=8GB                # Maximum memory for model cache
```

### Configuration File Support
Optional `models_config.yaml`:
```yaml
cache:
  size: 3
  ttl: 3600
  memory_limit: "8GB"

preload:
  - "anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2"
  
discovery:
  recursive: true
  include_patterns:
    - "anemll-*"
  exclude_patterns:
    - "*.tmp"
    - ".*"
```

## Error Handling Strategy

### Model Loading Errors
- **Missing Files**: Log error, exclude from registry
- **Invalid meta.yaml**: Skip model, continue scanning
- **CoreML Errors**: Graceful degradation, detailed logging
- **Memory Errors**: Trigger cache cleanup, retry

### Runtime Errors  
- **Unknown Model**: Return 404 with available models list
- **Loading Timeout**: Return 503 with retry suggestion
- **Cache Full**: Automatic LRU eviction
- **Model Corruption**: Remove from cache, reload on next request

## Testing Strategy

### Unit Tests
- Model discovery logic with various directory structures
- LRU cache behavior under different scenarios
- ModelInfo parsing with valid/invalid meta.yaml files
- Thread safety of ModelManager operations

### Integration Tests
- End-to-end model loading and chat completion
- Multi-model concurrent requests
- Cache eviction and memory management
- API endpoint responses with dynamic model list

### Performance Tests
- Model loading time benchmarks
- Cache hit/miss ratio optimization
- Memory usage under load
- Concurrent model switching performance

## Migration Strategy

### Backward Compatibility
- **Single Model Mode**: If MODEL_DIR points to a single model, maintain existing behavior
- **Default Model**: First discovered model becomes default if no model specified in request
- **Legacy Endpoints**: Existing clients continue working without changes

### Gradual Rollout
1. **Development**: Implement with feature flag
2. **Testing**: Deploy to test environment with multiple models
3. **Staging**: Validate performance and stability
4. **Production**: Enable with monitoring and rollback plan

## File Structure
```
anemll-server/
├── anemll-server.py          # Main server (updated)
├── model_registry.py         # New: Model discovery and cataloging
├── model_manager.py          # New: Model caching and lifecycle  
├── model_info.py            # New: Model metadata structures
├── chat_full.py             # Existing: Core model operations
├── config.py                # New: Configuration management
└── tests/
    ├── test_model_registry.py
    ├── test_model_manager.py
    └── test_integration.py
```

This plan provides a robust foundation for dynamic model loading while maintaining backward compatibility and optimizing for performance and memory usage.