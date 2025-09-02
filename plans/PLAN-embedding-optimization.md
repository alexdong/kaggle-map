# Embedding Model Optimization Plan

## Goal
Select the best embedding model for the MAP competition by testing top 7 candidates with memory-efficient architectures.

## Top 7 Embedding Models (Selected for Testing)

1. **MiniLM-L6-v2** (384 dims) - Current baseline
2. **MPNet-base-v2** (768 dims) - Already in codebase, proven improvement
3. **E5-base-v2** (768 dims) - Strong balanced model 
4. **Instructor-base** (768 dims) - Task-specific instructions for math
5. **BGE-base-en-v1.5** (768 dims) - Modern efficient architecture
6. **GTE-base-en-v1.5** (768 dims) - Strong on complex text
7. **UAE-base-V1** (768 dims) - AnglE optimized model

## Memory-Aware Architecture Scaling

Given OOM issues at batch_size=224-288 with xlarge architecture:

### Embedding-Specific Architecture Configs

```python
architecture_configs = {
    # 384-dim embeddings (MiniLM)
    384: {
        "small": [800, 512, 256, 128],      # Conservative
        "medium": [800, 768, 384, 192],     # Balanced
        "large": [800, 1024, 512, 256],     # Max feasible
    },
    # 768-dim embeddings (Most models)  
    768: {
        "small": [1568, 768, 384, 192],     # Conservative
        "medium": [1568, 1024, 512, 256],   # Balanced
        "large": [1568, 1536, 768, 384],    # Risky with batch>200
    }
}
```

## Implementation Changes

### [ ] 1. Add EmbeddingModel enum extensions
```python
# kaggle_map/core/embeddings/embedding_models.py
class EmbeddingModel(Enum):
    # Existing...
    E5_BASE = "intfloat/e5-base-v2"
    INSTRUCTOR_BASE = "hkunlp/instructor-base"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    GTE_BASE = "Alibaba-NLP/gte-base-en-v1.5"  # Already exists
    UAE_BASE = "WhereIsAI/UAE-Large-V1"
```

### [ ] 2. Modify MLPStrategy hyperparameter space
```python
@classmethod
def get_embedding_search_space(cls, trial) -> dict[str, Any]:
    """Focused embedding model search."""
    
    # Fixed best parameters from previous study
    base_params = {
        "learning_rate": 0.00028,
        "batch_size": 192,  # Reduced for memory safety
        "dropout": 0.32,
        "optimizer": "adamw",
        "weight_decay": 0.0065,
        "activation": "silu",
        "scheduler": "cosine",
        "early_stopping_patience": 19,
        "epochs": 35,
    }
    
    # Embedding selection
    embedding_model = trial.suggest_categorical("embedding_model", [
        "MiniLM", "MPNet", "E5_BASE", "INSTRUCTOR_BASE", 
        "BGE_BASE", "GTE_BASE", "UAE_BASE"
    ])
    
    # Dynamic architecture based on embedding dims
    embedding_dim = 384 if embedding_model == "MiniLM" else 768
    
    # Conservative architecture sizing for memory
    if embedding_dim == 384:
        architecture = trial.suggest_categorical("architecture", ["small", "medium", "large"])
    else:
        # More conservative for larger embeddings
        architecture = trial.suggest_categorical("architecture", ["small", "medium"])
    
    # Dynamic layer configuration
    trunk_layers = get_dynamic_layers(embedding_dim, architecture)
    
    return {
        **base_params,
        "embedding_model": embedding_model,
        "trunk_layers": trunk_layers,
        "num_layers": len(trunk_layers) - 1,
    }
```

### [ ] 3. Update fit() method to handle embedding selection
```python
@classmethod
def fit(cls, embedding_model: str = "MiniLM", **kwargs):
    # Get embedding model enum
    embed_model = getattr(EmbeddingModel, embedding_model)
    
    # Compute embeddings with selected model
    tokenizer = get_tokenizer(model=embed_model, device=str(device))
    
    # Adjust batch size for encoding based on model size
    encode_batch_size = 32 if embed_model.dim > 384 else 64
```

### [ ] 4. Create new optimise command
```python
# kaggle_map/optimise.py
@cli.command()
@click.argument("strategy_name")
@click.option("--trials", default=70, help="Number of trials (7 models x 10 configs)")
@click.option("--train-data", help="Path to training data")
def search_embeddings(strategy_name: str, trials: int, train_data: str | None):
    """Run embedding model comparison study."""
    # Special handling for embedding search
```

### [ ] 5. Add memory monitoring
```python
def monitor_memory(trial):
    """Track memory usage during training."""
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated() / 1024**3
        trial.set_user_attr("memory_before_gb", memory_before)
        
        # After model creation
        memory_after_model = torch.cuda.memory_allocated() / 1024**3
        trial.set_user_attr("memory_after_model_gb", memory_after_model)
        
        # Peak during training
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        trial.set_user_attr("peak_memory_gb", peak_memory)
```

## Execution Strategy

1. **Phase 1: Quick Elimination (20 trials)**
   - Test each embedding with "small" architecture
   - Batch size: 192
   - Epochs: 10 (early stopping)
   - Eliminate bottom 3 performers

2. **Phase 2: Top 4 Deep Dive (40 trials)**
   - Test remaining 4 embeddings
   - Try "small" and "medium" architectures
   - Full 35 epochs
   - Monitor memory carefully

3. **Phase 3: Winner Optimization (10 trials)**
   - Take best embedding
   - Fine-tune batch size and architecture
   - Try to push limits without OOM

## Success Metrics

- Primary: MAP@3 validation score
- Secondary: Training stability (no OOM)
- Tertiary: Inference speed

## Risk Mitigation

1. **OOM Prevention**:
   - Start with small architectures
   - Reduce batch size for larger models
   - Clear cache between trials
   - Monitor GPU memory

2. **Time Management**:
   - 70 trials total
   - ~4-5 minutes per trial
   - Total: ~5-6 hours

3. **Fallback**:
   - If all fail, revert to MiniLM with xlarge architecture
   - Document memory requirements for each combination