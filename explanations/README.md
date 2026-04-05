# mini-sglang Explanations

Practical guides for working with mini-sglang. Start here.

## Guides

| File | What it covers |
|------|---------------|
| [01_add_new_model.md](01_add_new_model.md) | Register a new HuggingFace architecture end-to-end |
| [02_modify_internal_components.md](02_modify_internal_components.md) | Swap or customize attention, MLP, norms, activations |
| [03_quantization.md](03_quantization.md) | Layer-by-layer quantization via the `BaseOP` class system |
| [04_run_with_minisglang.md](04_run_with_minisglang.md) | Serve or run inference from Python once your model is wired in |

## Mental model in one paragraph

mini-sglang is **not** a PyTorch `nn.Module` stack — it has its own `BaseOP` system with custom `state_dict` / `load_state_dict` that mirrors PyTorch's interface but controls weight loading precisely.
Models live in `python/minisgl/models/`, layers live in `python/minisgl/layers/`.
A model is registered by architecture name (matching the `architectures` field in HuggingFace `config.json`), and the engine loads weights by streaming safetensors files, sharding them per tensor-parallel rank, and calling the model's `load_state_dict`.
The global `Context` object (`minisgl.core.get_global_ctx()`) is the shared bus: it holds the current `Batch`, the KV cache, and the attention backend. Every `forward()` call reads from it instead of receiving arguments from the caller.
