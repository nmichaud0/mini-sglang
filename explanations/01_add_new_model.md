# Adding a New HuggingFace Model Architecture

This guide walks through every step required to add a brand-new model architecture to mini-sglang, from config parsing to running inference.

---

## Overview

Three things need to happen:

1. **Parse the HuggingFace config** → `ModelConfig` (possibly extend `from_hf`)
2. **Implement the model** as a hierarchy of `BaseOP` subclasses
3. **Register the architecture name** in the model registry

Weight loading is handled automatically as long as the weight names in the checkpoint match the structure your model exposes via `state_dict()`.

---

## Step 1 — Check / Extend `ModelConfig.from_hf`

File: `python/minisgl/models/config.py`

`ModelConfig` is a frozen dataclass. The engine calls `ModelConfig.from_hf(hf_config)` where `hf_config` is the `PretrainedConfig` object loaded from HuggingFace. The method already handles the common fields:

```python
# Already extracted automatically:
num_layers, num_qo_heads, num_kv_heads, head_dim, hidden_size,
vocab_size, intermediate_size, hidden_act, rms_norm_eps,
tie_word_embeddings, rotary_config (rope_theta / rope_scaling),
num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob
```

If your model uses **non-standard field names** in its HuggingFace config, add handling inside `from_hf`:

```python
@classmethod
def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
    # existing code ...

    # Example: your model calls the intermediate size "ffn_dim"
    intermediate_size = getattr(config, "intermediate_size", None) or config.ffn_dim

    return cls(
        # ... existing fields ...
        intermediate_size=intermediate_size,
    )
```

If your model needs a **new config field** (e.g. a custom sliding window size), add it to the `ModelConfig` dataclass and parse it in `from_hf`.

> `ModelConfig` is frozen (`frozen=True`). It is read-only after construction — never mutate it.

---

## Step 2 — Implement the Model

Create a new file: `python/minisgl/models/mymodel.py`

Every class inherits from `BaseOP` (defined in `python/minisgl/layers/base.py`). This is **not** `nn.Module`, but it has the same `state_dict` / `load_state_dict` contract.

### 2a. The decoder layer

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.layers import (
    BaseOP,
    RMSNormFused,
    VocabParallelEmbedding,
    OPList,
    ParallelLMHead,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP, RopeAttn   # reuse these if the arch is standard

if TYPE_CHECKING:
    from .config import ModelConfig


class MyDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id                         # underscore → ignored by state_dict

        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = RopeAttn(config, layer_id)       # standard RoPE attention
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = GatedMLP(config)                       # gate_up + down projection

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)   # fused add+norm
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual
```

Key rules:
- Attributes **starting with `_`** are invisible to `state_dict` — use them for layer_id and other non-parameter metadata.
- The `RMSNormFused.forward(x, residual)` pattern is the fused residual-add + norm kernel. It returns `(normalized, new_residual)`. The first layer passes `residual=None`.
- You **must** keep the same attribute names as the checkpoint — weight lookup is by attribute path (see Step 3).

### 2b. The trunk model

```python
class MyModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [MyDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]
```

### 2c. The causal LM head

```python
class MyForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = MyModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()          # must be called after attributes are set

    def forward(self) -> torch.Tensor:
        from minisgl.core import get_global_ctx
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["MyForCausalLM"]
```

Important: `forward()` takes **no arguments**. Input tokens are read from the global `Context` (`get_global_ctx().batch.input_ids`). This is intentional — the engine uses CUDA graphs and cannot have dynamic Python arguments in the replay path.

---

## Step 3 — Register the Architecture

File: `python/minisgl/models/register.py`

```python
_MODEL_REGISTRY = {
    # existing entries ...
    "MyForCausalLM": (".mymodel", "MyForCausalLM"),
}
```

The key (`"MyForCausalLM"`) must exactly match the string in the model's HuggingFace `config.json`:

```json
{
  "architectures": ["MyForCausalLM"],
  ...
}
```

The engine calls `get_model_class(architecture_string, model_config)`, which does `importlib.import_module(".mymodel", package="minisgl.models")` and returns `MyForCausalLM(model_config)`.

---

## Step 4 — Weight Name Mapping

The weight loader (`python/minisgl/models/weight.py`) streams safetensors files and calls your model's `load_state_dict`. It builds weight names from the file by joining attribute names with dots:

```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
...
```

The checkpoint's weight names must match this path, **or** you need to handle the discrepancy in the weight loader.

### Automatic merging

The loader already merges split projections:

| Checkpoint has | Loader yields |
|----------------|---------------|
| `q_proj`, `k_proj`, `v_proj` | `qkv_proj` |
| `gate_proj`, `up_proj` | `gate_up_proj` |

So a standard HuggingFace checkpoint (with split q/k/v) maps to mini-sglang's fused `LinearQKVMerged` and `LinearColParallelMerged` automatically.

### Custom name remapping

If your checkpoint uses different names (e.g. `attention.wq` instead of `self_attn.q_proj`), add entries to `_MERGE_GROUPS` or add a renaming pass before calling `load_state_dict`:

```python
# In weight.py or in a model-specific loader
def remap_keys(state_dict):
    mapping = {
        "attention.wq": "self_attn.q_proj",
        "attention.wk": "self_attn.k_proj",
        # ...
    }
    return {mapping.get(k, k): v for k, v in state_dict.items()}
```

---

## Step 5 — Verify the Architecture Variant

Check whether your model differs from Llama only in small ways:

| Feature | How to enable |
|---------|--------------|
| Attention bias (`q_proj` has bias) | `RopeAttn(config, layer_id, has_attn_bias=True)` |
| Q/K normalization (Qwen3-style) | `RopeAttn(config, layer_id, has_qk_norm=True)` |
| Mixture-of-Experts MLP | Replace `GatedMLP` with `MoEMLP` |
| GELU instead of SiLU | Set `config.hidden_act = "gelu"` — `GatedMLP` handles it |
| Different RoPE scaling | Handled by `RotaryConfig.scaling` — just ensure `rope_scaling` is set in HF config |

---

## Complete Checklist

- [ ] `ModelConfig.from_hf` parses all config fields your model needs
- [ ] `mymodel.py` implements `MyDecoderLayer`, `MyModel`, `MyForCausalLM`
- [ ] All attribute names match the checkpoint weight paths (after merge transforms)
- [ ] Architecture key added to `_MODEL_REGISTRY` in `register.py`
- [ ] `__all__` exported from `mymodel.py`
