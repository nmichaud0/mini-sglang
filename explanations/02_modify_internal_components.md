# Modifying Internal Components

This guide explains how to swap or customize individual building blocks — attention, MLP, normalization, activations, linear projections — without rewriting the entire model.

---

## The `BaseOP` contract

All components in mini-sglang inherit from `BaseOP` (`python/minisgl/layers/base.py`). The key rules:

```python
class BaseOP:
    def forward(self, *args, **kwargs): ...   # must implement

    def state_dict(self, *, prefix="", result=None) -> dict:
        # Recursively walks self.__dict__
        # Tensors → added to dict under "{prefix}.{attr_name}"
        # BaseOP subinstances → recursed into
        # Attrs starting with "_" → SKIPPED

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        # Matches by attr name, checks shape and dtype, replaces tensor in-place
```

Any class you create that inherits from `BaseOP` and stores tensors as plain attributes will automatically participate in weight loading. No registration needed beyond the attribute assignment.

---

## Modifying the MLP

The standard MLP is `GatedMLP` in `python/minisgl/models/utils.py`. It uses:
- `LinearColParallelMerged` for the fused `gate_up_proj` (column-parallel, output sharded)
- `LinearRowParallel` for `down_proj` (row-parallel, all-reduced)
- A gating activation (`silu_and_mul` or `gelu_and_mul`)

### Swapping the activation function

The activation is chosen from `config.hidden_act` in `GatedMLP.__init__`:

```python
FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
act_fn = FN_MAP.get(config.hidden_act, None)
```

To add a new activation (e.g. `swiglu`), add it to `python/minisgl/layers/activation.py` and extend the map:

```python
# activation.py
def swiglu_and_mul(x: torch.Tensor) -> torch.Tensor:
    gate, val = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * gate * val   # example

# models/utils.py GatedMLP.__init__
FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul, "swiglu": swiglu_and_mul}
```

### Building a completely different MLP

Subclass `BaseOP` and follow the tensor-parallelism conventions:

```python
from minisgl.layers import BaseOP, LinearColParallelMerged, LinearRowParallel

class MyMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        # column-parallel: each rank gets a slice of the output dimension
        self.w1 = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size],   # list of output sizes to merge
            has_bias=False,
        )
        # row-parallel: each rank gets a slice of the input dimension, all-reduced at end
        self.w2 = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(torch.relu(self.w1.forward(x)))
```

Replace `GatedMLP` with `MyMLP` in your decoder layer class.

---

## Modifying Attention

The standard attention component is `RopeAttn` in `python/minisgl/models/utils.py`. It is composed of:

| Sub-component | Class | Role |
|---|---|---|
| `qkv_proj` | `LinearQKVMerged` | Fused Q+K+V projection, column-parallel |
| `attn` | `AttentionLayer` | Splits QKV, applies RoPE, calls backend |
| `o_proj` | `LinearOProj` | Output projection, row-parallel with all-reduce |
| `q_norm` / `k_norm` | `RMSNorm` (optional) | Per-head Q/K normalization (Qwen3 style) |

### Enabling Q/K normalization

```python
self.self_attn = RopeAttn(config, layer_id, has_qk_norm=True)
```

This adds `self.q_norm` and `self.k_norm` as `RMSNorm` instances inside `RopeAttn`. The weight keys exposed are `self_attn.q_norm.weight` and `self_attn.k_norm.weight`.

### Enabling attention bias

```python
self.self_attn = RopeAttn(config, layer_id, has_attn_bias=True)
```

This makes `qkv_proj` carry a bias tensor, which is then picked up by `load_state_dict` automatically.

### Custom attention (non-RoPE or sliding window)

Build your own attention op inheriting from `BaseOP`. The only hard requirement is that you eventually call `ctx.attn_backend.forward(q, k, v, layer_id, ctx.batch)`:

```python
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, LinearQKVMerged, LinearOProj, AttentionLayer

class MyCustomAttn(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
        )
        self.o_proj = LinearOProj(
            config.head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )
        # custom extra weight
        self.my_scaling = torch.empty(config.num_qo_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        o = self.attn.forward(qkv)         # RoPE + backend call happens inside
        return self.o_proj.forward(o)
```

The `my_scaling` tensor will automatically be included in `state_dict()` as `self_attn.my_scaling` and loaded from the checkpoint if that key is present.

---

## Modifying Normalization

The two norm classes are in `python/minisgl/layers/norm.py`:

| Class | When to use |
|---|---|
| `RMSNormFused` | Layer input/output norms — fused add+norm kernel, returns `(normalized, residual)` |
| `RMSNorm` | Standalone norm inside attention (Q/K norms) |

### Replacing with LayerNorm

If your model uses LayerNorm instead of RMSNorm:

```python
import torch
from minisgl.layers import BaseOP

class LayerNormOP(BaseOP):
    def __init__(self, size: int, eps: float = 1e-5):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self._eps = eps             # underscore → not in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self._eps
        )
```

Note: If you use this as a drop-in for `RMSNormFused`, you need to also handle the residual manually in the decoder layer:

```python
def forward(self, x, residual=None):
    if residual is not None:
        x = x + residual
    residual = x
    x = self.input_layernorm.forward(x)   # returns just x, not (x, residual)
    x = self.self_attn.forward(x)
    ...
```

---

## Modifying Linear Layers

All linear layer variants are in `python/minisgl/layers/linear.py` and inherit from `_LinearTPImpl`.

| Class | Sharding | All-reduce | Use for |
|---|---|---|---|
| `LinearReplicated` | None (full weight per GPU) | No | Router/gate in MoE |
| `LinearColParallelMerged` | Output dim split across ranks | No | QKV, gate_up fused |
| `LinearQKVMerged` | Q sharded, KV replicated if GQA | No | QKV when GQA num_kv_heads < num_qo_heads |
| `LinearOProj` | Input dim split across ranks | Yes | Attention output projection |
| `LinearRowParallel` | Input dim split across ranks | Yes | MLP down projection |

### Adding a custom linear layer (e.g. with activation scale)

```python
import torch
import torch.nn.functional as F
from minisgl.layers.linear import _LinearTPImpl

class QuantizedLinear(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, tp_size: int):
        from minisgl.distributed import get_tp_info
        tp = get_tp_info()
        local_osize = output_size // tp.size
        super().__init__(input_size, output_size, input_size, local_osize, has_bias=False)
        # Additional tensors are tracked by state_dict automatically
        self.weight_scale = torch.empty(local_osize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dequantize on the fly, then matmul
        w = self.weight.to(torch.float16) * self.weight_scale.unsqueeze(1)
        return F.linear(x, w)
```

`weight` and `weight_scale` both appear in `state_dict()` and are loaded from the checkpoint. See the [quantization guide](03_quantization.md) for a full treatment.

---

## Modifying the Embedding Layer

`VocabParallelEmbedding` (`python/minisgl/layers/embedding.py`) shards the vocabulary across TP ranks using a custom indexing kernel. To add absolute positional embeddings or token type embeddings, add them as extra `BaseOP` attributes in your trunk model:

```python
class MyModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(...)
        self.embed_positions = torch.empty(config.max_position, config.hidden_size)  # replicated
        # ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        tok_emb = self.embed_tokens.forward(input_ids)
        pos_emb = self.embed_positions[ctx.batch.positions]
        x = tok_emb + pos_emb
        # ...
```

`embed_positions` is a plain tensor attribute → it is replicated across all ranks (each GPU holds the full matrix) and participates in `state_dict` / `load_state_dict` automatically.

---

## Key invariants to preserve

1. **Attribute names = weight keys.** The path `obj.sub.weight` produces the state_dict key `sub.weight` under whatever prefix the parent assigned. If you rename an attribute, the weight loader will not find the key.
2. **Underscore prefix = excluded from state_dict.** Use `self._layer_id = layer_id` for non-weight metadata.
3. **`OPList` indexes with integers.** `self.layers = OPList([...])` generates keys `layers.0`, `layers.1`, etc.
4. **Tensor shapes must match at `load_state_dict` time.** The model is instantiated on the `meta` device first (no memory allocated), then `load_state_dict` replaces the meta tensors with real ones. Shape and dtype are checked at that point.
