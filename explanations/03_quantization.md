# Layer-by-Layer Quantization

This guide explains how to implement quantization inside mini-sglang's `BaseOP` class system — the approach that fits how the engine loads weights and runs the forward pass.

---

## How weight loading works (the constraint that matters)

The engine does this at startup (`engine.py:52-53`):

```python
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)           # allocates on meta device
self.model.load_state_dict(self._load_weight_state_dict(config))  # replaces with real tensors
```

`load_state_dict` walks the model's `__dict__`, matches tensor attribute names to keys in the state_dict, checks shape+dtype, then replaces each tensor. This means:

- **Every tensor you declare becomes a weight key.** If you have `self.weight` and `self.weight_scale`, both are loaded.
- **Shape must match exactly at load time.** The meta-device instantiation must produce tensors of the correct final shape (the shape they will have after sharding).
- **dtype is applied by the engine** (`v.to(self.dtype)`) before passing to `load_state_dict` — unless you intercept earlier.

---

## Strategy: wrap `_LinearTPImpl` with quantized tensors

The most natural approach is to subclass `_LinearTPImpl` (from `python/minisgl/layers/linear.py`), replace `self.weight` with a quantized tensor, and add scale/zero tensors alongside it.

### Example: INT8 weight-only quantization (per-output-channel)

```python
# python/minisgl/layers/linear_int8.py

from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class LinearRowParallelInt8(BaseOP):
    """
    Row-parallel linear with INT8 weight-only quantization.
    Each rank holds its input shard as int8, with a per-output-channel fp16 scale.
    All-reduced after dequant+matmul.
    """

    def __init__(self, input_size: int, output_size: int):
        tp = get_tp_info()
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()

        local_input = div_even(input_size, tp.size)
        # Declare the quantized weight and its scale as plain tensor attributes.
        # These names must match what the weight loader will produce.
        self.weight = torch.empty(output_size, local_input, dtype=torch.int8)
        self.weight_scale = torch.empty(output_size, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: broadcast scale over input dim
        w_fp = self.weight.to(x.dtype) * self.weight_scale.unsqueeze(1)
        y = F.linear(x, w_fp)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
```

Replace `LinearRowParallel` with `LinearRowParallelInt8` in `GatedMLP.down_proj` for the MLP down projection.

---

## Providing quantized weights from a checkpoint

The weight loader (`python/minisgl/models/weight.py`) calls `_shard_tensor` for sharding and then passes tensors through as-is. If your checkpoint already stores quantized weights (e.g. exported with `torch.quantize_per_channel`), you need to:

1. **Store int8 weights in safetensors** (int8 tensors are supported).
2. **Make sure the key names match** (e.g. `model.layers.0.mlp.down_proj.weight` and `model.layers.0.mlp.down_proj.weight_scale`).
3. **Handle sharding for the scale tensor** — add the scale key to `_SPLIT_DIM_0` or `_SPLIT_DIM_1` in `weight.py` as appropriate.

```python
# weight.py — extend sharding lists
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj", ".down_proj_scale"]  # add scale key

# The scale for a row-parallel layer shards along output dim (dim 0):
_SPLIT_DIM_0 += [".down_proj_scale"]  # if scale is [output_size]
```

---

## Quantizing layer-by-layer at model load time (post-hoc)

If you want to **quantize a full-precision checkpoint** at load time rather than loading a pre-quantized checkpoint, intercept the weights inside `load_state_dict`:

```python
class LinearRowParallelInt8(BaseOP):
    def __init__(self, input_size: int, output_size: int):
        tp = get_tp_info()
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        local_input = div_even(input_size, tp.size)
        # Declare with the final dtypes — the engine checks shape+dtype at load time
        self.weight = torch.empty(output_size, local_input, dtype=torch.int8)
        self.weight_scale = torch.empty(output_size, dtype=torch.float16)

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        key = f"{prefix}.weight" if prefix else "weight"
        fp_weight = state_dict.pop(key)                     # fp16 weight from checkpoint

        # Quantize: per-output-channel INT8
        scale = fp_weight.abs().max(dim=1).values / 127.0   # [output_size]
        q_weight = (fp_weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        self.weight = q_weight
        self.weight_scale = scale.to(torch.float16)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys: {list(state_dict.keys())}")
```

The key insight: override `load_state_dict` to consume the fp16 weight key, quantize in-place, and set `self.weight` and `self.weight_scale` without them needing to exist in the checkpoint.

---

## Layer-by-layer control: different quantization per layer

Because each decoder layer is a separate `BaseOP` instance, you can choose quantization per layer. The cleanest approach is to pass a per-layer config:

```python
class MyDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        # Quantize MLP down/up in deeper layers only
        quantize_mlp = layer_id >= config.num_layers // 2

        if quantize_mlp:
            self.mlp = GatedMLPInt8(config)
        else:
            self.mlp = GatedMLP(config)

        self.self_attn = RopeAttn(config, layer_id)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, residual=None):
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual
```

`GatedMLPInt8` would be a copy of `GatedMLP` that uses `LinearRowParallelInt8` for `down_proj`.

---

## Quantizing the QKV projection

The QKV projection uses `LinearQKVMerged`, which handles GQA sharding. For INT8 on QKV:

```python
class LinearQKVMergedInt8(BaseOP):
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads):
        tp = get_tp_info()
        local_qo = div_even(num_qo_heads, tp.size)
        local_kv = div_even(num_kv_heads, tp.size, allow_replicate=True)
        local_osize = (local_qo + 2 * local_kv) * head_dim

        self.weight = torch.empty(local_osize, hidden_size, dtype=torch.int8)
        self.weight_scale = torch.empty(local_osize, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp = self.weight.to(x.dtype) * self.weight_scale.unsqueeze(1)
        return F.linear(x, w_fp)
```

Note: the QKV projection is column-parallel — no all-reduce needed. The output is consumed by the attention layer which then does its own communication via the KV cache.

---

## INT4 / GPTQ / AWQ: packed weights

For sub-8-bit quantization, weights are typically stored packed (2 INT4 values per byte). The approach is the same but you need an unpacking step:

```python
class LinearRowParallelInt4(BaseOP):
    def __init__(self, input_size, output_size, group_size=128):
        tp = get_tp_info()
        local_input = div_even(input_size, tp.size)
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        self._group_size = group_size

        num_groups = local_input // group_size
        # Packed: 2 int4 values per byte
        self.weight = torch.empty(output_size, local_input // 2, dtype=torch.uint8)
        self.weight_scale = torch.empty(output_size, num_groups, dtype=torch.float16)
        self.weight_zero = torch.empty(output_size, num_groups, dtype=torch.float16)

    def _unpack(self) -> torch.Tensor:
        # Each byte holds two 4-bit values
        low = (self.weight & 0x0F).to(torch.int8) - 8
        high = ((self.weight >> 4) & 0x0F).to(torch.int8) - 8
        return torch.stack([low, high], dim=-1).reshape(self.weight.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_int = self._unpack()                              # [out, in]
        # Dequantize with group scales
        gs = self._group_size
        w_fp = (w_int.reshape(w_int.shape[0], -1, gs).to(x.dtype)
                * self.weight_scale.unsqueeze(-1)
                + self.weight_zero.unsqueeze(-1)).reshape(w_int.shape)
        y = F.linear(x, w_fp)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
```

---

## Summary of the pattern

Every quantized layer follows the same template:

1. **Declare quantized weight tensors** (`weight: int8`, `weight_scale: fp16`, etc.) as plain attributes.
2. **Optionally override `load_state_dict`** to quantize on-the-fly from fp16 checkpoint weights.
3. **Override `forward`** to dequantize and compute.
4. **Preserve all-reduce calls** for row-parallel layers.
5. **Match the state_dict key names** so the weight loader finds them.

The engine never sees quantization — it just calls `model.forward()` and the quantized layers handle dequantization internally.
