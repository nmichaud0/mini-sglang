# Running Your Model with mini-sglang

Once your model is implemented and registered, there are two ways to run it: the **offline Python API** (no server) and the **HTTP server** (OpenAI-compatible REST API). Both use exactly the same engine underneath.

---

## Installation

```bash
pip install -e ".[dev]"          # editable install from repo root
# or just
pip install .
```

Dependencies: PyTorch with CUDA, FlashInfer, and optionally FlashAttention / TensorRT-LLM.

---

## Option 1 — Offline Python API (`LLM`)

Use this for scripts, notebooks, or testing. No server process needed.

```python
from minisgl import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",   # HuggingFace path or local dir
    tp_size=1,                                    # tensor parallelism degree
    dtype="bfloat16",
    attention_backend="auto",                     # "auto", "fa", "fi", "fa,fi", "trtllm"
)

outputs = llm.generate(
    prompts=["The capital of France is", "Write a haiku about rain"],
    sampling_params=SamplingParams(temperature=0.7, max_new_tokens=128),
)

for out in outputs:
    print(out["text"])
```

`SamplingParams` fields:
```python
SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,           # -1 = disabled
    max_new_tokens=128,
)
```

For tensor parallelism with `tp_size > 1`, mini-sglang spawns subprocess workers internally — you do not need to manage `torch.distributed` yourself.

---

## Option 2 — HTTP Server

### Start the server

```bash
python -m minisgl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 1 \
    --dtype bfloat16 \
    --port 8000
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID or local path |
| `--tp-size` | 1 | Number of GPUs for tensor parallelism |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` |
| `--attention-backend` | `auto` | `auto`, `fa`, `fi`, `fa,fi`, `trtllm` |
| `--port` | 8000 | HTTP port |
| `--memory-ratio` | 0.9 | Fraction of GPU memory for KV cache |
| `--page-size` | 1 | KV cache page size (must be 16/32/64 for trtllm) |
| `--max-running-req` | — | Max concurrent requests |
| `--use-dummy-weight` | False | Skip weight loading (for architecture testing) |

### Query the server (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

Or with `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

---

## Multi-GPU (Tensor Parallel)

```bash
python -m minisgl \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tp-size 4 \
    --dtype bfloat16
```

mini-sglang spawns one engine subprocess per GPU. Weight sharding happens automatically inside the weight loader. You don't need to do anything special in your model code beyond using the correct parallel linear layer types.

---

## Using a local or modified model

If your model is stored locally (e.g. you have modified the weights or used a custom saver):

```bash
python -m minisgl --model /path/to/my/model
```

The path must contain:
- `config.json` (with `"architectures": ["MyForCausalLM"]`)
- One or more `.safetensors` files

If the architecture string in `config.json` matches an entry in `_MODEL_REGISTRY`, mini-sglang will load it.

---

## Testing a new architecture without real weights

During development, use `--use-dummy-weight` to skip the weight download and just fill tensors with random values. This is useful for verifying that your model's forward pass runs without errors:

```bash
python -m minisgl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --use-dummy-weight \
    --port 8000
```

Or from Python:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    use_dummy_weight=True,
)
outputs = llm.generate(["hello"], SamplingParams(max_new_tokens=5))
# Output will be garbage, but the forward pass runs
```

---

## How the engine processes a request (what happens under the hood)

```
Request arrives
    │
    ▼
Scheduler: add to queue
    │
    ▼
Prefill phase: process all input tokens at once
    │  model.forward() reads batch.input_ids from global ctx
    │  attention backend fills KV cache
    ▼
Decode phase: one token per step per request
    │  model.forward() reads the single new token
    │  attention backend reads from KV cache
    ▼
Sampler: sample next token from logits
    │
    ▼
Token appended to request, repeat decode until EOS or max_new_tokens
```

The `Context` object (`minisgl.core.get_global_ctx()`) is the shared bus. It holds:
- `ctx.batch` — current batch being processed (input_ids, positions, output locations)
- `ctx.kv_cache` — paged KV cache pool
- `ctx.attn_backend` — attention backend (FlashInfer, FlashAttention, TRT-LLM)
- `ctx.page_table` — maps (request_idx, token_pos) → KV cache slot

Your model's `forward()` reads `get_global_ctx().batch.input_ids` directly — you never pass tensors as arguments.

---

## Attention backend selection

| Backend | Hardware | Notes |
|---|---|---|
| `auto` | any | Selects best for detected GPU (SM90 → `fa,fi`, SM100 → `trtllm`) |
| `fi` | any | FlashInfer for both prefill and decode |
| `fa` | any | FlashAttention for prefill, FlashInfer for decode |
| `fa,fi` | SM90 (H100) | FlashAttention prefill + FlashInfer decode (recommended for H100) |
| `trtllm` | SM100 (B200) | TensorRT-LLM decode kernel (requires `page_size` ∈ {16, 32, 64}) |

---

## CUDA graph capture

The engine automatically captures CUDA graphs for the decode phase at batch sizes it expects to see. This eliminates Python overhead on the critical decode path. Captured batch sizes are configured via `--cuda-graph-bs` (defaults are chosen automatically).

CUDA graph replay only works when:
- The batch is in decode phase (not prefill)
- The batch size is one of the captured sizes

Prefill always runs eagerly.

---

## Debugging and profiling

```bash
# Verbose logging
MINISGL_LOG_LEVEL=DEBUG python -m minisgl --model ...

# NVTX annotations (visible in Nsight Systems)
# Already applied via @nvtx_annotate on each layer and attention/MLP forward
nsys profile python -m minisgl --model ...
```

The `@nvtx_annotate` decorator on decoder layers produces NVTX ranges labeled `Layer_0`, `Layer_1`, etc. Attention is labeled `MHA` and MLP is labeled `MLP`.
