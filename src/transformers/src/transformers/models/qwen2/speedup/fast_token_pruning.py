import torch
import torch.nn.functional as F

# cache for a reusable attention‐weight buffer
_attn_buffer = {}

def _get_attn_buffer(shape, dtype, device):
    key = (shape, dtype, device)
    buf = _attn_buffer.get(key)
    if buf is None or buf.shape != shape:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _attn_buffer[key] = buf
    return buf

def prune_tokens_fast(
    query_states: torch.Tensor,    # (B, H, Q, D)
    key_states:   torch.Tensor,    # (B, H, K, D)
    value_states: torch.Tensor,    # (B, H, K, D)
    keep_ratio:   float,
    image_start_indices: torch.LongTensor,
    num_image_tokens_per_image: int,
):
    B, H, Q, D = query_states.shape
    K_chunk = num_image_tokens_per_image
    device = query_states.device
    dtype  = query_states.dtype

    tokens_to_keep = round(K_chunk * keep_ratio)
    attn_w = _get_attn_buffer((B, H, Q, K_chunk), dtype, device)

    keep_list = []

    # flatten dims for a single batched bmm
    BH = B * H
    q_flat = query_states.reshape(BH, Q, D)

    for start in image_start_indices.tolist():
        # slice out K‐ and V‐chunks
        k_chunk = key_states  [..., start:start+K_chunk, :].reshape(BH, K_chunk, D)
        # raw matmul: (BH, Q, D) × (BH, D, K_chunk) → (BH, Q, K_chunk)
        raw = torch.bmm(q_flat, k_chunk.transpose(-1, -2)) * (D ** -0.5)
        # softmax along last dim
        w_flat = F.softmax(raw, dim=-1)
        # reshape back to (B, H, Q, K_chunk)
        attn_w.copy_(w_flat.view(B, H, Q, K_chunk))

        # reduction to (B, K_chunk): mean over heads, then over Q
        scores = attn_w.mean(dim=1).mean(dim=1)  # (B, K_chunk)
        # top-k per batch
        topk = torch.topk(scores, tokens_to_keep, dim=-1, largest=True).indices
        # offset and collect
        keep_list.append((topk + start).reshape(-1))

    # prefix
    first = image_start_indices[0].item()
    if first > 0:
        keep_list.append(torch.arange(first, device=device))

    # suffix
    last_end = image_start_indices[-1].item() + K_chunk
    total_K = key_states.size(2)
    if last_end < total_K:
        keep_list.append(torch.arange(last_end, total_K, device=device))

    # concat, sort & dedupe
    keep_indices = torch.cat(keep_list).unique(sorted=True)

    pruned_key   = key_states  .index_select(2, keep_indices)
    pruned_value = value_states.index_select(2, keep_indices)
    return pruned_key, pruned_value, keep_indices
