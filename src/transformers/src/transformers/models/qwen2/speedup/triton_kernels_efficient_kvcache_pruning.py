import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ——————————————————————————————————————————————————————————————
# 1) Your Triton softmax kernel
@triton.jit
def attention_softmax_kernel(
    query_ptr, key_ptr, output_ptr,
    Q, K, D,
    stride_qz, stride_qh, stride_qn, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_oz, stride_oh, stride_on,  stride_ok,
    scale: tl.constexpr,   # = 1/√D
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    q_off = batch_id * stride_qz + head_id * stride_qh
    k_off = batch_id * stride_kz + head_id * stride_kh
    o_off = batch_id * stride_oz + head_id * stride_oh

    offs_q = tl.arange(0, BLOCK_Q)   # (BLOCK_Q,)
    offs_k = tl.arange(0, BLOCK_K)   # (BLOCK_K,)
    offs_d = tl.arange(0, BLOCK_D)   # (BLOCK_D,)

    q_mask  = offs_q[:, None] < Q    # (BLOCK_Q,1)
    d_mask  = offs_d[None, :] < D    # (1,BLOCK_D)
    mask_qd = q_mask & d_mask        # (BLOCK_Q, BLOCK_D)

    # load Q-tile
    q_ptrs = query_ptr \
           + q_off \
           + offs_q[:, None] * stride_qn \
           + offs_d[None, :]   * stride_qd
    q = tl.load(q_ptrs, mask=mask_qd, other=0.0)   # (BLOCK_Q, BLOCK_D)

    neg_inf = tl.cast(-1e9, tl.float32)
    row_max = tl.full((BLOCK_Q,), neg_inf, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,),   dtype=tl.float32)

    # loop over K in BLOCK_K chunks
    for k_start in range(0, K, BLOCK_K):
        mask_k_col = offs_k[None, :] < (K - k_start)  # (1, BLOCK_K)
        mask_k_row = offs_k[:, None] < (K - k_start)  # (BLOCK_K,1)
        mask_kd    = mask_k_row & d_mask              # (BLOCK_K, BLOCK_D)

        k_ptrs = key_ptr \
               + k_off \
               + (k_start + offs_k)[:, None] * stride_kn \
               + offs_d[None, :]           * stride_kd
        k_tile = tl.load(k_ptrs, mask=mask_kd, other=0.0)  # (BLOCK_K, BLOCK_D)

        logits = tl.dot(q, tl.trans(k_tile)) * scale       # (BLOCK_Q, BLOCK_K)

        # PASS 1: update max
        masked_logits = tl.where(mask_k_col, logits, neg_inf)
        row_max = tl.maximum(row_max, tl.max(masked_logits, axis=1))

        # PASS 2: accumulate exp
        logits2    = logits - row_max[:, None]
        exp_logits = tl.exp(logits2) * mask_k_col
        sum_exp   += tl.sum(exp_logits, axis=1)

        # PASS 3: write softmax
        softmax = exp_logits / sum_exp[:, None]            # (BLOCK_Q, BLOCK_K)
        o_ptrs   = output_ptr \
                 + o_off \
                 + offs_q[:, None]       * stride_on \
                 + (k_start + offs_k)[None, :] * stride_ok
        tl.store(o_ptrs, softmax, mask=(q_mask & mask_k_col))


# ——————————————————————————————————————————————————————————————
# 2) The pruning function (returns pruned K, V and the keep‐indices)
def prune_tokens_triton(
    query_states: torch.Tensor,    # (B, H, Q, D)
    key_states:   torch.Tensor,    # (B, H, K, D)
    value_states: torch.Tensor,    # (B, H, K, D)
    keep_ratio:   float,
    image_start_indices: torch.LongTensor,
    num_image_tokens_per_image: int,
):
    B, H, Q, D = query_states.shape
    scale = D ** -0.5
    tokens_to_keep = round(num_image_tokens_per_image * keep_ratio)

    BLOCK_Q = max(16, min(32, Q))
    BLOCK_K = max(16, min(128, num_image_tokens_per_image))
    BLOCK_D = max(16, min(64, D))

    keep_list = []

    # scratch buffer for each chunk’s softmax weights
    attn_weights = torch.empty((B, H, Q, num_image_tokens_per_image),
                               device=query_states.device,
                               dtype=query_states.dtype)

    # run Triton‐softmax per image chunk
    for start in image_start_indices.tolist():
        Kc = num_image_tokens_per_image
        attention_softmax_kernel[(B, H)](
            query_states,
            key_states[:, :, start : start+Kc, :],
            attn_weights,
            Q, Kc, D,
            *query_states.stride(),
            *key_states[:, :, start : start+Kc, :].stride(),
            *attn_weights.stride(),
            scale=scale,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # average over heads & queries → (B, Kc)
        scores = attn_weights.mean(dim=1).mean(dim=1)  # (B, Kc)
        topk   = torch.topk(scores, tokens_to_keep, dim=-1, largest=True).indices
        # flatten across batch
        keep_list.append((topk + start).view(-1))

    # prefix
    first = image_start_indices[0].item()
    if first > 0:
        keep_list.append(torch.arange(0, first, device=key_states.device))
    # suffix
    last_end = image_start_indices[-1].item() + num_image_tokens_per_image
    K_tot   = key_states.size(2)
    if last_end < K_tot:
        keep_list.append(torch.arange(last_end, K_tot, device=key_states.device))

    keep_indices = torch.cat(keep_list).unique(sorted=True)

    pruned_k = key_states  .index_select(2, keep_indices)  # (B, H, K', D)
    pruned_v = value_states.index_select(2, keep_indices)  # (B, H, K', D)
    return pruned_k, pruned_v, keep_indices


# ——————————————————————————————————————————————————————————————
# 3) The “middle-ground” final attention:
#    we never call `_scaled_attn` again—just two bmm’s on the pruned set.
def prune_and_attend(
    query_states: torch.Tensor,    # (B, H, Q, D)
    key_states:   torch.Tensor,    # (B, H, K, D)
    value_states: torch.Tensor,    # (B, H, K, D)
    keep_ratio:   float,
    image_start_indices: torch.LongTensor,
    num_image_tokens_per_image: int,
):
    # 1) prune
    pruned_k, pruned_v, keep_idx = prune_tokens_triton(
        query_states, key_states, value_states,
        keep_ratio, image_start_indices, num_image_tokens_per_image,
    )
    B, H, Q, D = query_states.shape
    Kp = pruned_k.size(2)

    # 2) flatten batch+heads
    BH = B * H
    q_flat = query_states.reshape(BH, Q, D)
    k_flat = pruned_k      .reshape(BH, Kp, D)
    v_flat = pruned_v      .reshape(BH, Kp, D)

    # 3) QKᵀ
    raw = torch.bmm(q_flat, k_flat.transpose(-1, -2)) * (D ** -0.5)
    # 4) softmax
    w   = F.softmax(raw, dim=-1)                         # (BH, Q, Kp)
    # 5) weighted sum
    out = torch.bmm(w, v_flat)                           # (BH, Q, D)

    # 6) reshape back → (B, H, Q, D)
    attn_output = out.view(B, H, Q, D)
    return attn_output, keep_idx

# ——————————————————————————————————————————————————————————————
# 4) Usage:
#    replace your two‐step:
#
#      key_states, value_states, _ = prune_tokens_triton(...)
#      attn_output = _scaled_attn(...)
#
#    with:
#
#      attn_output, keep_idx = prune_and_attend(
#          query_states, key_states, value_states,
#          keep_ratio, image_start_indices, num_image_tokens_per_image
#      )
#
#    Now you do *one* Triton‐softmax pass (for pruning) plus
#    two very efficient bmm’s on the *much smaller* pruned set.
