import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def attention_softmax_kernel(
    query_ptr, key_ptr, output_ptr,
    Q, K, D,
    stride_qz, stride_qh, stride_qn, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_oz, stride_oh, stride_on, stride_ok,
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

    # compile-time ranges
    offs_q = tl.arange(0, BLOCK_Q)      # (BLOCK_Q,)
    offs_k = tl.arange(0, BLOCK_K)      # (BLOCK_K,)
    offs_d = tl.arange(0, BLOCK_D)      # (BLOCK_D,)

    # masks for valid Q-rows and D-dims
    q_mask   = offs_q[:, None] < Q      # (BLOCK_Q,1)
    d_mask   = offs_d[None, :] < D      # (1, BLOCK_D)
    mask_qd  = q_mask & d_mask          # (BLOCK_Q, BLOCK_D)

    # 1) load Q-tile once
    q_ptrs = (query_ptr
            + q_off
            + offs_q[:, None] * stride_qn
            + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_qd, other=0.0)  # (BLOCK_Q, BLOCK_D)

    # prepare accumulators
    neg_inf = tl.cast(-1e9, tl.float32)
    row_max = tl.full((BLOCK_Q,), neg_inf, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # 2) loop over K in BLOCK_K chunks
    for k_start in range(0, K, BLOCK_K):
        # two masks for this chunk:
        #   mask_k_col: which columns < K?
        #   mask_k_row: which rows < K?  (for dimension mask)
        mask_k_col = offs_k[None, :] < (K - k_start)    # (1, BLOCK_K)
        mask_k_row = offs_k[:, None] < (K - k_start)    # (BLOCK_K, 1)
        mask_kd    = mask_k_row & d_mask                # (BLOCK_K, BLOCK_D)

        # load K-tile
        k_ptrs = (key_ptr
                + k_off
                + (k_start + offs_k)[:, None] * stride_kn
                + offs_d[None, :]         * stride_kd)
        k_tile = tl.load(k_ptrs, mask=mask_kd, other=0.0)  # (BLOCK_K, BLOCK_D)

        # raw logits
        logits = tl.dot(q, tl.trans(k_tile)) * scale      # (BLOCK_Q, BLOCK_K)

        # PASS 1: compute max per Q-row
        masked_logits = tl.where(mask_k_col, logits, neg_inf)
        row_max = tl.maximum(row_max, tl.max(masked_logits, axis=1))

        # PASS 2: accumulate sum of exps
        logits2    = logits - row_max[:, None]
        exp_logits = tl.exp(logits2) * mask_k_col
        sum_exp   += tl.sum(exp_logits, axis=1)

        # PASS 3: write normalized softmax
        softmax = exp_logits / sum_exp[:, None]           # (BLOCK_Q, BLOCK_K)
        o_ptrs   = (output_ptr
                  + o_off
                  + offs_q[:, None]         * stride_on
                  + (k_start + offs_k)[None, :] * stride_ok)
        tl.store(o_ptrs, softmax, mask=(q_mask & mask_k_col))


def prune_tokens_triton(
    query_states, key_states, value_states,
    keep_ratio: float,
    image_start_indices: torch.LongTensor,
    num_image_tokens_per_image: int,
):
    """
    Exactly mirrors the Python version:
     1. For each image chunk:
        a) run scaled dot-product + softmax
        b) average over heads & queries
        c) top-k within that chunk, shifted by image_start
     2. Append prefix & suffix indices
     3. Sort & index_select
    """
    B, H, Q, D = query_states.shape
    scale = D ** -0.5
    tokens_to_keep = round(num_image_tokens_per_image * keep_ratio)

    # Triton demands block dims >= 16 for its GEMM, so we pad with masks
    BLOCK_Q = max(16, min(32, Q))
    BLOCK_K = max(16, min(128, num_image_tokens_per_image))
    BLOCK_D = max(16, min(64, D))

    keep_list = []

    # loop over each image chunk
    for start in image_start_indices.tolist():
        K_chunk = num_image_tokens_per_image

        # allocate space for softmax’d attention: (B, H, Q, K_chunk)
        attn_weights = torch.empty((B, H, Q, K_chunk),
                                   device=query_states.device,
                                   dtype=query_states.dtype)

        # launch one kernel per (batch, head)
        attention_softmax_kernel[(B, H)](
            query_states, key_states[:, :, start:start+K_chunk, :], attn_weights,
            Q, K_chunk, D,
            *query_states.stride(),
            *key_states[:, :, start:start+K_chunk, :].stride(),
            *attn_weights.stride(),
            scale=scale,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # now on the host: average over heads & Q, then top-k
        # attn_weights: (B, H, Q, K_chunk)
        token_scores = attn_weights.mean(dim=1)    # (B, Q, K_chunk)
        token_scores = token_scores.squeeze(1).squeeze(0)  # (K_chunk,)

        topk = torch.topk(token_scores, tokens_to_keep, largest=True).indices
        keep_list.append(topk + start)

    # keep prefix (everything before first image block)
    first_start = image_start_indices[0].item()
    if first_start > 0:
        keep_list.append(torch.arange(0, first_start,
                                      device=key_states.device))

    # keep suffix (everything after last image block)
    last_end = image_start_indices[-1].item() + num_image_tokens_per_image
    total_K = key_states.shape[2]
    if last_end < total_K:
        keep_list.append(torch.arange(last_end, total_K,
                                      device=key_states.device))

    # concat, sort, and prune
    keep_indices = torch.cat(keep_list).sort().values
    pruned_key   = key_states.index_select(2, keep_indices)
    pruned_value = value_states.index_select(2, keep_indices)

    return pruned_key, pruned_value, keep_indices
