"""
model.py -- Simple HalfKP NNUE network.

Architecture (mirrors a Stockfish HalfKP NNUE):

  Input:  Two sets of HalfKP sparse indices (white POV + black POV).
          Each set has MAX_ACTIVE slots; -1 slots are padding and are ignored.

  FT:     Embedding lookup + sum → two vectors of shape [hidden_size]
          (one per king perspective). Concatenated → [hidden_size * 2].
          Activation: clipped ReLU (clamp to [0, 1]).

  L1:     Linear(hidden_size*2, 32) + clipped ReLU.
  L2:     Linear(32, 32) + clipped ReLU.

  Output: Linear(32, num_buckets) → scalar per bucket.
          During training, each position uses its material-count bucket.
          During inference, the engine selects the bucket by material count.

Forward signature:
  net(white_indices, black_indices, bucket=None, stm_white=None) → Tensor [B]

  stm_white controls accumulator ordering: STM perspective is concatenated first,
  matching search_core.c nnue_eval() and IncrementalAccumulator.evaluate().
  Pass a [B] bool tensor (True = White to move) during training, a scalar bool
  for single-side batches, or None for the legacy white-always-first path.

Indices are int64 tensors of shape [B, MAX_ACTIVE]; -1 = padding.

State-dict key layout (used by serialize.py):
  ft.weight      [HALFKP_SIZE, hidden_size]  — embedding table
  ft.bias        [hidden_size]               — accumulator bias (stored as plain Parameter)
  l1.weight      [32, hidden_size*2]
  l1.bias        [32]
  l2.weight      [32, 32]
  l2.bias        [32]
  output.weight  [num_buckets, 32]
  output.bias    [num_buckets]

Note: ft.bias is a plain nn.Parameter (not part of nn.Embedding, which has no
bias).  It is registered under the name "ft_bias" internally but exposed via
the state-dict as "ft.bias" so that serialize.py can address all FT parameters
under the "ft.*" namespace consistently.
"""

import torch
import torch.nn as nn

import data  # for HALFKP_SIZE, MAX_ACTIVE


def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, hidden_size: int = 256, num_buckets: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_buckets = num_buckets

        # Feature transformer — shared embedding table for both king perspectives
        self.ft = nn.Embedding(data.HALFKP_SIZE, hidden_size, padding_idx=None)
        nn.init.uniform_(self.ft.weight, -0.01, 0.01)

        # Accumulator bias — separate Parameter so we can zero-init it and
        # serialize it independently.  Exposed as "ft.bias" in the state dict.
        self.ft_bias = nn.Parameter(torch.zeros(hidden_size))

        # Hidden layer 1: takes concatenated white+black accumulators
        self.l1 = nn.Linear(hidden_size * 2, 32)
        nn.init.kaiming_uniform_(self.l1.weight, nonlinearity="relu")
        nn.init.zeros_(self.l1.bias)

        # Hidden layer 2
        self.l2 = nn.Linear(32, 32)
        nn.init.kaiming_uniform_(self.l2.weight, nonlinearity="relu")
        nn.init.zeros_(self.l2.bias)

        # Output: one logit per material bucket
        self.output = nn.Linear(32, num_buckets)
        nn.init.uniform_(self.output.weight, -0.01, 0.01)
        nn.init.zeros_(self.output.bias)

    # ── Accumulator ──────────────────────────────────────────────────────────

    def _accumulate(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Sparse feature lookup with padding mask.

        indices : [B, MAX_ACTIVE] int64  — -1 for padding slots
        Returns : [B, hidden_size]       — raw (pre-clipped) accumulator values.
                  Callers are responsible for applying clipped_relu before the
                  linear layers, so that STM ordering can be chosen first.

        Padding: -1 slots are clamped to 0 before the embedding lookup (to avoid
        an out-of-bounds index), then zeroed by the boolean mask.  Index 0 is a
        real feature, but the mask ensures its embedding is not accumulated for
        padding slots.
        """
        mask    = (indices >= 0).unsqueeze(-1).float()   # [B, MAX_ACTIVE, 1]
        safe_ix = indices.clamp(min=0)                   # map -1 → 0 (masked out below)
        emb     = self.ft(safe_ix)                       # [B, MAX_ACTIVE, H]
        acc     = (emb * mask).sum(dim=1) + self.ft_bias # [B, H]
        return acc  # raw; clipped_relu applied by forward() after STM ordering

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self,
                white_indices: torch.Tensor,
                black_indices: torch.Tensor,
                bucket: torch.Tensor | int | None = None,
                stm_white: torch.Tensor | bool | None = None) -> torch.Tensor:
        """
        white_indices, black_indices : [B, MAX_ACTIVE] int64
        stm_white : [B] bool tensor, scalar bool, or None.
          Determines which perspective goes first in the concatenated accumulator,
          matching search_core.c nnue_eval() and IncrementalAccumulator.evaluate():
            first  = STM perspective,  second = opponent.
          None (default) → always white-first (useful for quick tests / export).
          Pass stm_white=True/False for a full batch with one side to move, or
          pass a [B] bool tensor for mixed batches (the common training case).
        bucket:
          None          → mean over all buckets, returns [B]
          torch.Tensor  → per-sample bucket indices [B], returns [B]
          int           → single static bucket for the whole batch, returns [B]
        """
        w_acc = self._accumulate(white_indices)   # [B, H]  — raw
        b_acc = self._accumulate(black_indices)   # [B, H]  — raw

        if stm_white is None:
            # Legacy / export path: white-first, no reordering.
            first  = clipped_relu(w_acc)
            second = clipped_relu(b_acc)
        elif isinstance(stm_white, bool):
            if stm_white:
                first, second = clipped_relu(w_acc), clipped_relu(b_acc)
            else:
                first, second = clipped_relu(b_acc), clipped_relu(w_acc)
        else:
            # stm_white is a [B] bool tensor — reorder per sample.
            # Expand to [B, H] for masked selection.
            stm = stm_white.unsqueeze(-1).float()          # [B, 1]
            w_c, b_c = clipped_relu(w_acc), clipped_relu(b_acc)
            first  = stm * w_c + (1.0 - stm) * b_c        # [B, H]
            second = stm * b_c + (1.0 - stm) * w_c        # [B, H]

        x = torch.cat([first, second], dim=-1)    # [B, H*2]

        x = clipped_relu(self.l1(x))              # [B, 32]  — L1
        x = clipped_relu(self.l2(x))              # [B, 32]
        logits = self.output(x)                   # [B, num_buckets]

        if bucket is None:
            return logits.mean(dim=-1)
        elif isinstance(bucket, torch.Tensor):
            return logits.gather(1, bucket.long().unsqueeze(-1)).squeeze(-1)
        else:
            return logits[:, int(bucket)]

    # ── State dict ───────────────────────────────────────────────────────────
    # ft_bias is an nn.Parameter named "ft_bias" by PyTorch internally.
    # We expose it as "ft.bias" so serialize.py can group all FT tensors
    # under the "ft.*" prefix without any special-casing.

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Rename "ft_bias" → "ft.bias" for a clean ft.* namespace
        if "ft_bias" in sd:
            sd["ft.bias"] = sd.pop("ft_bias")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Accept both the canonical "ft.bias" key (from our own state_dict)
        # and the raw "ft_bias" key (in case someone passes the raw PyTorch dict)
        sd = dict(state_dict)
        if "ft.bias" in sd and "ft_bias" not in sd:
            sd["ft_bias"] = sd.pop("ft.bias")
        return super().load_state_dict(sd, strict=strict)
