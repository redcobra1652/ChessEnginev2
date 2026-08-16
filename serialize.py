"""
serialize.py -- Convert NNUE checkpoints between formats.

Binary layout (all little-endian):
  [header]
    uint32  version       = 0x00000001
    uint32  hash          = SHA-256 of entire buffer (first 4 bytes, written last)
    uint32  desc_len
    char[]  description
  [feature transformer]
    int16[] ft_weights    shape: [HALFKP_SIZE, hidden_size]  — row-major
    int16[] ft_biases     shape: [hidden_size]
  [hidden layer 1]
    int8[]  l1_weights    shape: [32, hidden_size*2]         — row-major (out, in)
    int32[] l1_biases     shape: [32]
  [hidden layer 2]
    int8[]  l2_weights    shape: [32, 32]
    int32[] l2_biases     shape: [32]
  [output layer]
    int8[]  out_weights   shape: [num_buckets, 32]
    int32[] out_biases    shape: [num_buckets]

Quantisation scales (matching a fixed-point integer engine):
  FT activations are in [0, 1] (float clipped-relu).
  Multiply by FT_SCALE to get [0, 127] integer range.

  Layer bias scales encode the full accumulation path so the engine can use
  integer dot products without per-layer rescaling:
    l1 bias scale = FT_SCALE * L1_SCALE   (FT output is in [0, FT_SCALE])
    l2 bias scale = L1_SCALE * L2_SCALE   (L1 output is in [0, L1_SCALE])
    out bias scale = OUT_SCALE             (L2 output is in [0, L2_SCALE])
"""

import argparse
import hashlib
import os
import struct
import tempfile

import torch

import data
import model as M


# ── Quantisation helpers ─────────────────────────────────────────────────────

FT_SCALE  = 127      # FT weights/bias → int16;  FT activations ∈ [0, FT_SCALE]
L1_SCALE  = 64       # L1 weights → int8
L2_SCALE  = 64       # L2 weights → int8
OUT_SCALE = 600      # Output weights → int8


def _clamp(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return t.clamp(lo, hi).round().to(torch.int32)


def quantise_ft(weight: torch.Tensor, bias: torch.Tensor):
    """
    weight: [HALFKP_SIZE, hidden_size]  float32
    bias:   [hidden_size]               float32
    Returns bytes for int16 weight and int16 bias.
    """
    w = _clamp(weight * FT_SCALE, -32768, 32767).to(torch.int16)
    b = _clamp(bias   * FT_SCALE, -32768, 32767).to(torch.int16)
    return w.numpy().tobytes(), b.numpy().tobytes()


def quantise_l1(weight: torch.Tensor, bias: torch.Tensor):
    """
    weight: [32, hidden_size*2]  float32
    bias:   [32]                 float32

    Bias scale = FT_SCALE * L1_SCALE because in integer inference the
    accumulator values are integers in [0, FT_SCALE], so the bias must be
    pre-scaled by the same factor to stay in the same fixed-point domain.
    """
    w = _clamp(weight * L1_SCALE,             -128, 127).to(torch.int8)
    b = _clamp(bias   * FT_SCALE * L1_SCALE,  -(2**31), 2**31 - 1).to(torch.int32)
    return w.numpy().tobytes(), b.numpy().tobytes()


def quantise_l2(weight: torch.Tensor, bias: torch.Tensor):
    """
    weight: [32, 32]  float32
    bias:   [32]      float32

    Bias scale = L1_SCALE * L2_SCALE.
    """
    w = _clamp(weight * L2_SCALE,            -128, 127).to(torch.int8)
    b = _clamp(bias   * L1_SCALE * L2_SCALE, -(2**31), 2**31 - 1).to(torch.int32)
    return w.numpy().tobytes(), b.numpy().tobytes()


def quantise_out(weight: torch.Tensor, bias: torch.Tensor):
    """
    weight: [num_buckets, 32]  float32
    bias:   [num_buckets]      float32
    """
    w = _clamp(weight * OUT_SCALE, -128, 127).to(torch.int8)
    b = _clamp(bias   * OUT_SCALE, -(2**31), 2**31 - 1).to(torch.int32)
    return w.numpy().tobytes(), b.numpy().tobytes()


# ── .nnue writer ─────────────────────────────────────────────────────────────

def write_nnue(net: M.NNUE, path: str, description: str = "") -> bytes:
    """Quantise and write the network to a .nnue binary file."""
    net.eval()
    sd = {k: v.detach().cpu().float() for k, v in net.state_dict().items()}

    # state_dict() exposes "ft.bias" (renamed from the internal "ft_bias")
    ft_w  = sd["ft.weight"]      # [HALFKP_SIZE, hidden_size]
    ft_b  = sd["ft.bias"]        # [hidden_size]
    l1_w  = sd["l1.weight"]      # [32, hidden_size*2]
    l1_b  = sd["l1.bias"]        # [32]
    l2_w  = sd["l2.weight"]      # [32, 32]
    l2_b  = sd["l2.bias"]        # [32]
    out_w = sd["output.weight"]  # [num_buckets, 32]
    out_b = sd["output.bias"]    # [num_buckets]

    ft_w_b,  ft_b_b  = quantise_ft(ft_w,  ft_b)
    l1_w_b,  l1_b_b  = quantise_l1(l1_w,  l1_b)
    l2_w_b,  l2_b_b  = quantise_l2(l2_w,  l2_b)
    out_w_b, out_b_b = quantise_out(out_w, out_b)

    desc_bytes = description.encode("utf-8")

    # Write placeholder hash (0xDEADBEEF); patch it after the full buffer is built.
    buf  = struct.pack("<III", 0x00000001, 0xDEADBEEF, len(desc_bytes))
    buf += desc_bytes
    buf += ft_w_b  + ft_b_b
    buf += l1_w_b  + l1_b_b
    buf += l2_w_b  + l2_b_b
    buf += out_w_b + out_b_b

    sha = hashlib.sha256(buf).digest()[:4]
    buf = buf[:4] + sha + buf[8:]   # overwrite placeholder with real SHA

    with open(path, "wb") as f:
        f.write(buf)

    return buf


# ── .nnue reader ─────────────────────────────────────────────────────────────

def read_nnue(path: str, net: M.NNUE) -> str:
    """
    Load a .nnue binary file into net.  Returns the description string.

    Shape notes:
      ft.weight  : [HALFKP_SIZE, hidden_size]  — Embedding stores (vocab, embed_dim)
      l1.weight  : [32, hidden_size*2]         — Linear stores (out_features, in_features)
      output.weight : [num_buckets, 32]
    """
    with open(path, "rb") as f:
        raw = f.read()

    offset = 0

    def read_struct(fmt):
        nonlocal offset
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, raw, offset)
        offset += size
        return vals

    version, _hash, desc_len = read_struct("<III")
    if version != 0x00000001:
        raise ValueError(f"Unknown .nnue version 0x{version:08X}")
    description = raw[offset: offset + desc_len].decode("utf-8")
    offset += desc_len

    sd = {}
    H = net.hidden_size
    B = net.num_buckets

    def load_int16(name, shape):
        nonlocal offset
        n = 1
        for d in shape:
            n *= d
        arr = struct.unpack_from(f"<{n}h", raw, offset)
        offset += n * 2
        sd[name] = torch.tensor(arr, dtype=torch.float32).reshape(shape) / FT_SCALE

    def load_int8(name, shape, scale):
        nonlocal offset
        n = 1
        for d in shape:
            n *= d
        arr = struct.unpack_from(f"<{n}b", raw, offset)
        offset += n
        sd[name] = torch.tensor(arr, dtype=torch.float32).reshape(shape) / scale

    def load_int32(name, shape, scale):
        nonlocal offset
        n = 1
        for d in shape:
            n *= d
        arr = struct.unpack_from(f"<{n}i", raw, offset)
        offset += n * 4
        sd[name] = torch.tensor(arr, dtype=torch.float32).reshape(shape) / scale

    # ── Feature transformer ──────────────────────────────────────────────────
    # ft.weight is written row-major as [HALFKP_SIZE, hidden_size].
    # This matches nn.Embedding's (vocab_size, embed_dim) storage.
    load_int16("ft.weight", (data.HALFKP_SIZE, H))   # ← fixed: was (H, HALFKP_SIZE)
    load_int16("ft.bias",   (H,))

    # ── Hidden layers ────────────────────────────────────────────────────────
    load_int8 ("l1.weight", (32, H * 2),  L1_SCALE)
    load_int32("l1.bias",   (32,),        FT_SCALE * L1_SCALE)
    load_int8 ("l2.weight", (32, 32),     L2_SCALE)
    load_int32("l2.bias",   (32,),        L1_SCALE * L2_SCALE)

    # ── Output layer ─────────────────────────────────────────────────────────
    load_int8 ("output.weight", (B, 32),  OUT_SCALE)
    load_int32("output.bias",   (B,),     OUT_SCALE)

    net.load_state_dict(sd, strict=True)
    return description


# ── .pt helpers ──────────────────────────────────────────────────────────────

def load_pt(path: str, net: M.NNUE, map_location="cpu") -> int:
    """Load a .pt checkpoint into net.  Returns the epoch number (0 if unknown)."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict):
        if "state_dict" in obj:
            net.load_state_dict(obj["state_dict"])
            return obj.get("epoch", 0)
        else:
            net.load_state_dict(obj)
            return 0
    elif isinstance(obj, M.NNUE):
        net.load_state_dict(obj.state_dict())
        return 0
    else:
        raise ValueError(f"Unrecognised .pt contents: {type(obj)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Convert NNUE checkpoints between .pt and .nnue")
    p.add_argument("source",  help="Input file (.pt or .nnue)")
    p.add_argument("target",  help="Output file (.pt or .nnue), or a directory when --out-sha")
    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--buckets",     type=int, default=8)
    p.add_argument("--description", type=str, default="")
    p.add_argument("--out-sha",     action="store_true",
                   help="Name the output file nn-<sha12>.nnue inside target directory")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source file not found: {args.source}")

    src_ext = os.path.splitext(args.source)[1].lower()
    tgt_ext = os.path.splitext(args.target)[1].lower()

    print(f"Converting  {args.source}  →  {args.target}")
    net = M.NNUE(hidden_size=args.hidden, num_buckets=args.buckets)
    description = args.description

    if src_ext == ".pt":
        epoch = load_pt(args.source, net)
        print(f"  Loaded .pt (epoch {epoch})" if epoch else "  Loaded .pt weights")
    elif src_ext == ".nnue":
        description = read_nnue(args.source, net) or description
        print(f"  Loaded .nnue  description={description!r}")
    else:
        raise ValueError(f"Unsupported source format: {src_ext!r}")

    net.eval()

    if tgt_ext == ".pt":
        torch.save({"state_dict": net.state_dict(), "epoch": 0}, args.target)
        print(f"  Wrote {args.target}")

    elif tgt_ext == ".nnue" or args.out_sha:
        if args.out_sha:
            tmp = tempfile.mktemp(suffix=".nnue")
            buf = write_nnue(net, tmp, description=description)
            sha = hashlib.sha256(buf).hexdigest()[:12]
            out_dir = (args.target if os.path.isdir(args.target)
                       else os.path.dirname(os.path.abspath(args.target)) or os.getcwd())
            os.makedirs(out_dir, exist_ok=True)
            final = os.path.join(out_dir, f"nn-{sha}.nnue")
            os.replace(tmp, final)
            print(f"  Wrote {final}  (sha={sha})")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.target)) or ".", exist_ok=True)
            write_nnue(net, args.target, description=description)
            size_kb = os.path.getsize(args.target) / 1024
            print(f"  Wrote {args.target}  ({size_kb:.1f} KB)")
    else:
        raise ValueError(f"Unsupported target format: {tgt_ext!r}")


if __name__ == "__main__":
    main()
