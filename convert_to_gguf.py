#!/usr/bin/env python
"""Convert Asterisk PyTorch weights to GGUF.

Requires gguf (from llama.cpp) and torch:
  pip install torch
  pip install git+https://github.com/ggerganov/llama.cpp#subdirectory=gguf-py
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np

try:
    import gguf
except ImportError:
    sys.stderr.write("gguf not installed. Install with: pip install git+https://github.com/ggerganov/llama.cpp#subdirectory=gguf-py\n")
    sys.exit(1)


QUANT_MAP = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

BLOCK_SIZE = 32  # Q8_0 and Q4_0 use 32-element blocks


def quantize_q8_0(arr: np.ndarray) -> np.ndarray:
    """Quantize float32 array to Q8_0 format.
    
    Q8_0 block (34 bytes): 2 bytes fp16 scale + 32 int8 values
    """
    arr = arr.astype(np.float32).flatten()
    n = len(arr)
    # Pad to multiple of block size
    padded_n = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_n > n:
        arr = np.pad(arr, (0, padded_n - n))
    
    arr = arr.reshape(-1, BLOCK_SIZE)
    n_blocks = arr.shape[0]
    
    # Compute scale per block (max abs value / 127)
    amax = np.abs(arr).max(axis=1, keepdims=True)
    scales = amax / 127.0
    scales = np.where(scales == 0, 1.0, scales)  # Avoid div by zero
    
    # Quantize to int8
    quants = np.round(arr / scales).astype(np.int8)
    
    # Pack: fp16 scale + 32 int8 values per block
    scales_f16 = scales.astype(np.float16).flatten()
    
    # Create output buffer: 34 bytes per block
    out = np.zeros((n_blocks, 34), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = quants.view(np.uint8)
    
    return out.flatten()


def quantize_q4_0(arr: np.ndarray) -> np.ndarray:
    """Quantize float32 array to Q4_0 format.
    
    Q4_0 block (18 bytes): 2 bytes fp16 scale + 16 bytes (32 4-bit values)
    """
    arr = arr.astype(np.float32).flatten()
    n = len(arr)
    padded_n = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_n > n:
        arr = np.pad(arr, (0, padded_n - n))
    
    arr = arr.reshape(-1, BLOCK_SIZE)
    n_blocks = arr.shape[0]
    
    # Compute scale per block
    amax = np.abs(arr).max(axis=1, keepdims=True)
    scales = amax / 7.0  # Q4 uses -8 to 7 range
    scales = np.where(scales == 0, 1.0, scales)
    
    # Quantize to 4-bit (stored as int8, then packed)
    quants = np.round(arr / scales).clip(-8, 7).astype(np.int8) + 8  # Offset to 0-15
    
    # Pack two 4-bit values per byte
    quants = quants.reshape(n_blocks, 16, 2)
    packed = (quants[:, :, 0] & 0x0F) | ((quants[:, :, 1] & 0x0F) << 4)
    
    # Create output buffer: 18 bytes per block
    scales_f16 = scales.astype(np.float16).flatten()
    out = np.zeros((n_blocks, 18), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed.astype(np.uint8)
    
    return out.flatten()


def main():
    p = argparse.ArgumentParser(description="Convert Asterisk model.pt to GGUF")
    p.add_argument("--pt", default="build/model.pt", help="Path to model.pt")
    p.add_argument("--out", default="dist/model.gguf", help="Output GGUF path")
    p.add_argument("--quant", default="q8_0", choices=QUANT_MAP.keys(), help="Quantization type")
    args = p.parse_args()

    state = torch.load(args.pt, map_location="cpu")
    # Handle state dict wrapped in checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qtype = QUANT_MAP[args.quant]

    writer = gguf.GGUFWriter(out_path.as_posix(), arch="custom")
    writer.add_name("asterisk")
    writer.add_file_type(qtype)

    # Basic metadata
    writer.add_uint32("asterisk.hidden_size", 512)
    writer.add_uint32("asterisk.layers", 2)
    writer.add_uint32("asterisk.heads", 2)
    writer.add_uint32("asterisk.output_dim", 256)

    # Store original tensor shapes for quantized formats (needed for dequantization)
    tensor_shapes = {}
    
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        arr = tensor.cpu().numpy().astype(np.float32)
        orig_shape = arr.shape
        tensor_shapes[name] = orig_shape
        
        # Quantize based on selected type
        if args.quant == "q8_0":
            # Q8_0: 34 bytes per block of 32 elements
            # Don't pass raw_shape - gguf validates using it but expects quantized dims
            if arr.ndim == 1:
                n = len(arr)
                padded_n = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if padded_n > n:
                    arr = np.pad(arr, (0, padded_n - n))
                quantized = quantize_q8_0(arr)  # Returns flat uint8 array
                writer.add_tensor(name, quantized, raw_dtype=qtype)
            else:
                # 2D tensor: flatten, quantize, reshape
                rows, cols = arr.shape
                padded_cols = ((cols + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if padded_cols > cols:
                    arr = np.pad(arr, ((0, 0), (0, padded_cols - cols)))
                n_blocks_per_row = padded_cols // BLOCK_SIZE
                bytes_per_row = n_blocks_per_row * 34
                quantized = np.zeros((rows, bytes_per_row), dtype=np.uint8)
                for i in range(rows):
                    quantized[i] = quantize_q8_0(arr[i])
                # Flatten and store - shape is stored in metadata
                writer.add_tensor(name, quantized.flatten(), raw_dtype=qtype)
        elif args.quant == "q4_0":
            # Q4_0: 18 bytes per block of 32 elements
            if arr.ndim == 1:
                n = len(arr)
                padded_n = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if padded_n > n:
                    arr = np.pad(arr, (0, padded_n - n))
                quantized = quantize_q4_0(arr)
                writer.add_tensor(name, quantized, raw_dtype=qtype)
            else:
                rows, cols = arr.shape
                padded_cols = ((cols + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                if padded_cols > cols:
                    arr = np.pad(arr, ((0, 0), (0, padded_cols - cols)))
                n_blocks_per_row = padded_cols // BLOCK_SIZE
                bytes_per_row = n_blocks_per_row * 18
                quantized = np.zeros((rows, bytes_per_row), dtype=np.uint8)
                for i in range(rows):
                    quantized[i] = quantize_q4_0(arr[i])
                writer.add_tensor(name, quantized.flatten(), raw_dtype=qtype)
        elif args.quant == "f16":
            writer.add_tensor(name, arr.astype(np.float16), raw_dtype=qtype)
        else:  # f32
            writer.add_tensor(name, arr, raw_dtype=qtype)
    
    # Store original shapes for quantized tensors as metadata
    if args.quant in ("q8_0", "q4_0"):
        for name, shape in tensor_shapes.items():
            shape_str = ",".join(str(d) for d in shape)
            writer.add_string(f"asterisk.shape.{name}", shape_str)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"✅ Wrote GGUF to {out_path} (quant={args.quant})")


if __name__ == "__main__":
    main()
