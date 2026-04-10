import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import numpy as np
import shutil
from kernels.turbo_quant_mse import TurboQuantMSE
def run_comparison_quantize():
    INPUT = "./R1-1.5B-Raw"
    PATH_NAIVE = "./R1-1.5B-Naive"
    PATH_TURBO_8 = "./R1-1.5B-Turbo-8bit"
    PATH_TURBO_4 = "./R1-1.5B-Turbo-4bit"

    print("📁 Setting up output directories...")
    for path in [PATH_NAIVE, PATH_TURBO_8, PATH_TURBO_4]:
        if not os.path.exists(path): 
            os.makedirs(path)
        
        # Copy config files
        for item in os.listdir(INPUT):
            src_path = os.path.join(INPUT, item)
            if os.path.isfile(src_path) and not item.endswith(".safetensors"):
                shutil.copy2(src_path, os.path.join(path, item))

    print("🚀 Loading Raw Weights (Lazy)...")
    weights_path = os.path.join(INPUT, "model.safetensors")
    raw_weights = mx.load(weights_path)
    
    # --- PHASE 1: NAIVE ---
    print("\n🛠️  Phase 1: Generating Naive Quantization...")
    weights_naive = {}
    for key, w in raw_weights.items():
        w = w.astype(mx.float32)
        # ONLY quantize linear layers in the transformer blocks
        # Skip embeddings and LM head as they are too sensitive for 4-bit naive
        if ".layers." in key and "weight" in key and len(w.shape) == 2:
            # DYNAMIC SCALING: Use abs-max for 4-bit precision
            # (1 << 3) - 1 = 7, so we map [-max, max] to [-7, 7]
            amax = mx.abs(w).max() + 1e-8
            scale = amax / 7.0
            weights_naive[key] = mx.round(w / scale) * scale
        else:
            weights_naive[key] = w.astype(mx.float16)
        
        weights_naive[key] = weights_naive[key].astype(mx.float16)
        mx.eval(weights_naive[key]) # Force compute

    print("💾 Saving Naive Model...")
    mx.save_safetensors(os.path.join(PATH_NAIVE, "model.safetensors"), weights_naive)
    del weights_naive
    print("🧹 Naive weights cleared from memory.")

    # --- PHASE 2: TURBOQUANT 8-BIT ---
    print("\n💎 Phase 2: Generating TurboQuant 8-bit...")
    weights_turbo_8 = {}
    for key, w in raw_weights.items():
        w = w.astype(mx.float32)
        if ".layers." in key and "weight" in key and len(w.shape) == 2:
            print(f"   -> Processing: {key}")
            tq = TurboQuantMSE(dim=w.shape[-1], num_bits=8)
            indices, scale = tq.quantize(w)
            recon = tq.dequantize(indices, scale)
            weights_turbo_8[key] = recon.astype(mx.float16)
        else:
            weights_turbo_8[key] = w.astype(mx.float16)
        mx.eval(weights_turbo_8[key])

    print("💾 Saving TurboQuant 8-bit Model...")
    mx.save_safetensors(os.path.join(PATH_TURBO_8, "model.safetensors"), weights_turbo_8)
    del weights_turbo_8

    # --- PHASE 3: TURBOQUANT 4-BIT ---
    print("\n💎 Phase 3: Generating TurboQuant 4-bit...")
    weights_turbo_4 = {}
    for key, w in raw_weights.items():
        w = w.astype(mx.float32)
        if ".layers." in key and "weight" in key and len(w.shape) == 2:
            print(f"   -> Processing: {key}")
            tq = TurboQuantMSE(dim=w.shape[-1], num_bits=4)
            indices, scale = tq.quantize(w)
            recon = tq.dequantize(indices, scale)
            weights_turbo_4[key] = recon.astype(mx.float16)
        else:
            weights_turbo_4[key] = w.astype(mx.float16)
        mx.eval(weights_turbo_4[key])

    print("💾 Saving TurboQuant 4-bit Model...")
    mx.save_safetensors(os.path.join(PATH_TURBO_4, "model.safetensors"), weights_turbo_4)
    del weights_turbo_4
    
    print("\n✅ SUCCESS! Models are ready.")

if __name__ == "__main__":
    run_comparison_quantize()
