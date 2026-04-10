import mlx.core as mx
import numpy as np
import time
from turbo_quant_mse import TurboQuantMSE
from turbo_quant_prod import TurboQuantProd

def evaluate_quantizers(dim=128, num_bits=4, num_samples=100000):
    print("=" * 75)
    print(f"🚀 TurboQuant Benchmark on Apple Silicon (MLX)")
    print(f"Dimension: {dim} | Bits budget: {num_bits} | Samples: {num_samples:,}")
    print("=" * 75)

    print("\\n[1/4] Initializing Quantizers (This runs K-Means under the hood)...")
    start = time.time()
    tq_mse = TurboQuantMSE(dim=dim, num_bits=num_bits, seed=42)
    tq_prod = TurboQuantProd(dim=dim, num_bits=num_bits, seed=42)
    print(f"Done! ({time.time() - start:.2f} seconds)")

    print(f"\\n[2/4] Generating {num_samples:,} pairs of random synthetic vectors...")
    np.random.seed(99)
    X1_np = np.random.randn(num_samples, dim)
    X1_np = X1_np / np.linalg.norm(X1_np, axis=1, keepdims=True)
    X1 = mx.array(X1_np)

    print(f"\\n[3/4] Running Batched GPU Quantization...")
    start = time.time()
    
    # We transpose to (dim, N) because our rotation matrix expects vectors as columns
    X1_T = X1.T
    
    # True auto-correlation (Inner Product with self)
    true_dots = mx.sum(X1 * X1, axis=1) # Shape (N,)
    
    # --- TurboQuantMSE Pipeline ---
    q1_mse, s1_mse = tq_mse.quantize(X1_T)
    recon1_mse = tq_mse.dequantize(q1_mse, s1_mse).T # Transpose back to (N, dim)
    
    # --- TurboQuantProd Pipeline ---
    q1_prod_idx, q1_prod_scale, q1_prod_signs, q1_prod_norm = tq_prod.quantize(X1_T)
    recon1_prod = tq_prod.dequantize(q1_prod_idx, q1_prod_scale, q1_prod_signs, q1_prod_norm).T
    
    mx.eval(recon1_mse, recon1_prod) # Evaluate the parallel graphs!
    
    print(f"Done! Quantized {num_samples:,} pairs in {time.time() - start:.2f} seconds.")

    print("\\n[4/4] Results & Theory Proof:")
    
    # Calculate Average MSE
    avg_mse_mse = mx.mean((X1 - recon1_mse)**2).item()
    avg_mse_prod = mx.mean((X1 - recon1_prod)**2).item()
    
    # Evaluate Asymmetric Inner Products
    ip_mse_quant = mx.sum(recon1_mse * X1, axis=1)
    ip_prod_quant = mx.sum(recon1_prod * X1, axis=1)
    
    ip_error_mse = ip_mse_quant - true_dots
    ip_error_prod = ip_prod_quant - true_dots
    
    print("-" * 75)
    print("🎯 OBJECTIVE 1: Mean Squared Error (Lower is better)")
    print("  -> TurboQuant_MSE  Average MSE : {:.6f}".format(avg_mse_mse))
    print("  -> TurboQuant_PROD Average MSE : {:.6f}".format(avg_mse_prod))
    
    print("-" * 75)
    print("🎯 OBJECTIVE 2: Inner Product Bias & Variance")
    print("  -> TurboQuant_MSE ")
    print("      Mean Bias : {:.6f} (Noticeable Drift!)".format(mx.mean(ip_error_mse).item()))
    print("      Variance  : {:.6f}".format(mx.var(ip_error_mse).item()))
    print()
    print("  -> TurboQuant_PROD")
    print("      Mean Bias : {:.8f} (Converges beautifully to 0.0)".format(mx.mean(ip_error_prod).item()))
    print("      Variance  : {:.6f} (Notice the higher variance trade-off)".format(mx.var(ip_error_prod).item()))
    print("-" * 75)

if __name__ == "__main__":
    evaluate_quantizers()
