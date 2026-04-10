import mlx.core as mx
import numpy as np
from turbo_quant_mse import TurboQuantMSE

# ---------------------------------------------------------------------------
# TURBOQUANT PROD: Optimized for Unbiased Inner Products
# ---------------------------------------------------------------------------
# Theory Breakdown:
# While `TurboQuantMSE` is great at minimizing reconstruction MSE, it often
# shrinks the magnitude of vectors and introduces a bias when taking the dot
# product between two separate quantized vectors. 
# 
# To fix this, DeepMind proposed a 2-stage "residual" approach:
# Stage 1: Run the MSE quantizer, but give it 1 less bit budget (b - 1 bits).
# Stage 2: Calculate the exact residual error left over: r = x - x_mse.
# Stage 3: We spend our remaining 1 bit on a Quantized Johnson-Lindenstrauss
#          (QJL) transform. By multiplying the residual by a random standard
#          normal matrix S and taking the `sign()`, we capture the direction
#          of the residual error extremely evenly.
#
# Because $E[S^T sign(Sr)] \\propto r$ (due to properties of Gaussians),
# this 1-bit scheme naturally acts as an *unbiased estimator* for the direction!
# ---------------------------------------------------------------------------

class TurboQuantProd:
    _s_cache = {}

    def __init__(self, dim: int, num_bits: int, seed: int = 42):
        """
        Initializes the TurboQuant_prod compressor.
        """
        assert num_bits >= 2, "Need at least 2 bits for TurboQuant Prod"
        self.dim = dim
        self.seed = seed
        
        # Initialize Stage 1 (MSE Quantizer) using b-1 bits
        self.mse_quantizer = TurboQuantMSE(dim=dim, num_bits=num_bits - 1, seed=seed)
        
        # Initialize Stage 2 (QJL Matrix)
        cache_key_s = (dim, seed + 1)
        if cache_key_s not in TurboQuantProd._s_cache:
            np.random.seed(seed + 1)
            s_mat = mx.array(np.random.randn(dim, dim))
            TurboQuantProd._s_cache[cache_key_s] = (s_mat, s_mat.T)
            
        self.S, self.S_T = TurboQuantProd._s_cache[cache_key_s]
        
    def quantize(self, x: mx.array):
        """
        Quantizes an input vector `x`.
        Returns:
            mse_indices (mx.array): Unsigned integer indices for the MSE codebook.
            qjl_signs (mx.array): Int8 array of +/- 1 signs.
            residual_norm (mx.array): A scaler float track for the magnitude.
        """
        # --- STAGE 1 ---
        # Get the b-1 bit MSE approximation indices
        mse_indices, mse_scale = self.mse_quantizer.quantize(x)
        
        # We dequantize it temporarily to calculate our mistake (residual)
        x_mse = self.mse_quantizer.dequantize(mse_indices, mse_scale)
        
        # --- STAGE 2 ---
        # Calculate residual error 
        r = x - x_mse
        
        # Calculate the scalar magnitude (L2 Norm) that we need to save!
        # Summing along axis=0 ensures that if x is a batched matrix (dimension x batch), it norms per vector
        residual_norm = mx.sqrt(mx.sum(r * r, axis=0))
        
        # Perform the 1-bit QJL transform: sign(S * r)
        # 1. Project the residual
        Sr = mx.matmul(self.S, r)
        
        # 2. Extract signs (We store it simply as +1 or -1)
        # Alternatively, boolean 0 and 1, but we'll use int8 for simplicity
        qjl_signs = mx.sign(Sr)
        
        # For simulation, we return the tuple components.
        return mse_indices, mse_scale, qjl_signs, residual_norm
        
    def dequantize(self, mse_indices: mx.array, mse_scale: mx.array, qjl_signs: mx.array, residual_norm: mx.array) -> mx.array:
        """
        Reconstructs the vector.
        """
        # --- RECONSTRUCT STAGE 1 ---
        x_mse_recon = self.mse_quantizer.dequantize(mse_indices, mse_scale)
        
        # --- RECONSTRUCT STAGE 2 ---
        # To invert QJL, we project the signs backwards using S^T
        # The theoretical estimator is proportional to S^T * sign(Sr)
        dir_recon = mx.matmul(self.S_T, qjl_signs)
        
        # By standard QJL math: E[S^T sign(S*r)] = dim * sqrt(2/pi) * (r / ||r||)
        # Therefore, exactly recovering 'r' unbiasedly requires the theoretical scale:
        scale = residual_norm * mx.sqrt(mx.array(np.pi) / 2.0) / self.dim
        
        r_recon = dir_recon * scale
        
        # Final Vector = MSE Component + Recovered QJL Residual
        x_recon = x_mse_recon + r_recon
        return x_recon

