import mlx.core as mx
import numpy as np
import math
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# TURBOQUANT MSE: Optimized for Mean-Squared Error
# ---------------------------------------------------------------------------
# Theory Breakdown:
# Normally, vector quantization struggles with outliers in independent coordinates.
# TurboQuant_mse combats this by applying a random orthogonal rotation matrix to 
# the input vectors. This acts conceptually like a localized Johnson-Lindenstrauss
# transform, "spreading" the variance evenly across all dimensions.
# 
# Once the variance is uniform and the data is normally distributed along 
# any single axis, we can independently quantize each coordinate using a single 
# 1-Dimensional K-Means codebook (precomputed to minimize MSE for random Gaussian data).
# ---------------------------------------------------------------------------

# Custom Metal Shader (Kernel) for 1D K-Means Quantization
# 
# Apple Silicon (M3) runs Metal natively. When we want to find the nearest
# centroid in a codebook (e.g., 16 values for 4-bit) for millions of vector 
# dimensions, broadcasting in Python requires excessive memory allocations.
# Instead, we directly write a `.metal` kernel that keeps the codebook in 
# extremely fast thread-group (local) memory, yielding massive speedups.



class TurboQuantMSE:
    _rotation_cache = {}
    _codebook_cache = {}

    def __init__(self, dim: int, num_bits: int, seed: int = 42):
        """
        Initializes the TurboQuant_mse compressor.
        """
        self.dim = dim
        self.num_bits = num_bits
        self.num_centroids = 1 << num_bits
        
        # 1. Cache/Get rotation matrix
        cache_key_rot = (dim, seed)
        if cache_key_rot not in TurboQuantMSE._rotation_cache:
            np.random.seed(seed)
            G = np.random.randn(dim, dim)
            Q, R = np.linalg.qr(G)
            d = np.diagonal(R)
            phmap = np.diag(d / np.abs(d))
            rot = mx.array(Q @ phmap)
            TurboQuantMSE._rotation_cache[cache_key_rot] = (rot, rot.T)
        
        self.rotation_matrix, self.rotation_matrix_T = TurboQuantMSE._rotation_cache[cache_key_rot]
        
        # 2. Cache/Get codebook (Standard Normal N(0, 1))
        cache_key_cb = (num_bits, seed) # No longer depends on dim
        if cache_key_cb not in TurboQuantMSE._codebook_cache:
            # We train on unit variance because we normalize our weights to unit variance
            simulated_data = np.random.normal(0, 1.0, size=(100000, 1))
            kmeans = KMeans(n_clusters=self.num_centroids, random_state=seed, max_iter=100, n_init=1)
            kmeans.fit(simulated_data)
            centroids = np.sort(kmeans.cluster_centers_.flatten())
            TurboQuantMSE._codebook_cache[cache_key_cb] = mx.array(centroids)
            
        self.codebook = TurboQuantMSE._codebook_cache[cache_key_cb]

    def quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Quantizes an input vector or matrix `x`.
        Returns: (indices, scale)
        """
        # Step 0: Calculate Scale (Dynamic Scaling)
        # We use std-dev to match our Gaussian-optimized codebook
        scale = mx.std(x)
        x_norm = x / (scale + 1e-8)

        # Step 1: Random Rotation
        if len(x_norm.shape) == 2:
            y = mx.matmul(x_norm, self.rotation_matrix)
        else:
            y = mx.matmul(self.rotation_matrix, x_norm)
        
        y_flat = mx.flatten(y)
        grid = (y_flat.size, 1, 1)
        thread_group = (256, 1, 1) 
        
        quantize_src = f"""
        uint elem_idx = thread_position_in_grid.x;
        if (elem_idx >= {y_flat.size}) {{ return; }}
        float val = static_cast<float>(inp[elem_idx]);
        float min_dist = 1e9;
        uint best_idx = 0;
        uint cb_size = {self.codebook.size};
        for (uint i = 0; i < cb_size; ++i) {{
            float c = static_cast<float>(codebook[i]);
            float dist = (val - c) * (val - c);
            if (dist < min_dist) {{
                min_dist = dist;
                best_idx = i;
            }}
        }}
        out[elem_idx] = static_cast<uint32_t>(best_idx);
        """
        
        quantize_op = mx.fast.metal_kernel(
            name="tq_quantize_1d",
            input_names=["inp", "codebook"],
            output_names=["out"],
            source=quantize_src,
            header="",
            ensure_row_contiguous=True
        )
        
        indices_flat = quantize_op(
            inputs=[y_flat, self.codebook], 
            output_shapes=[y_flat.shape],
            output_dtypes=[mx.uint32],
            grid=grid,
            threadgroup=thread_group
        )[0]
        
        indices = mx.reshape(indices_flat, y.shape)
        return indices, scale

    def dequantize(self, indices: mx.array, scale: mx.array = None) -> mx.array:
        """
        Dequantizes indices back to the reconstructed vector/matrix.
        """
        indices_flat = mx.flatten(indices)
        grid = (indices_flat.size, 1, 1)
        thread_group = (256, 1, 1)
        
        dequantize_src = f"""
        uint elem_idx = thread_position_in_grid.x;
        if (elem_idx >= {indices_flat.size}) {{ return; }}
        uint32_t c_idx = static_cast<uint32_t>(indices[elem_idx]);
        out[elem_idx] = static_cast<float>(codebook[c_idx]);
        """
        
        dequantize_op = mx.fast.metal_kernel(
            name="tq_dequantize_1d",
            input_names=["indices", "codebook"],
            output_names=["out"],
            source=dequantize_src,
            header="",
            ensure_row_contiguous=True
        )
        
        y_recon_flat = dequantize_op(
            inputs=[indices_flat, self.codebook],
            output_shapes=[indices_flat.shape],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=thread_group
        )[0]
        
        y_recon = mx.reshape(y_recon_flat, indices.shape)
        
        # Step 3: Inverse Rotation
        if len(y_recon.shape) == 2:
            x_recon = mx.matmul(y_recon, self.rotation_matrix_T)
        else:
            x_recon = mx.matmul(self.rotation_matrix_T, y_recon)
        
        # Step 4: Rescale back to original range
        if scale is not None:
            x_recon = x_recon * scale
            
        return x_recon

# To understand this code optimally simply import and instantiate the class.
