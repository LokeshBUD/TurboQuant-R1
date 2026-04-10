# 🧠 TurboQuant: Preserving DeepSeek-R1's Reasoning Under 4-Bit Compression

**Why 4-bit quantization breaks reasoning in DeepSeek-R1 and how to fix it with Unbiased Estimators.**

This repository serves as a technical proof-of-concept demonstrating how standard matrix quantization techniques (like AWQ) destroy the complex logical chains inside reasoning LLMs, and how applying **Random Orthogonal Rotations** with **Unbiased Estimators** preserves true reasoning capabilities under extreme compression on Apple Silicon (M3).

---

## 🚨 The Problem: "CoT Collapse" and the Logic Death Loop
Standard LLMs generally survive naive integer rounding because small errors in single tokens self-correct. However, advanced Reasoning Models (like DeepSeek-R1) rely on a delicate, non-linear Chain-of-Thought (CoT). Standard quantization scales destruct the few "Heavy Hitter" outlier weights that act as logic gates. 

When the logic chain breaks, the model enters a **Logic Death Loop**, repeating its previous thoughts infinitely and totally forgetting its goal. We call this **CoT Collapse.**

## 📐 The Math: The Unbiased Residual QJL Estimator
Following the theoretical principles outlined in Zandieh et al. (2025), replacing simple scaling with an orthogonal transform protects these logic gates.

1. **TurboQuant-MSE (Structure):** A fully dense Random Orthogonal Rotation ($\Pi$) obtained via QR-decomposition is applied to the weight tensor. This structurally "smears" the logic-critical outliers uniformly across all coordinates, perfectly matching a Gaussian distribution so we can apply an optimal 1D Lloyd-Max scalar quantizer.
2. **Residual Bias (Intelligence):** Traditional MSE quantizers shift the mean ($\mathbb{E}[E] \neq 0$). By applying a Quantized Johnson-Lindenstrauss (QJL) stage to the residual, TurboQuant computes an **unbiased estimator**, ensuring that the internal activation signals don't drift away from their intended logic paths.

## 📊 Empirical Benchmarks (Apple Silicon M3)

We ran an automated validation suite (`scripts/validation_suite.py`) testing Layer-Wise Statistics and Automated Programmatic Logic Checks against the raw 1.5B distillation reference:

| Model | Hardware TPS | Mean Bias (Error) | Cosine Sim | CoT "Death Loop" State |
| :--- | :--- | :--- | :--- | :--- |
| **RAW (FP16 Baseline)** | 25.24 | 0.0 | 1.000 | 🟢 Clean |
| **TURBOQUANT (8-bit)** | 23.94 | $\approx$ -0.000007 | 0.999 | 🟢 Clean |
| **TURBOQUANT (4-bit)** | 20.14 | $\approx$ 0.000001 | 0.994 | 🟢 Clean |
| **Official MLX AWQ (4-bit)**| 89.92* | _Drifts_ | 0.910 | 🔴 DETECTED (Trapped) |

*\*Note: AWQ's native TPS is explicitly due to standard 4-bit integer packing. TurboQuant's TPS represents the offline Dense Rotation computation requirement for preserving the $N(0,1)$ mathematical bounds required to pass the reasoning tests.*

## 💡 The "Aha!" Prompt validation
To explicitly witness the "Death Loop" boundary, we prompt the models with a non-linear logic puzzle:
> _"Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have? Explain your reasoning step-by-step."_

**Official AWQ (Failed):**
Gets stuck looping endlessly through the relationship tree without assigning identities, often repeating: *"Sally has 2 sisters, her brothers have 2 sisters, so..."* until OOM.

**TurboQuant (Passed):**
Maintains the necessary signal fidelity to close the relationship tree, accurately deducing that Sally herself is one of the sisters, and cleanly halting reasoning with the correct answer (1 sister).

---

## 📁 Repository Structure
*   `/kernels`: Custom Apple Silicon Metal Shader logic for Lloyd-Max quantizers and QJL estimators.
*   `/scripts`: The automated logic tests, metrics monitors, and quantization wrappers.
*   `/docs`: Internal Research Engineer architecture summaries detailing the Memory vs Compute paradox. 

## 🚀 How to Run Locally

You must execute scripts from the root directory to ensure paths to data folders connect correctly.
1. `python scripts/quantize_comparison.py` to generate the mathematically reconstructed models.
2. `python scripts/validation_suite.py` or `python scripts/test_r1_intelligence.py` to run the automated benchmarking monitors.
