import os
import time
import mlx.core as mx
from mlx_lm import load, generate

MODELS = {
    "16-BIT (RAW FP16)": "./R1-1.5B-Raw",
    "8-BIT (TURBOQUANT)": "./R1-1.5B-Turbo-8bit",
    "4-BIT (TURBOQUANT)": "./R1-1.5B-Turbo-4bit",
    "4-BIT (OFFICIAL AWQ)": "./R1-1.5B-Official-4bit"
}

def analyze_response(response, duration, tokenizer):
    """Tier 2 & 3 Analysis Logic"""
    tokens_generated = len(tokenizer.encode(response))
    tps = tokens_generated / duration if duration > 0 else 0
    
    # 1. Tag Integrity
    has_close_tag = "</think>" in response
    
    # 2. Repetition Monitor (Unique Token Density)
    think_content = response.split("</think>")[0] if has_close_tag else response
    tokens = think_content.split()
    unique_tokens = set(tokens)
    
    # A true "Death Loop" prints the same few words over hundreds of tokens.
    # A concise, successful answer might naturally have a density ~0.20
    density = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
    repetition_trap = density < 0.10 
    
    return {
        "tokens_sec": tps,
        "has_close_tag": has_close_tag,
        "density": density,
        "repetition_trap": repetition_trap,
        "response": response
    }

def run_tier_1_weights():
    print("\n" + "="*50)
    print("🔬 TIER 1: STATISTICAL VERIFICATION (Weight-Level)")
    print("="*50)
    
    raw_path = os.path.join("./R1-1.5B-Raw", "model.safetensors")
    turbo_path = os.path.join("./R1-1.5B-Turbo-4bit", "model.safetensors")
    
    if not os.path.exists(raw_path) or not os.path.exists(turbo_path):
        print("⚠️ Raw or Turbo model.safetensors not found locally. Skipping weight stats.")
        return
        
    raw_weights = mx.load(raw_path)
    turbo_weights = mx.load(turbo_path)
    
    # Sample a middle and deep layer to demonstrate stability
    target_keys = [
        "model.layers.12.self_attn.v_proj.weight",
        "model.layers.24.self_attn.q_proj.weight"
    ]
    
    for key in target_keys:
        if key in raw_weights and key in turbo_weights:
            w_r = raw_weights[key].astype(mx.float32)
            w_t = turbo_weights[key].astype(mx.float32)
            
            # Mean error bias
            diff = w_r - w_t
            mean_bias = mx.mean(diff).item()
            
            # Cosine Similarity
            w_r_flat = mx.flatten(w_r)
            w_t_flat = mx.flatten(w_t)
            cos_sim = mx.sum(w_r_flat * w_t_flat) / (mx.sqrt(mx.sum(w_r_flat**2)) * mx.sqrt(mx.sum(w_t_flat**2)))
            
            print(f"\nEvaluating Layer: {key}")
            print(f" ➔ Mean Bias (Expect ~0)       : {mean_bias:.8f}")
            print(f" ➔ Cosine Sim (Expect >0.95)   : {cos_sim.item():.4f}")

def run_tier_2_and_3():
    print("\n" + "="*50)
    print("🧪 TIER 2 & 3: LOGIC STRESS-TEST & HARDWARE PERF")
    print("="*50)
    
    prompt = "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have? Explain your reasoning step-by-step."
    formatted_prompt = f"<|User|>{prompt}<|Assistant|><|thought|>\n"
    
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"\n⚠️ Skipping {name}: Path {path} not found.")
            continue
            
        print(f"\nLoading {name} ...")
        try:
            model, tokenizer = load(path)
            
            start_time = time.time()
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=600,
                verbose=False
            )
            duration = time.time() - start_time
            
            metrics = analyze_response(response, duration, tokenizer)
            
            print(f"\n📊 {name} METRICS:")
            print(f" ➔ Hardware Perf : {metrics['tokens_sec']:.2f} Tokens / Sec")
            print(f" ➔ Tag Integrity : {'✅ Closed' if metrics['has_close_tag'] else '❌ Broken/Missing'}")
            print(f" ➔ Vocab Density : {metrics['density']:.2f} (Unique/Total ratio)")
            print(f" ➔ Death Loop    : {'🚨 DETECTED' if metrics['repetition_trap'] else '✅ Clean'}")
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
        
        # Clear GPU memory between runs
        mx.metal.clear_cache()

if __name__ == "__main__":
    print("Initializing Comprehensive Validation Suite...\n")
    run_tier_1_weights()
    run_tier_2_and_3()
    print("\n✅ Verification Complete.")
