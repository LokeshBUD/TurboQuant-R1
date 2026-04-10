import mlx.core as mx
from mlx_lm import load, generate
import time

def test_model(model_path, prompt, label):
    print(f"\n{'='*20} TESTING: {label} {'='*20}")
    print(f"Path: {model_path}")
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load(model_path)
        
        # Prepare the prompt
        # DeepSeek-R1 usually prefers a specific format
        formatted_prompt = f"<|User|>{prompt}<|Assistant|><|thought|>\n"
        
        start_time = time.time()
        
        # Generate response
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=1024,
            verbose=False
        )
        
        duration = time.time() - start_time
        
        print(f"\n--- RESPONSE ---\n{response}")
        print(f"\n--- METRICS ---")
        print(f"Time taken: {duration:.2f}s")
        
        # Improved reasoning check
        reasoning_tokens = ["<think>", "</think>", "<|step-by-step explanation|>"]
        if any(token in response for token in reasoning_tokens):
            print("✅ Reasoning detected.")
        else:
            print("❌ No reasoning tags found - Model might be losing coherence.")
            
    except Exception as e:
        print(f"❌ Error testing {label}: {e}")

if __name__ == "__main__":
    MODELS = [
        ("./R1-1.5B-Raw", "16-BIT (RAW FP16)"),
        ("./R1-1.5B-Turbo-8bit", "8-BIT (TURBOQUANT)"),
        ("./R1-1.5B-Turbo-4bit", "4-BIT (TURBOQUANT)"),
        ("./R1-1.5B-Official-4bit", "4-BIT (OFFICIAL AWQ)"),
        ("./R1-1.5B-Naive", "4-BIT (NAIVE QUANT)")
    ]
    
    TEST_CASES = [
        "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
        "I have 5 apples. I give 2 to Steve. Steve gives 1 back. How many do I have?",
        "A man is looking at a portrait. He says: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the portrait?"
    ]
    
    print("🧠 Starting Multi-Factor Intelligence Comparison...")
    
    for prompt in TEST_CASES:
        print(f"\n\n{'#'*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'#'*80}")
        
        for path, label in MODELS:
            test_model(path, prompt, label)
            mx.metal.clear_cache()
