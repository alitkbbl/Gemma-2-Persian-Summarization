import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model_and_tokenizer(base_model_id, adapter_path):
    """
    Loads the base model in 4-bit quantization and attaches the fine-tuned LoRA adapter.
    """
    print(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" # Automatically uses GPU if available
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    # Load the fine-tuned model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_summary(text, model, tokenizer, max_new_tokens=256):
    """
    Generates a summary for the given Persian text using Gemma-2 chat template.
    """
    # Gemma-2 standard instruction format
    messages = [
        {"role": "user", "content": f"متن زیر را به صورت دقیق و کوتاه خلاصه کن:\n\n{text}"}
    ]
    
    # Apply the model's specific chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3, # Low temperature for more deterministic/factual summaries
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    # Decode the output and extract only the generated response
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return summary.strip()

if __name__ == "__main__":
    # --- Configuration ---
    BASE_MODEL = "google/gemma-2-9b-it"
    ADAPTER_PATH = "./gemma2-persian-summary-adapter" # Directory where your LoRA weights are saved
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, ADAPTER_PATH)
    
    # Sample Persian text for testing
    sample_text = """
    هوش مصنوعی به مجموعه‌ای از فناوری‌ها گفته می‌شود که به کامپیوترها اجازه می‌دهد کارهایی را انجام دهند که معمولاً نیازمند هوش انسانی هستند. 
    این کارها شامل یادگیری، استدلال، حل مسئله و درک زبان طبیعی می‌شود. 
    در سال‌های اخیر، با پیشرفت مدل‌های زبانی بزرگ، هوش مصنوعی توانسته است در تولید متن، ترجمه زبان‌ها و حتی خلاصه‌سازی مقالات طولانی به موفقیت‌های چشمگیری دست یابد. 
    بسیاری از شرکت‌های فناوری اکنون در حال سرمایه‌گذاری عظیمی در این حوزه هستند تا خدمات بهتری به کاربران خود ارائه دهند.
    """
    
    print("\n--- Original Text ---")
    print(sample_text.strip())
    
    print("\n--- Generating Summary ---")
    summary = generate_summary(sample_text, model, tokenizer)
    
    print("\n--- Model Summary ---")
    print(summary)
