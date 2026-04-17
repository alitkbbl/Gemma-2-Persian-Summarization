import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Configuration ---
MODEL_ID = "google/gemma-2-9b-it"
DATASET_ID = "pn_summary"
OUTPUT_DIR = "./gemma2-persian-summary-adapter"

# --- 2. Load Dataset ---
print(f"Loading dataset {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split="train[:10000]")

# FIX: verify column names exist before training
print(f"Dataset columns: {dataset.column_names}")
assert "article" in dataset.column_names, "Column 'article' not found in dataset!"
assert "summary" in dataset.column_names, "Column 'summary' not found in dataset!"

# --- 3. Load Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"  # Recommended for Gemma

# --- 4. Format Prompt ---
# FIX: use apply_chat_template instead of hardcoded string
# to stay consistent with inference.py and avoid format mismatch
def formatting_prompts_func(example):
    """
    Formats dataset into Gemma-2 instruction format using the tokenizer's
    chat template — consistent with inference.py.
    """
    output_texts = []
    for i in range(len(example["article"])):
        text = example["article"][i]
        summary = example["summary"][i]

        messages = [
            {"role": "user", "content": f"متن زیر را به صورت دقیق و کوتاه خلاصه کن:\n\n{text}"},
            {"role": "assistant", "content": summary}
        ]

        # apply_chat_template handles BOS/EOS and turn tokens correctly
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        output_texts.append(prompt)
    return output_texts

# --- 5. Load Model in 4-bit ---
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# --- 6. LoRA Configuration ---
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 7. Training Arguments ---
# FIX: detect bf16 support at runtime instead of hardcoding True
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 8
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=not bf16_supported,   # fallback to fp16 if bf16 not available
    bf16=bf16_supported,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# --- 8. Initialize Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# --- 9. Start Training ---
print("Starting training...")
trainer.train()

# --- 10. Save Model ---
print(f"Saving final adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
