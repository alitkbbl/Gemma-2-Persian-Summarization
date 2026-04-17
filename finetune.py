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
# Loading only a subset (e.g., train split) for demonstration
dataset = load_dataset(DATASET_ID, split="train[:5000]") 

# --- 3. Format Prompt ---
def formatting_prompts_func(example):
    """
    Formats the input dataset into the Gemma-2 instruction format.
    """
    output_texts = []
    for i in range(len(example['article'])):
        text = example['article'][i]
        summary = example['summary'][i]
        
        # Using the standard instruction format
        prompt = f"<bos><start_of_turn>user\nمتن زیر را به صورت دقیق و کوتاه خلاصه کن:\n\n{text}<end_of_turn>\n<start_of_turn>model\n{summary}<end_of_turn><eos>"
        output_texts.append(prompt)
    return output_texts

# --- 4. Load Tokenizer and Model in 4-bit ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = 'right' # Recommended for Gemma

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

# --- 5. LoRA Configuration ---
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=16, # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target all linear layers for better fine-tuning
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 6. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # Suitable for 24GB VRAM
    gradient_accumulation_steps=4, # Effective batch size = 8
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True, # bfloat16 is highly recommended for newer GPUs like RTX 4090
    max_grad_norm=0.3,
    max_steps=500, # Set a limit for quick demonstration
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# --- 7. Initialize Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024, # Maximum sequence length
    tokenizer=tokenizer,
    args=training_args,
)

# --- 8. Start Training ---
print("Starting training...")
trainer.train()

# --- 9. Save Model ---
print(f"Saving final adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
