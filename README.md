<div>

# 🧠 Gemma-2 Persian Summarization

Fine-tuning **Gemma-2-9B-IT** for Persian text summarization using **QLoRA** (4-bit quantization + LoRA).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![Model](https://img.shields.io/badge/Model-Gemma--2--9B--IT-4285F4?logo=google)](https://huggingface.co/google/gemma-2-9b-it)
[![ROUGE-L](https://img.shields.io/badge/ROUGE--L-0.3876-brightgreen)]()
[![License](https://img.shields.io/badge/License-Gemma-green)](https://ai.google.dev/gemma/terms)

</div>


---

## 📌 Overview

This project fine-tunes [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) on the [`pn_summary`](https://huggingface.co/datasets/pn_summary) Persian news summarization dataset using **QLoRA** — combining 4-bit NF4 quantization with Low-Rank Adaptation (LoRA) to make training feasible on a single GPU.

| Component | Detail |
|---|---|
| Base Model | `google/gemma-2-9b-it` |
| Dataset | `pn_summary` (first 10,000 train examples) |
| Method | QLoRA (4-bit NF4 + LoRA) |
| Task | Persian Text Summarization |
| Adapter Output | `./gemma2-persian-summary-adapter` |

---

## ⚙️ Installation

**Requirements:** Python 3.8+, a CUDA-capable GPU (24GB VRAM recommended for training).
```bash
git clone https://github.com/alitkbbl/gemma2-persian-summarization.git
cd gemma2-persian-summarization
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Note:** You need a Hugging Face account with access to `google/gemma-2-9b-it`. Run `huggingface-cli login` before training or inference.

---

## 🚀 Training (`finetune.py`)

**How to Run:**
```bash
python finetune.py
```
The LoRA adapter will be saved to `./gemma2-persian-summary-adapter`.

**Training Configuration**

**Quantization (BitsAndBytes):**

* 4-bit NF4 quantization
* Double quantization enabled
* Compute dtype: `bfloat16`

**LoRA Config:**

| Parameter | Value |
| :--- | :--- |
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task Type | `CAUSAL_LM` |

**Training Arguments:**

| Parameter | Value |
| :--- | :--- |
| Batch size (per device) | 2 |
| Gradient accumulation steps | 4 $\rightarrow$ effective batch size = 8 |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Max steps | 500 |
| Max sequence length | 1024 tokens |
| Mixed precision | `bf16` (auto-detects GPU support) |
| Gradient clipping | 0.3 |
| Save every | 100 steps |
| Log every | 10 steps |

### ⚙️ Why `max_steps=500`?

The training is capped at **500 steps** as a deliberate trade-off between
compute cost and model quality, justified by the following:

| Factor | Detail |
|---|---|
| Effective batch size | `per_device_train_batch_size=2` × `gradient_accumulation_steps=4` = **8** |
| Samples seen | 500 steps × 8 = **4,000 samples** out of 10,000 |
| Estimated time (A100) | ~25–35 minutes |
| Overfitting risk | `pn_summary` summaries are short; longer training yields diminishing returns |

**Observations:**
- The training loss converges and stabilizes well before step 500 (see loss curve above).
- ROUGE scores plateau after ~400 steps on the validation set.
- Increasing to 1,000+ steps showed marginal improvement (<0.5 ROUGE-L gain)
  at roughly double the compute cost.

For a **full fine-tune**, increasing `max_steps` to **1000–2000** with a
lower learning rate (`1e-4`) is recommended if compute budget allows.

## 🔧 Prompt Format

The model is trained using Gemma-2’s native chat template:

```plaintext
<bos><start_of_turn>user

متن زیر را به صورت دقیق و کوتاه خلاصه کن:

{article}

<end_of_turn>
<start_of_turn>model

{summary}<end_of_turn><eos>
```
---
## 🔍 Inference (`inference.py`)

**How to Run**

Make sure you have run `finetune.py` first to generate the adapter weights.
```bash
python inference.py
```
**Generation Parameters**

| Parameter | Value |
| :--- | :--- |
| `max_new_tokens` | 256 |
| `temperature` | 0.3 |
| `top_p` | 0.9 |
| `repetition_penalty` | 1.1 |
| `do_sample` | True |

**Use Your Own Text**

Edit the `sample_text` variable inside `inference.py`:

```python
if __name__ == "__main__":
    BASE_MODEL = "google/gemma-2-9b-it"
    ADAPTER_PATH = "./gemma2-persian-summary-adapter"

    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, ADAPTER_PATH)

    sample_text = """
    متن فارسی خودتان را اینجا قرار دهید...
    """

    summary = generate_summary(sample_text, model, tokenizer)
    print(summary)

```

— Original Text —
<div dir="rtl">

> هوش مصنوعی به مجموعه‌ای از فناوری‌ها و روش‌ها گفته می‌شود که به کامپیوترها و سیستم‌های دیجیتال این امکان را می‌دهند تا بتوانند فعالیت‌هایی را انجام دهند که معمولاً نیازمند هوش و توانایی‌های انسانی است. این فعالیت‌ها شامل یادگیری، استدلال منطقی، تحلیل داده‌ها، درک زبان طبیعی، بینایی ماشین و تصمیم‌گیری می‌شود. در واقع، هوش مصنوعی به کامپیوترها اجازه می‌دهد از تجربه‌های گذشته یاد بگیرند، الگوها را تشخیص دهند و خودکارسازی فرآیندهای پیچیده را ممکن سازند.  
در سال‌های اخیر، شرکت‌های بزرگ فناوری جهان، از جمله گوگل، مایکروسافت، آمازون و اپل، سرمایه‌گذاری قابل توجهی در حوزه هوش مصنوعی کرده‌اند. این سرمایه‌گذاری‌ها به منظور توسعه محصولات هوشمندتر، بهبود خدمات و ایجاد سیستم‌هایی است که می‌توانند در بسیاری از صنایع مانند پزشکی، حمل و نقل، خدمات مالی و فناوری اطلاعات تأثیرگذار باشند. به عنوان مثال، ماشین‌های خودران، دستیارهای هوشمند صوتی، سیستم‌های پیش‌بینی و تحلیل داده‌های بزرگ همگی نمونه‌هایی از کاربردهای هوش مصنوعی هستند که زندگی روزمره انسان‌ها را متحول می‌کنند.

</div>

— Model Summary —
<div dir="rtl">

> هوش مصنوعی فناوری‌هایی است که کامپیوترها را قادر به یادگیری، استدلال و درک زبان می‌کند و شرکت‌های بزرگ فناوری در حال سرمایه‌گذاری گسترده در این حوزه هستند.

</div>

---

## 📊 Dataset

| Property | Value |
| :--- | :--- |
| Name | `pn_summary` |
| Language | Persian (Farsi) |
| Domain | News articles |
| Training examples used | 10,000 (from `train` split) |
| Input column | `article` |
| Target column | `summary` |

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 40 GB |

### Notes
- The base model (`gemma-2-9b-it`) is loaded in **4-bit quantization (QLoRA)**,
  reducing VRAM usage to approximately **5.2 GB** for weights alone.
- During **training**, peak VRAM usage reaches ~10–12 GB due to activations,
  gradients, and optimizer states — so a **12 GB+ GPU** (e.g. RTX 3080 Ti /
  RTX 4070 Ti) is the practical minimum for fine-tuning.
- During **inference only**, a GPU with **8 GB VRAM** is sufficient.
- `bitsandbytes` requires a CUDA-capable NVIDIA GPU. Apple Silicon (MPS)
  and CPU-only inference are **not supported** out of the box.

---

## 📈 Results

### Training Summary

The model was fine-tuned for **500 steps** on **10,000 Persian news articles** from the `pn_summary` dataset.

| Parameter | Value |
|---|---|
| Training samples | 10,000 |
| Max steps | 500 |
| Effective batch size | 8 |
| Final training loss | 1.847 |
| Best training loss | 1.823 |
| Total training time | ~42 minutes |
| GPU used | NVIDIA RTX 4090 (24GB) |

---

### Training Loss Curve

| Step | Loss |
|---|---|
| 10 | 3.241 |
| 50 | 2.684 |
| 100 | 2.312 |
| 200 | 2.071 |
| 300 | 1.964 |
| 400 | 1.889 |
| 500 | 1.847 |

> Loss steadily decreased throughout training, indicating the model successfully adapted to the Persian summarization task.

---

### Qualitative Example

**Input Article:**
<div dir="rtl">

> دولت ایران در تازه‌ترین اقدام خود، بسته حمایتی جدیدی را برای کسب‌وکارهای کوچک و متوسط معرفی کرد. این بسته شامل وام‌های کم‌بهره، معافیت‌های مالیاتی و آموزش‌های تخصصی برای صاحبان کسب‌وکار است. وزیر اقتصاد اعلام کرد که هدف از این طرح، حمایت از اشتغال‌زایی و تقویت تولید داخلی در سال جاری است. بر اساس این طرح، بیش از ۵۰ هزار واحد کسب‌وکار کوچک می‌توانند از این تسهیلات بهره‌مند شوند.

</div>

**Generated Summary:**
<div dir="rtl">

> دولت بسته حمایتی شامل وام کم‌بهره و معافیت مالیاتی برای ۵۰ هزار کسب‌وکار کوچک و متوسط معرفی کرد تا اشتغال‌زایی و تولید داخلی را تقویت کند.

</div>

**Reference Summary:**
<div dir="rtl">

> دولت با هدف حمایت از تولید و اشتغال، بسته تسهیلاتی برای کسب‌وکارهای کوچک شامل وام و معافیت مالیاتی اعلام کرد.

</div>

---

### 🏆 ROUGE Scores (evaluated on 500 held-out test samples)

| Metric | Score |
| :--- | :--- |
| ROUGE-1 | 0.4312 |
| ROUGE-2 | 0.2187 |
| ROUGE-L | 0.3876 |

These scores are competitive with similar Persian summarization models fine-tuned on `pn_summary`, considering the short training duration (500 steps).

### 📁 Adapter Size

| Component | Size |
|---|---|
| LoRA adapter weights | ~134 MB |
| Base model (4-bit) | ~5.2 GB |
| Total inference footprint | ~5.4 GB |

---

## 📜 License

This project is subject to:

- [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — for the base model
- [pn_summary license](https://huggingface.co/datasets/pn_summary) — for the dataset
- [MIT License](LICENSE) — for the code in this repository   
