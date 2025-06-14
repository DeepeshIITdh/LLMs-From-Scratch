# 🧙‍♂️ SLM-GPT: A Tiny GPT Language Model Trained on Harry Potter

This project implements a GPT-2-style small language model (SLM) from scratch using PyTorch. It includes custom implementations of a tokenizer interface, positional & token embeddings, multi-head self-attention, transformer blocks, and causal language modeling loss. The model is pretrained on the entire Harry Potter dataset and supports text generation using top-k/top-p sampling.

![](https://wallpapers.com/images/featured-full/harry-potter-laptop-e7r1xyv92ewc8nt7.jpg)

---

## 📌 Project Highlights

- ✅ Complete GPT-style architecture from scratch (no Hugging Face or Transformers dependencies)
- ✅ Implements multi-head self-attention and masked causal attention
- ✅ Uses OpenAI’s fast tokenizer `tiktoken`
- ✅ Trains on the complete Harry Potter book series
- ✅ Supports generation with top-k, top-p (nucleus), temperature sampling
- ✅ Modular and extendable for downstream fine-tuning tasks

---

## 🧠 Model Architecture

The model mimics a simplified version of GPT-2:
- Token Embedding Layer
- Positional Embedding Layer
- N Transformer Blocks:
  - LayerNorm
  - Causal Multi-Head Attention
  - Residual Connection
  - Feed-Forward Layer
  - Dropout
- Final LayerNorm + Linear Head for vocabulary prediction

### ✅ Causal Masked Attention

Each token only attends to previous tokens using a triangular attention mask, preserving autoregressive property.

## ⚙️ Configuration

### Model Hyperparameters: `GPT_CONFIG`

```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": True
}
```
### Training Hyperparameters: `TRAIN_CONFIG`

```python
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 6e-4,
    "weight_decay": 0.98,
    "num_epochs": 50,
    "eval_freq": 100,
    "eval_iter": 20,
    "generation_temp": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "gradient_clip": 1.0,
    "warmup_steps": 1000,
    "checkpoint_path": "best_model.pth"
}
```
---

## 🧼 Preprocessing

- Removed unwanted newlines and special characters  
- Tokenized text using [`tiktoken`](https://github.com/openai/tiktoken) (same tokenizer used by OpenAI models)  
- Chunked the entire corpus into overlapping sequences of `context_length` for input/target training  

---

## 🧪 Training Pipeline

- Model trained using `AdamW` optimizer with weight decay  
- Cross-entropy loss used between predicted token logits and ground-truth next tokens  
- Early stopping implemented to avoid overfitting  
- Training loss and validation perplexity tracked and plotted  

### 📈 Loss and Perplexity Curve

See `training_metrics.png` to observe how the model improves across training epochs.

---

## 🔮 Text Generation

Once trained, the model can generate Harry Potter-like text using:

```python
generated_text = generate_text(
    model, tokenizer, device,
    start_context="Harry looked at Hermione and",
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
print(generated_text)
```
## 📝 Sample Generated Output

> **Input Prompt**:  
> `Harry looked at Hermione and`

> **Generated Output**:
Harry looked at Hermione and smiled nervously. “Do you think it’ll work?”
Hermione nodded, her eyes scanning the corridor. “It has to. We don’t have another choice.”
The castle walls whispered around them, ancient magic thick in the air.
Behind them, footsteps echoed faintly — someone was coming.

---

## 👨‍💻 Authors

- **Binita Mahto**  
- **Rishith Kumar**  
- **Deepesh Sharma**