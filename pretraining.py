import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Dataset # Import Dataset
from architecture import GPTModel  # Your model implementation

# Configuration - Enhanced with better defaults
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.2,  # Increased dropout
    "qkv_bias": False
}

TRAIN_CONFIG = {
    "batch_size": 8,  # Increased batch size
    "learning_rate": 3e-4,  # More stable learning rate
    "weight_decay": 0.2,  # Stronger regularization
    "num_epochs": 15,  # More epochs
    "eval_freq": 100,  # Less frequent evaluation
    "eval_iter": 10,  # More evaluation batches
    "generation_temp": 0.7,  # Temperature for sampling
    "top_k": 50  # Top-k sampling
}

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(text, tokenizer, batch_size, max_length, stride, shuffle=True):
    dataset = TextDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def generate_text(model, tokenizer, device, start_context, max_length=100, temperature=0.7, top_k=50):
    model.eval()
    context_size = GPT_CONFIG["context_length"]
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(encoded[:, -context_size:])[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            encoded = torch.cat((encoded, next_token), dim=1)

    return tokenizer.decode(encoded[0].tolist())

def train_model(model, train_loader, val_loader, optimizer, device):
    model.train()
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Evaluation and logging
            if batch_idx % TRAIN_CONFIG["eval_freq"] == 0:
                val_loss = evaluate(model, val_loader, device, TRAIN_CONFIG["eval_iter"])
                train_losses.append(loss.item())      # Add this
                val_losses.append(val_loss)           # And this

                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")

                print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} | "
                      f"Batch {batch_idx} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f}")

        # Generate sample after epoch
        sample = generate_text(
            model, tiktoken.get_encoding("gpt2"), device,
            start_context="Harry looked at Hermione and",
            temperature=TRAIN_CONFIG["generation_temp"],
            top_k=TRAIN_CONFIG["top_k"]
        )
        print(f"\nSample after Epoch {epoch+1}:\n{sample}\n")

    return train_losses, val_losses

# Define the evaluate function used in train_model
def evaluate(model, data_loader, device, num_batches=5):
    """Evaluate model on data"""
    model.eval()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1))
        total_loss += loss.item()
    return total_loss / num_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # Load and prepare data
    with open("/content/harrypotter_1.txt", "r") as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader = create_dataloader(
        text[:int(0.9*len(text))], tokenizer,
        TRAIN_CONFIG["batch_size"], GPT_CONFIG["context_length"],
        GPT_CONFIG["context_length"]//2
    )
    val_loader = create_dataloader(
        text[int(0.9*len(text)):], tokenizer,
        TRAIN_CONFIG["batch_size"], GPT_CONFIG["context_length"],
        GPT_CONFIG["context_length"], shuffle=False
    )

    # Initialize and train
    model = GPTModel(GPT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"hp_gpt_{timestamp}.pth")

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curve.png")

if __name__ == "__main__":
    main()
