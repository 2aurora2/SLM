import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm

from model import GPT
from dataset import MyDataset

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

torch.manual_seed(42)

@dataclass
class GPTConfig:
    max_seq: int = 512
    batch_size: int = 16
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 768
    hidden_dim: int = n_embd  # For tie_embedding_weight
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    vocab_size: int = 50257   # GPT-2's vocabulary size

    epochs: int = 10
    lr: float = 3e-4
    t_max: int = 1000

def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Create progress bar for training
    progress_bar = tqdm(enumerate(dataloader), 
                        total=num_batches,
                        desc=f"Epoch {epoch+1}/{GPTConfig.epochs} - Training",
                        bar_format="{l_bar}{bar:20}{r_bar}")
    
    for batch_idx, (ids, target) in progress_bar:
        ids, target = ids.to(device), target.to(device)
        
        optimizer.zero_grad()
        _, loss = model(ids, target)

        # add aggregation operation: convert multi-GPU loss to scalar
        if loss.dim() > 0: 
            loss = loss.mean() 

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar with current metrics
        progress_bar.set_postfix({
            "Batch": f"{batch_idx+1}/{num_batches}",
            "Loss": f"{loss.item():.4f}",
            "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
            "LR": f"{current_lr:.2e}"
        })
    
    return total_loss / num_batches

def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Create progress bar for evaluation
    progress_bar = tqdm(enumerate(dataloader), 
                        total=num_batches,
                        desc=f"Epoch {epoch+1}/{GPTConfig.epochs} - Validation",
                        bar_format="{l_bar}{bar:20}{r_bar}")
    
    with torch.no_grad():
        for batch_idx, (ids, target) in progress_bar:
            ids, target = ids.to(device), target.to(device)
            logits, loss = model(ids, target)

            # add aggregation operation: convert multi-GPU loss to scalar
            if loss.dim() > 0: 
                loss = loss.mean() 

            total_loss += loss.item()
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "Batch": f"{batch_idx+1}/{num_batches}",
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss/(batch_idx+1):.4f}"
            })
    
    return total_loss / num_batches

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    config = GPTConfig()
    model = GPT(config)
    model.to(device)

    # DataParallel
    if torch.cuda.device_count() >= 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)
    else:
        print("No GPUs available, using CPU for training.")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.t_max
    )

    # Load dataset
    print("Loading dataset...")
    train_dataset = MyDataset(dataset="wikitext", type="train", max_len=config.max_seq)
    valid_dataset = MyDataset(dataset="wikitext", type="validation", max_len=config.max_seq)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params / 1e6:.2f}M total parameters")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Training loop
    print("\nStarting training process...")
    for epoch in range(config.epochs):
        # Training phase
        train_loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Validation phase
        val_loss = evaluate(model, valid_loader, device, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Training Loss:   {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_{epoch + 1}.pt")
        print(f"  Checkpoint saved: checkpoints/checkpoint_{epoch + 1}.pt\n")

    print("Training completed successfully!")

if __name__ == "__main__":
    main()