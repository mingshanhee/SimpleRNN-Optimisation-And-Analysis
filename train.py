#!/usr/bin/env python3
"""
Training pipeline for the SimpleRNN language model using the BSD Japanese-English dataset.
"""

import argparse
from typing import List

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import wandb

from modified_rnn import LM as ModifiedLM
from simple_rnn import LM as SimpleLM

from transformers import AutoTokenizer
from datasets import load_dataset

from tokenizers_utils import load_fast_tokenizer, SRC_TOKEN, TGT_TOKEN
from data import BSDTextDataset, DailyDialogDataset, RottenTomatoesDataset, collate_fn

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
):
    """Train the language model."""
    print(f"Training on device: {device}")
    
    model = model.to(device)
    model.train()
    
    # use cosine annealing learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tokenizer.pad_token_id)  # Ignore padding tokens
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # Process each sequence in the batch separately (since model expects 1D input)
            batch_loss = 0.0
            
            for i in range(batch_size):
                optimizer.zero_grad()
                
                # Get predictions for this sequence
                logits = model(input_ids[i])  # (seq_len, vocab_size)
                
                # Compute loss
                loss = criterion(logits, labels[i])
                
                # Backward pass
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                batch_loss += loss.item()
            
            avg_batch_loss = batch_loss / batch_size
            total_loss += avg_batch_loss
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
        
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")


def train_optimised_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    save_path: str = 'best_model.pt',
    tokenizer = None,
    args = None,
    use_wandb: bool = False,
    wandb_project: str = 'simple-rnn'
):
    """Train the language model and save the best model based on validation loss."""
    print(f"Training on device: {device}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                'hidden_dim': args.hidden_dim if args else None,
                'key_dim': args.key_dim if args else None,
                'value_dim': args.value_dim if args else None,
                'output_dim': args.output_dim if args else None,
                'num_layers': args.num_layers if args else None,
                'batch_size': args.batch_size if args else None,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'vocab_size': len(tokenizer) if tokenizer else None,
                'device': device
            }
        )
        # Watch the model to track gradients and parameters
        wandb.watch(model, log='all', log_freq=100)
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tokenizer.pad_token_id)  # Ignore padding tokens
    
    # Track best validation loss and early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        printed = False
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # Process each sequence in the batch separately (since model expects 1D input)
            batch_loss = 0.0
            
            optimizer.zero_grad()
            
            # Get predictions for this sequence
            logits, _ = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss
            batch_size, seq_len, vocab_size = logits.shape
            # Shift logits and labels for next-token prediction
            # logits[:-1] predicts labels[1:]
            logits_shifted = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
            labels_shifted = labels[:, 1:].contiguous()  # (batch_size, seq_len-1)
            
            logits_flat = logits_shifted.view(-1, vocab_size)  # (batch_size * (seq_len-1), vocab_size)
            labels_flat = labels_shifted.view(-1)  # (batch_size * (seq_len-1),)
            
            # Compute loss
            loss = criterion(logits_flat, labels_flat)
                
            # Backward pass
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

            optimizer.step()
            
            batch_loss += loss.item()
            
            avg_batch_loss = batch_loss / batch_size
            total_loss += avg_batch_loss
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
        
        avg_epoch_loss = total_loss / num_batches
        
        # Evaluate on validation set if provided
        val_loss = evaluate_model(model, validation_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss
            })
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
            
            # Save the best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': args
            }
            
            if tokenizer is not None:
                checkpoint['tokenizer'] = tokenizer
            
            torch.save(checkpoint, save_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f} -> {save_path}")
            
            # Log best validation loss to wandb
            if use_wandb:
                wandb.log({
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch
                })
        else:
            # Increment patience counter if validation loss didn't improve
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{patience}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
                break
        
        model.train()  # Set back to training mode
    
    # Print summary
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {save_path}")

def evaluate_model(model: nn.Module, validation_loader: DataLoader, device: str = 'cpu'):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=validation_loader.dataset.tokenizer.pad_token_id)
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions
            logits, _ = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss
            batch_size, seq_len, vocab_size = logits.shape
            # Shift logits and labels for next-token prediction
            # logits[:-1] predicts labels[1:]
            logits_shifted = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
            labels_shifted = labels[:, 1:].contiguous()  # (batch_size, seq_len-1)
            
            logits_flat = logits_shifted.view(-1, vocab_size)
            labels_flat = labels_shifted.view(-1)
            
            # Compute loss
            loss = criterion(logits_flat, labels_flat)

            avg_batch_loss = loss.item() / batch_size
            total_loss += avg_batch_loss
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def load_best_model(save_path: str, model_class, device: str = 'cpu'):
    """Load the best saved model from checkpoint."""
    checkpoint = torch.load(save_path, map_location=device)
    
    # Extract model arguments from saved args
    args = checkpoint.get('args')
    if args is None:
        raise ValueError("Model arguments not found in checkpoint. Cannot reconstruct model.")
    
    # Reconstruct model
    model = model_class(
        vocab_size=checkpoint.get('vocab_size', args.vocab_size if hasattr(args, 'vocab_size') else None),
        hidden_dim=args.hidden_dim,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Extract other useful information
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0),
        'val_loss': checkpoint.get('val_loss', 0),
        'best_val_loss': checkpoint.get('best_val_loss', 0),
        'tokenizer': checkpoint.get('tokenizer'),
        'args': args
    }
    
    print(f"Loaded model from {save_path}")
    print(f"  - Epoch: {info['epoch']}")
    print(f"  - Train Loss: {info['train_loss']:.4f}")
    print(f"  - Val Loss: {info['val_loss']:.4f}")
    
    return model, info

def main():
    parser = argparse.ArgumentParser(description='Train SimpleRNN language model on BSD dataset')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--key_dim', type=int, default=32, help='Key dimension')
    parser.add_argument('--value_dim', type=int, default=32, help='Value dimension')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Pretrained tokenizer model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset choice', choices=['bsd_ja_en', 'dailydialog', 'rotten_tomatoes'])
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='simple-rnn', help='Weights & Biases project name')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Arguments: {args}")
    print(f"Using device: {device}")

    # Load dataset
    if args.dataset_name == 'bsd_ja_en':
        ds = load_dataset("ryo0634/bsd_ja_en")
        ds_cls = BSDTextDataset
    elif args.dataset_name == 'dailydialog':
        ds = load_dataset("roskoN/dailydialog")
        ds_cls = DailyDialogDataset
    elif args.dataset_name == 'rotten_tomatoes':
        ds = load_dataset("rotten_tomatoes")
        ds_cls = RottenTomatoesDataset

    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
        tokenizer.add_special_tokens({
            'additional_special_tokens': [SRC_TOKEN, TGT_TOKEN]
        })
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Building tokenizer from scratch...")
        # tokenizer = load_fast_tokenizer(ds, "tokenizers/bsd_ja_en_bpe-tokenizer.json")
        # tokenizer = load_fast_tokenizer(ds, "tokenizers/dailydialog_bpe-tokenizer.json")
        tokenizer = load_fast_tokenizer(ds, "tokenizers/rotten_tomatoes_bpe-tokenizer.json")

    # Initialize model
    model = ModifiedLM(
        vocab_size=len(tokenizer),
        hidden_dim=args.hidden_dim,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    train_ds = ds_cls(ds['train'], tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True,
        num_workers=4
    )
    validation_ds = ds_cls(ds['validation'], tokenizer, max_length=args.max_length)
    validation_loader = DataLoader(
        validation_ds, 
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True,
        num_workers=4
    )
    
    # Train model
    train_optimised_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_path=args.save_path,
        tokenizer=tokenizer,
        args=args,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
