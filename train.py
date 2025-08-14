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
    model_name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    save_path: str = 'best_model.pt',
    tokenizer = None,
    args = None,
    use_wandb: bool = False,
    wandb_project: str = 'simple-rnn',
    run_name: str = None
):
    """Train the language model and save the best model based on validation loss."""
    print(f"Training on device: {device}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
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
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            if args.model_name == 'simple':
                input_ids = batch['input_ids'].squeeze(0).to(device) # (seq_len)
                labels = batch['labels'].squeeze(0).to(device) # (seq_len)
                
                # Process each sequence in the batch separately (since model expects 1D input)
                batch_loss = 0.0
                
                optimizer.zero_grad()
                
                # Get predictions for this sequence
                logits = model(input_ids) 
                
                # Reshape for CrossEntropyLoss
                _, vocab_size = logits.shape # (seq_len, vocab_size)

                # Shift logits and labels for next-token prediction
                logits_shifted = logits[:-1, :].contiguous()  # (seq_len-1, vocab_size)
                labels_shifted = labels[1:].contiguous()  # (seq_len-1,)
                
                logits_flat = logits_shifted.view(-1, vocab_size)  # (seq_len-1, vocab_size)
                labels_flat = labels_shifted.view(-1)  # (seq_len-1,)
                
                # Compute loss
                loss = criterion(logits_flat, labels_flat)
                    
                # Backward pass
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

            else:
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
        val_loss = evaluate_model(model, model_name, validation_loader, device)
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

def evaluate_model(model: nn.Module, model_name: str, validation_loader: DataLoader, device: str = 'cpu'):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=validation_loader.dataset.tokenizer.pad_token_id)
    
    with torch.no_grad():
        for batch in validation_loader:

            if model_name == 'simple':
                input_ids = batch['input_ids'].squeeze(0).to(device) # (seq_len)
                labels = batch['labels'].squeeze(0).to(device) # (seq_len)
                
                # Get predictions for this sequence
                logits = model(input_ids) 
                
                # Reshape for CrossEntropyLoss
                _, vocab_size = logits.shape # (seq_len, vocab_size)

                # Shift logits and labels for next-token prediction
                logits_shifted = logits[:-1, :].contiguous()  # (seq_len-1, vocab_size)
                labels_shifted = labels[1:].contiguous()  # (seq_len-1,)
                
                logits_flat = logits_shifted.view(-1, vocab_size)  # (seq_len-1, vocab_size)
                labels_flat = labels_shifted.view(-1)  # (seq_len-1,)
                
                # Compute loss
                loss = criterion(logits_flat, labels_flat)

                total_loss += loss.item()
                num_batches += 1

            else:
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

def main():
    parser = argparse.ArgumentParser(description='Train SimpleRNN language model on BSD dataset')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--model_name', type=str, required=True, choices=['simple', 'modified'], help='Model type to use')
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
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset choice', choices=['bsd-ja-en', 'dailydialog', 'rotten-tomatoes'])
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='simple-rnn', help='Weights & Biases project name')

    parser.add_argument('--fused_projection', action='store_true', help='Use fused projection for efficiency')
    parser.add_argument('--use_luong_attention', action='store_true', help='Use Luong attention mechanism')
    parser.add_argument('--luong_score', type=str, default='general', help='Luong attention score type (dot/general)')
    parser.add_argument('--layer_dropout_p', type=float, default=0.1, help='Layer dropout probability')

    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Arguments: {args}")
    print(f"Using device: {device}")

    # assert args.model_name == "simple" and args.batch_size == 1, "Only simple model with batch size 1 is supported for now."
    if args.model_name == "simple":
        print("Using SimpleRNN model with batch size 1 for training.")
        args.batch_size = 1

    # Load dataset
    if args.dataset_name == 'bsd-ja-en':
        ds = load_dataset("ryo0634/bsd_ja_en")
        ds_cls = BSDTextDataset
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else "tokenizers/bsd-ja-en_bpe-tokenizer.json"
    elif args.dataset_name == 'dailydialog':
        ds = load_dataset("roskoN/dailydialog")
        ds_cls = DailyDialogDataset
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else "tokenizers/dailydialog_bpe-tokenizer.json"
    elif args.dataset_name == 'rotten-tomatoes':
        ds = load_dataset("rotten_tomatoes")
        ds_cls = RottenTomatoesDataset
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else "tokenizers/rotten-tomatoes_bpe-tokenizer.json"

    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
        tokenizer.add_special_tokens({
            'additional_special_tokens': [SRC_TOKEN, TGT_TOKEN]
        })
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Building tokenizer from scratch...")
        tokenizer = load_fast_tokenizer(ds, tokenizer_name)

    # Initialize model
    if args.model_name == 'simple':
        model = SimpleLM(
            vocab_size=len(tokenizer),
            hidden_dim=args.hidden_dim,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers
        )
    elif args.model_name == 'modified':
        model = ModifiedLM(
            vocab_size=len(tokenizer),
            hidden_dim=args.hidden_dim,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            fused_projection=args.fused_projection,
            use_luong_attention=args.use_luong_attention,
            luong_score=args.luong_score,
            layer_dropout_p=args.layer_dropout_p
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
    train_model(
        model=model,
        model_name=args.model_name,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_path=args.save_path.format(params=f"{args.num_layers}layers-{args.hidden_dim}hidden-{args.key_dim}key-{args.value_dim}value"),
        tokenizer=tokenizer,
        args=args,
        use_wandb=args.use_wandb,
        wandb_project=args.model_name,
        run_name=f"{args.dataset_name}_{args.num_layers}layers-{args.hidden_dim}hidden-{args.key_dim}key-{args.value_dim}value" 
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
