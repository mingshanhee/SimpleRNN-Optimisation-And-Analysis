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

from modified_rnn import LM as ModifiedLM
from simple_rnn import LM as SimpleLM

from transformers import AutoTokenizer
from datasets import load_dataset

from tokenizers_utils import load_fast_tokenizer, EN_TOKEN, JA_TOKEN


class BSDTextDataset(Dataset):
    """Dataset wrapper for BSD Japanese-English corpus."""
    
    def __init__(self, ds_split, tokenizer = None, max_length: int = 512):
        self.ds = ds_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        """Create joint sequence  <en> EN ids <ja> JA ids </s>  for Prefix-LM."""
        en = self.ds[idx]["en_sentence"]
        ja = self.ds[idx]["ja_sentence"]

        src_ids = self.tokenizer.encode(en, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(ja, add_special_tokens=False)

        src_bos = self.tokenizer.convert_tokens_to_ids(EN_TOKEN)
        tgt_bos = self.tokenizer.convert_tokens_to_ids(JA_TOKEN)

        input_ids = [src_bos] + src_ids + [tgt_bos] + tgt_ids + [
            self.tokenizer.eos_token_id
        ]

        trans_pos = input_ids.index(tgt_bos) + 1
        labels = [self.tokenizer.pad_token_id] * trans_pos + input_ids[trans_pos:]

        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch, tokenizer):
    """Custom collate function to pad sequences to equal length."""
    # Extract input_ids and labels from batch
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    # Pad sequences to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

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
    device: str = 'cpu'
):
    """Train the language model."""
    print(f"Training on device: {device}")
    
    model = model.to(device)
    model.train()
    
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
            
            optimizer.zero_grad()
            
            # Get predictions for this sequence
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            labels_flat = labels.view(-1)  # (batch_size * seq_len,)
            
            # Compute loss
            loss = criterion(logits_flat, labels_flat)
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            
            avg_batch_loss = batch_loss / batch_size
            total_loss += avg_batch_loss
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
        
        avg_epoch_loss = total_loss / num_batches
        
        # Evaluate on validation set if provided
        if validation_loader is not None:
            val_loss = evaluate_model(model, validation_loader, device)
            print(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            model.train()  # Set back to training mode
        else:
            print(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f}")

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
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # Compute loss
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train SimpleRNN language model on BSD dataset')
    parser.add_argument('--debug', action='store_true', help='Use debug mode with limited data')
    parser.add_argument('--num_debug_samples', type=int, default=128, help='Number of samples in debug mode')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--key_dim', type=int, default=32, help='Key dimension')
    parser.add_argument('--value_dim', type=int, default=64, help='Value dimension')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Pretrained tokenizer model name')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Arguments: {args}")
    print(f"Using device: {device}")

    # Load dataset
    ds = load_dataset("ryo0634/bsd_ja_en")

    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Building tokenizer from scratch...")
        tokenizer = load_fast_tokenizer(ds, "tokenizers/bsd_ja_en_bpe-tokenizer.json")

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
    train_ds = BSDTextDataset(ds['train'], tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    validation_ds = BSDTextDataset(ds['validation'], tokenizer, max_length=args.max_length)
    validation_loader = DataLoader(
        validation_ds, 
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Train model
    train_optimised_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # print("Training completed!")
    
    # # Save model
    # save_path = 'trained_lm_model.pt'
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'tokenizer': dataset.tokenizer,
    #     'args': args
    # }, save_path)
    # print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
