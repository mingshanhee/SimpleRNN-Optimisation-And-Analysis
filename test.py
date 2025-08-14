#!/usr/bin/env python3
"""
Training pipeline for the SimpleRNN language model using the BSD Japanese-English dataset.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import wandb
import evaluate
import nltk
from bert_score import score as bert_score

from modified_rnn import LM as ModifiedLM
from simple_rnn import LM as SimpleLM

from datasets import load_dataset

from tokenizers_utils import TGT_TOKEN
from data import BSDTextDataset, DailyDialogDataset, RottenTomatoesDataset, collate_fn

from decode_utils import simple_inference, modified_inference

def load_best_model(save_path: str, model_class, device: str = 'cpu', vocab_size: int = None):
    """Load the best saved model from checkpoint."""
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    
    # Extract model arguments from saved args
    args = checkpoint.get('args')
    if args is None:
        raise ValueError("Model arguments not found in checkpoint. Cannot reconstruct model.")
    
    # Determine vocab size
    if vocab_size is None:
        vocab_size = checkpoint.get('vocab_size')
        if vocab_size is None and hasattr(args, 'vocab_size'):
            vocab_size = args.vocab_size
        elif vocab_size is None:
            # Try to get from tokenizer if available
            tokenizer = checkpoint.get('tokenizer')
            if tokenizer is not None:
                vocab_size = len(tokenizer)
            else:
                raise ValueError("Cannot determine vocab_size from checkpoint.")
    
    # Reconstruct model
    if isinstance(model_class, ModifiedLM):
        model = model_class(
            vocab_size=vocab_size,
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
    else:
        model = model_class(
            vocab_size=vocab_size,
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

def evaluate_model_metrics(model, model_name, dataset_name, test_loader, tokenizer, device='auto', num_samples=100):
    """Evaluate model on test set with BLEU, ROUGE, and BERTScore metrics."""
    print("Evaluating model with BLEU, ROUGE, and BERTScore metrics...")
    
    model.eval()
    
    # Initialize metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    generated_texts = []
    reference_texts = []
    
    sample_count = 0

    ja_token_id = tokenizer.convert_tokens_to_ids(TGT_TOKEN)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
                
            input_ids = batch['input_ids'][0].to(device)
            labels = batch['labels'][0].to(device)
            
            try:
                # Find the position where target translation starts
                matches = (input_ids == ja_token_id).nonzero(as_tuple=True)[0]
                ja_pos = matches[0].item()
                prefix_input = input_ids[:ja_pos+1]
                reference_translation = labels[ja_pos+1:]
            except Exception as e:
                continue
            
            # Generate translation
            if model_name == 'simple':
                generated_ids = simple_inference(
                    model, prefix_input, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id
                )
            else:
                generated_ids = modified_inference(
                    model, prefix_input.unsqueeze(0), max_new_tokens=50, eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract the generated translation part 
            generated_translation = generated_ids[0]
            
            # Remove EOS token if present
            if len(reference_translation) > 0 and reference_translation[-1] == tokenizer.eos_token_id:
                reference_translation = reference_translation[:-1]
            
            # Convert to text
            generated_text = tokenizer.decode(generated_translation, skip_special_tokens=True)
            reference_text = tokenizer.decode(reference_translation, skip_special_tokens=True)
            
            # Only add if both texts are non-empty
            if generated_text.strip() and reference_text.strip():
                generated_texts.append(generated_text.strip())
                reference_texts.append(reference_text.strip())
                sample_count += 1
    
    print(f"Collected {len(generated_texts)} valid prediction pairs")
    
    if len(generated_texts) == 0:
        print("No valid predictions generated!")
        return None
    
    if dataset_name == 'rotten-tomatoes':
        # Calculate accuracy for Rotten Tomatoes dataset
        correct_count = sum(1 for gen, ref in zip(generated_texts, reference_texts) if gen == ref)
        accuracy = correct_count / len(generated_texts)
        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(generated_texts)})")
        
        # Calculate F1 score - convert text labels to numeric
        f1_metric = evaluate.load("f1")
        
        # Convert text labels to numeric (0 for negative, 1 for positive)
        label_to_id = {'negative': 0, 'positive': 1}
        
        # Convert predictions and references to numeric, handling unknown labels
        numeric_predictions = []
        numeric_references = []
        
        for pred, ref in zip(generated_texts, reference_texts):
            if pred.strip().lower() in label_to_id and ref.strip().lower() in label_to_id:
                numeric_predictions.append(label_to_id[pred.strip().lower()])
                numeric_references.append(label_to_id[ref.strip().lower()])
        
        if len(numeric_predictions) > 0:
            f1_results = f1_metric.compute(
                predictions=numeric_predictions,
                references=numeric_references,
                average='macro'
            )
            print(f"F1 Score: {f1_results['f1']:.4f}")
            print(f"Valid F1 samples: {len(numeric_predictions)}/{len(generated_texts)}")
        else:
            print("No valid predictions for F1 calculation")
    
    else:
        # Calculate BLEU score
        print("Calculating BLEU score...")
        bleu_results = bleu_metric.compute(
            predictions=generated_texts,
            references=[[ref] for ref in reference_texts]
        )
        
        # Calculate ROUGE scores
        print("Calculating ROUGE scores...")
        rouge_results = rouge_metric.compute(
            predictions=generated_texts,
            references=reference_texts
        )
        
        # Calculate BERTScore
        print("Calculating BERTScore...")
        bert_precision, bert_recall, bert_f1 = bert_score(
            generated_texts, reference_texts, lang='ja', verbose=False
        )
        
        # Compile results
        results = {
            'bleu': bleu_results['bleu'],
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'bert_precision': bert_precision.mean().item(),
            'bert_recall': bert_recall.mean().item(),
            'bert_f1': bert_f1.mean().item(),
            'num_samples': len(generated_texts)
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of samples evaluated: {results['num_samples']}")
        print(f"BLEU Score: {results['bleu']:.4f}")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print(f"BERTScore Precision: {results['bert_precision']:.4f}")
        print(f"BERTScore Recall: {results['bert_recall']:.4f}")
        print(f"BERTScore F1: {results['bert_f1']:.4f}")
        print("="*50)
        
    # Show some examples
    print("\nSample Predictions:")
    print("-" * 30)
    for i in range(min(5, len(generated_texts))):
        print(f"Reference: {reference_texts[i]}")
        print(f"Generated: {generated_texts[i]}")
        print("-" * 30)

def evaluate_best_model(
    model_path: str,
    model_name: str,
    dataset_name: str,
    max_length: int = 256,
    device: str = 'auto',
    num_samples: int = 100
):
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    """Load the best trained model and evaluate it on test set."""
    print(f"Loading best model from: {model_path}")
    
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load dataset
    # Load dataset
    if dataset_name == 'bsd-ja-en':
        ds = load_dataset("ryo0634/bsd_ja_en")
        ds_cls = BSDTextDataset
    elif dataset_name == 'dailydialog':
        ds = load_dataset("roskoN/dailydialog")
        ds_cls = DailyDialogDataset
    elif dataset_name == 'rotten-tomatoes':
        ds = load_dataset("rotten_tomatoes")
        ds_cls = RottenTomatoesDataset

    
    # Load the saved model first to get tokenizer and args
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    tokenizer = checkpoint.get('tokenizer')

    # Load the best model
    model, info = load_best_model(model_path, SimpleLM if model_name == 'simple' else ModifiedLM, device, len(tokenizer))

    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test dataset and dataloader
    test_ds = ds_cls(ds['test'], tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_ds, 
        batch_size=1,  # Use batch size of 1 for evaluation
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Evaluating {min(num_samples, len(test_ds))} samples...")
    
    # Evaluate model
    evaluate_model_metrics(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        num_samples=num_samples
    )

def main():
    parser = argparse.ArgumentParser(description='Train SimpleRNN language model on BSD dataset')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='simple-rnn', help='Weights & Biases project name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (bsd-ja-en/dailydialog/rotten-tomatoes)', choices=['bsd-ja-en', 'dailydialog', 'rotten-tomatoes'])
    parser.add_argument('--model_name', type=str, required=True, choices=['simple', 'modified'], help='Model type to use')
    
    # Add evaluation mode arguments
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the best saved model instead of training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model for evaluation')
    parser.add_argument('--num_eval_samples', type=int, default=10, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Arguments: {args}")
    print(f"Using device: {device}")

    # Evaluation mode
    print("Running in evaluation mode...")
    evaluate_best_model(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length,
        device=device,
        num_samples=args.num_eval_samples,
        dataset_name=args.dataset_name
    )
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
