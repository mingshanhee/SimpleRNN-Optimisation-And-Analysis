import os

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast

EN_TOKEN = "<en>"
JA_TOKEN = "<ja>"
LANG_TOKENS = [EN_TOKEN, JA_TOKEN]

def _get_corpus(ds: DatasetDict):
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            yield ex["en_sentence"]
            yield ex["ja_sentence"]

def _train_and_save_tokenizer(ds: DatasetDict, tokenizer_name: str):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        special_tokens=["<unk>", "</s>", "<pad>"] + LANG_TOKENS
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(_get_corpus(ds), trainer)
    tokenizer.save(tokenizer_name)

def load_fast_tokenizer(ds: DatasetDict, tokenizer_name: str):
    """Create and return a fast tokenizer from a pre-trained model."""

    # Check if the tokenizer file exists
    if not os.path.exists(tokenizer_name):
        print("Tokenizer file not found. Training a new tokenizer...")
        _train_and_save_tokenizer(ds, tokenizer_name)
    
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_name,
        unk_token="<unk>",
        bos_token=None,
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=LANG_TOKENS,
    )