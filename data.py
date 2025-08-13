import torch
from torch.utils.data import Dataset
from tokenizers_utils import EN_TOKEN, JA_TOKEN

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
    """Custom collate function to pad sequences to equal length and prepare attention masks."""
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